#!/usr/bin/env bash
# Run at runtime in vllm/vllm-openai:v0.26.0. Never install the serving stack.
set -euo pipefail
umask 077

: "${REPOSITORY_URL:?Supply the actual GitHub HTTPS repository URL}"
: "${REPOSITORY_REVISION:?Supply the full repository commit to deploy}"
: "${AWQ_SOURCE_REVISION:?Supply the immutable source-model revision used for AWQ}"
: "${NETWORK_VOLUME_ID:?Supply the Network Volume ID attached in RunPod Console}"
[[ "$REPOSITORY_URL" =~ ^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(\.git)?$ ]] || { echo "Use the actual GitHub HTTPS repository URL." >&2; exit 1; }
[[ "$REPOSITORY_REVISION" =~ ^[a-f0-9]{40}$ && "$AWQ_SOURCE_REVISION" =~ ^[a-f0-9]{40}$ ]] || { echo "Repository/model revisions must be full commit SHAs." >&2; exit 1; }

REPO_DIR=/workspace/qwen-vllm-production
MODEL_ROOT=/workspace/models/qwen3-14b
AWQ_DIR="$MODEL_ROOT/awq_w4a16"
STAGING_DIR="$MODEL_ROOT/.awq_w4a16.download"
MODEL_VARIANT="${MODEL_VARIANT:-awq_w4a16}"
RUNTIME_PYTHON="${RUNTIME_PYTHON:-$(command -v python3)}"
VLLM_BIN="$(command -v vllm)"
[[ -x "$RUNTIME_PYTHON" && -x "$VLLM_BIN" ]] || { echo "Run this in the official vLLM image, without activating another environment." >&2; exit 1; }
export PATH="$(dirname "$VLLM_BIN"):$PATH"
export PYTHONNOUSERSITE=1

"$RUNTIME_PYTHON" - <<'PY'
import os
import sys
if sys.version_info < (3, 11):
    raise SystemExit("Python >= 3.11 is required.")
if not os.path.ismount("/workspace") or not os.access("/workspace", os.W_OK):
    raise SystemExit("Attach the Network Volume at writable /workspace before starting.")
PY

# Install only missing OS utilities, never Python/CUDA/torch/vLLM packages.
install_os_tools() {
  [[ "$(id -u)" == 0 ]] || { echo "Missing utilities require the official image's default root user." >&2; exit 1; }
  apt-get update
  apt-get install -y --no-install-recommends "$@"
}
missing_tools=()
command -v git >/dev/null || missing_tools+=(git)
command -v flock >/dev/null || missing_tools+=(util-linux)
command -v curl >/dev/null || missing_tools+=(curl ca-certificates)
if (( ${#missing_tools[@]} )); then install_os_tools "${missing_tools[@]}"; fi

mkdir -p "$MODEL_ROOT"
exec 9>"$MODEL_ROOT/.bootstrap.lock"
flock -n 9 || { echo "Another bootstrap is preparing this volume." >&2; exit 1; }
exec 8>"$MODEL_ROOT/.serving.lock"
flock -n 8 || { echo "Serving already holds this volume; stop it before another bootstrap." >&2; exit 1; }

# Check the image's actual packages. A mismatch stops; no pip repair is attempted.
"$RUNTIME_PYTHON" - <<'PY'
from importlib.metadata import version
from packaging.version import Version
import torch

expected = {"vllm": "0.26.0", "compressed-tensors": "0.17.0", "torch": "2.11.0"}
for package, wanted in expected.items():
    actual = version(package)
    print(f"image: {package}={actual}")
    if actual.split("+", 1)[0] != wanted:
        raise SystemExit(f"Image package mismatch: expected {package} {wanted}; do not replace the serving stack.")
transformers_version = version("transformers")
print(f"image: transformers={transformers_version}")
if Version(transformers_version) < Version("5.5.3"):
    raise SystemExit(f"Image package mismatch: expected transformers >=5.5.3, found {transformers_version}; do not replace the serving stack.")
if torch.version.cuda != "13.0":
    raise SystemExit(f"Expected the validated CUDA 13.0 wheel, found {torch.version.cuda}")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("Exactly one visible CUDA GPU is required.")
gpu = torch.cuda.get_device_properties(0)
if "A100" not in gpu.name or "SXM" not in gpu.name or gpu.total_memory < 79 * 1024**3:
    raise SystemExit(f"Expected A100 SXM 80GB, found {gpu.name}")
PY

# Clone a real commit once; preserve an existing checkout and any local changes.
if [[ ! -e "$REPO_DIR" ]]; then
  repo_staging=/workspace/.qwen-vllm-production.clone
  [[ ! -e "$repo_staging" ]] || { echo "An incomplete clone exists; inspect it before retrying." >&2; exit 1; }
  git clone --no-checkout --depth 1 "$REPOSITORY_URL" "$repo_staging"
  git -C "$repo_staging" fetch --depth 1 origin "$REPOSITORY_REVISION"
  git -C "$repo_staging" checkout --detach "$REPOSITORY_REVISION"
  mv -T "$repo_staging" "$REPO_DIR"
fi
[[ -d "$REPO_DIR/.git" ]] || { echo "Existing repository is not a Git checkout; preserve/migrate it explicitly." >&2; exit 1; }
[[ "$(git -C "$REPO_DIR" rev-parse HEAD)" == "$REPOSITORY_REVISION" ]] || { echo "Existing repository revision differs; no automatic checkout/reset is performed." >&2; exit 1; }
git -C "$REPO_DIR" diff --quiet || { echo "Tracked local modifications must be reviewed before deployment." >&2; exit 1; }
git -C "$REPO_DIR" diff --cached --quiet || { echo "Staged modifications must be reviewed before deployment." >&2; exit 1; }
[[ -f "$REPO_DIR/uv.lock" && -f "$REPO_DIR/scripts/serve_vllm.py" ]] || { echo "Repository snapshot is incomplete." >&2; exit 1; }

RESULT_DIR="$REPO_DIR/results/serving/runpod"
BENCHMARK_ENV="$REPO_DIR/.venv-benchmark"
BENCHMARK_PYTHON="$BENCHMARK_ENV/bin/python"
mkdir -p "$RESULT_DIR"
export PYTHONPATH="$REPO_DIR/src"
export HF_HOME="$REPO_DIR/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

# A dedicated client venv reads the image's core packages but writes only locally.
if [[ ! -e "$BENCHMARK_ENV" ]]; then
  "$RUNTIME_PYTHON" -m venv --system-site-packages "$BENCHMARK_ENV"
fi
[[ -x "$BENCHMARK_PYTHON" && -f "$BENCHMARK_ENV/pyvenv.cfg" ]] || { echo "Incomplete benchmark venv; inspect it explicitly." >&2; exit 1; }
"$BENCHMARK_PYTHON" - <<'PY'
import sys
from pathlib import Path
if sys.prefix == sys.base_prefix:
    raise SystemExit("Client installation must target a venv.")
if "include-system-site-packages = true" not in (Path(sys.prefix) / "pyvenv.cfg").read_text():
    raise SystemExit("Client venv must reuse the official image's core packages.")
PY

# Use only the small HTTP/YAML dependency closure from the existing uv.lock.
# No project extras, uv sync, dependency resolver, or core-stack installation.
"$RUNTIME_PYTHON" - "$REPO_DIR/uv.lock" "$RESULT_DIR/benchmark-requirements.txt" <<'PY'
import sys
import tomllib
from pathlib import Path

lock = tomllib.loads(Path(sys.argv[1]).read_text())
allowed = {"httpx", "httpcore", "anyio", "certifi", "idna", "h11", "sniffio", "typing-extensions", "pyyaml"}
by_name = {}
for package in lock["package"]:
    by_name.setdefault(package["name"], []).append(package)
pending, selected = ["httpx", "pyyaml"], {}
while pending:
    name = pending.pop()
    if name in selected:
        continue
    if name not in allowed or len(by_name.get(name, [])) != 1:
        raise SystemExit(f"Unexpected client dependency in uv.lock: {name}")
    package = by_name[name][0]
    selected[name] = package["version"]
    pending.extend(dependency["name"] for dependency in package.get("dependencies", []))
Path(sys.argv[2]).write_text("".join(f"{name}=={selected[name]}\n" for name in sorted(selected)), encoding="utf-8")
PY
"$BENCHMARK_PYTHON" -m pip install --disable-pip-version-check --no-deps \
  --only-binary=:all: --requirement "$RESULT_DIR/benchmark-requirements.txt"
"$BENCHMARK_PYTHON" - "$RESULT_DIR/runtime-packages.json" <<'PY'
import json
import sys
from importlib.metadata import version
from pathlib import Path
from packaging.version import Version
expected = {"vllm": "0.26.0", "compressed-tensors": "0.17.0", "torch": "2.11.0"}
for package, wanted in expected.items():
    if version(package).split("+", 1)[0] != wanted:
        raise SystemExit(f"Client venv changed the visible core version: {package}")
transformers_version = version("transformers")
if Version(transformers_version) < Version("5.5.3"):
    raise SystemExit(f"Client venv requires transformers >=5.5.3, found {transformers_version}; do not replace the serving stack.")
packages = {name: version(name) for name in (*expected, "transformers", "httpx", "PyYAML")}
Path(sys.argv[1]).write_text(json.dumps(packages, indent=2) + "\n", encoding="utf-8")
PY

validate_checkpoint() {
  "$RUNTIME_PYTHON" - "$1" "$2" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
config = json.loads((root / "config.json").read_text())
if config.get("model_type") != "qwen3":
    raise SystemExit("Checkpoint must be Qwen3.")
for name in ("tokenizer.json", "tokenizer_config.json"):
    if not (root / name).is_file():
        raise SystemExit(f"Missing local tokenizer file: {name}")
quantization = config.get("quantization_config")
if sys.argv[2] == "awq":
    if not isinstance(quantization, dict) or quantization.get("quant_method") not in ("awq", "compressed-tensors"):
        raise SystemExit("Missing AWQ/compressed-tensors metadata.")
elif quantization:
    raise SystemExit("BF16 baseline must not contain quantization metadata.")
index = root / "model.safetensors.index.json"
weights = set(json.loads(index.read_text())["weight_map"].values()) if index.is_file() else {p.name for p in root.glob("*.safetensors")}
if not weights:
    raise SystemExit("No safetensors weights found.")
for relative in weights:
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"Missing, empty, or invalid weight shard: {relative}")
PY
}

if [[ ! -e "$AWQ_DIR" ]]; then
  : "${AWQ_GDRIVE_SOURCE:?First copy requires a Google Drive rclone remote:folder}"
  : "${RCLONE_CONFIG:?Supply the private rclone config file}"
  [[ -r "$RCLONE_CONFIG" ]] || { echo "Unreadable rclone credential file." >&2; exit 1; }
  if ! command -v rclone >/dev/null; then
    command -v unzip >/dev/null || install_os_tools unzip
    [[ "$(id -u)" == 0 ]] || { echo "Install rclone explicitly when not using the image's default root user." >&2; exit 1; }
    curl --fail --location https://rclone.org/install.sh --output /tmp/qwen-rclone-install.sh
    bash /tmp/qwen-rclone-install.sh
  fi
  "$RUNTIME_PYTHON" - "$RCLONE_CONFIG" "$AWQ_GDRIVE_SOURCE" <<'PY'
import configparser
import sys
config = configparser.ConfigParser(interpolation=None)
config.read(sys.argv[1])
remote, separator, folder = sys.argv[2].partition(":")
if not separator or not folder or any(c in sys.argv[2] for c in "\r\n") or config.get(remote, "type", fallback="") != "drive":
    raise SystemExit("Use a configured type=drive rclone remote:folder.")
PY
  transfer_identity="$AWQ_SOURCE_REVISION"$'\n'"$AWQ_GDRIVE_SOURCE"
  identity_file="$MODEL_ROOT/.awq_w4a16.transfer-source"
  if [[ -e "$identity_file" ]]; then
    [[ "$(cat "$identity_file")" == "$transfer_identity" ]] || { echo "Staging belongs to another checkpoint; inspect it explicitly." >&2; exit 1; }
  else
    [[ ! -e "$STAGING_DIR" ]] || { echo "Unidentified checkpoint staging exists." >&2; exit 1; }
    printf '%s\n' "$transfer_identity" > "$identity_file"
  fi
  mkdir -p "$STAGING_DIR"
  rclone version > "$RESULT_DIR/rclone-version.txt"
  rclone copy "$AWQ_GDRIVE_SOURCE" "$STAGING_DIR" --config "$RCLONE_CONFIG" --checksum --transfers 4
  rclone check "$AWQ_GDRIVE_SOURCE" "$STAGING_DIR" --config "$RCLONE_CONFIG"
  validate_checkpoint "$STAGING_DIR" awq
  mv -T "$STAGING_DIR" "$AWQ_DIR"
fi
# Existing AWQ skips rclone installation, credentials, and all Drive access.
validate_checkpoint "$AWQ_DIR" awq
source_marker="$MODEL_ROOT/.awq_w4a16.source-revision"
if [[ -f "$MODEL_ROOT/.awq_w4a16.transfer-source" ]]; then
  [[ "$(head -n 1 "$MODEL_ROOT/.awq_w4a16.transfer-source")" == "$AWQ_SOURCE_REVISION" ]] || { echo "Downloaded AWQ source revision mismatch." >&2; exit 1; }
fi
if [[ -e "$source_marker" ]]; then
  [[ "$(cat "$source_marker")" == "$AWQ_SOURCE_REVISION" ]] || { echo "Persistent AWQ source revision mismatch." >&2; exit 1; }
else
  # Operator declaration, not proof of model provenance.
  printf '%s\n' "$AWQ_SOURCE_REVISION" > "$source_marker"
fi

case "$MODEL_VARIANT" in
  awq_w4a16) export MODEL_PATH="$AWQ_DIR" MODEL_REVISION="" ;;
  bf16)
    export MODEL_PATH="$MODEL_ROOT/bf16"
    validate_checkpoint "$MODEL_PATH" bf16
    [[ -f "$MODEL_PATH/.source-revision" && "$(cat "$MODEL_PATH/.source-revision")" == "$AWQ_SOURCE_REVISION" ]] || { echo "BF16 must have the same verified source revision as AWQ." >&2; exit 1; }
    export MODEL_REVISION="$AWQ_SOURCE_REVISION"
    ;;
  *) echo "MODEL_VARIANT must be awq_w4a16 or bf16." >&2; exit 1 ;;
esac

export SERVED_MODEL_NAME=qwen3-14b MAX_MODEL_LEN=8192 GPU_MEMORY_UTILIZATION=0.90
export TENSOR_PARALLEL_SIZE=1 DTYPE=bfloat16 SEED=42 HOST=0.0.0.0 PORT=8000
export BENCHMARK_BASE_URL=http://127.0.0.1:8000
# Source this file in a second GPU Pod terminal; it does not activate the client venv.
{
  printf 'export BENCHMARK_PYTHON=%q\n' "$BENCHMARK_PYTHON"
  printf 'export PROJECT_PYTHON=%q\n' "$BENCHMARK_PYTHON"
  printf 'export PYTHONPATH=%q\n' "$PYTHONPATH"
  printf 'export HF_HOME=%q\n' "$HF_HOME"
  printf 'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0\n'
  printf 'export AWQ_SOURCE_REVISION=%q\n' "$AWQ_SOURCE_REVISION"
  printf 'export BENCHMARK_BASE_URL=http://127.0.0.1:8000\n'
} > "$RESULT_DIR/client-env.sh"
flock -u 9
cd "$REPO_DIR"
echo "Client setup is ready at $RESULT_DIR/client-env.sh; use localhost on this GPU Pod."
# PATH points to the official image's CLI, never the client venv.
exec "$RUNTIME_PYTHON" "$REPO_DIR/scripts/serve_vllm.py"
