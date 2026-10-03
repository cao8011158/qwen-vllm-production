#!/usr/bin/env bash
# Run manually inside the GPU Pod, or bake this as the image's ENTRYPOINT.
# No benchmark, dependency installation, or quality evaluation is automatic.
set -euo pipefail
umask 077

: "${DEPLOYMENT_REVISION:?Supply the actual repository commit used in the image}"
: "${AWQ_SOURCE_REVISION:?Supply the immutable Hugging Face revision used to produce AWQ}"
: "${NETWORK_VOLUME_ID:?Supply the attached Network Volume ID from Terraform}"
[[ "$DEPLOYMENT_REVISION" =~ ^[a-f0-9]{40}$ ]] || { echo "Invalid repository revision." >&2; exit 1; }
[[ "$AWQ_SOURCE_REVISION" =~ ^[a-f0-9]{40}$ ]] || { echo "Invalid AWQ source revision." >&2; exit 1; }

REPO_SEED_DIR="${REPO_SEED_DIR:-/opt/qwen-vllm-production}"
REPO_DIR=/workspace/qwen-vllm-production
MODEL_ROOT=/workspace/models/qwen3-14b
AWQ_DIR="$MODEL_ROOT/awq_w4a16"
STAGING_DIR="$MODEL_ROOT/.awq_w4a16.download"
PROJECT_PYTHON="${PROJECT_PYTHON:-/opt/venv/bin/python}"
MODEL_VARIANT="${MODEL_VARIANT:-awq_w4a16}"

command -v mountpoint >/dev/null
command -v flock >/dev/null
mountpoint -q /workspace || { echo "/workspace is not a separate mounted volume; refusing to copy models." >&2; exit 1; }
[[ -w /workspace ]] || { echo "/workspace must be writable." >&2; exit 1; }
[[ -x "$PROJECT_PYTHON" ]] || { echo "The image must contain the validated Python environment." >&2; exit 1; }
python_bin_dir="$(dirname "$PROJECT_PYTHON")"
[[ -x "$python_bin_dir/vllm" ]] || { echo "The validated Python environment must include its vllm executable." >&2; exit 1; }
export PATH="$python_bin_dir:$PATH"
[[ -f "$REPO_SEED_DIR/.deployment-revision" ]] || { echo "Image is missing its repository revision marker." >&2; exit 1; }
[[ "$(cat "$REPO_SEED_DIR/.deployment-revision")" == "$DEPLOYMENT_REVISION" ]] || { echo "Image repository revision mismatch." >&2; exit 1; }

mkdir -p "$MODEL_ROOT"
exec 9>"$MODEL_ROOT/.bootstrap.lock"
flock -n 9 || { echo "Another bootstrap is modifying this volume." >&2; exit 1; }
# Hold this lock through the existing wrapper's exec. A second bootstrap
# must exit before probing the GPU or changing any model/repository files.
exec 8>"$MODEL_ROOT/.serving.lock"
flock -n 8 || { echo "A serving process already holds this checkpoint volume." >&2; exit 1; }

# Seed once; never pull, reset, or overwrite a persistent checkout.
if [[ ! -e "$REPO_DIR" ]]; then
  repo_staging=/workspace/.qwen-vllm-production.seed
  mkdir -p "$repo_staging"
  cp -a "$REPO_SEED_DIR/." "$repo_staging/"
  mv -T "$repo_staging" "$REPO_DIR"
fi
[[ -f "$REPO_DIR/.deployment-revision" ]] || { echo "Persistent checkout needs an operator-verified revision marker." >&2; exit 1; }
[[ "$(cat "$REPO_DIR/.deployment-revision")" == "$DEPLOYMENT_REVISION" ]] || { echo "Persistent checkout differs from the image; update it explicitly before bootstrap." >&2; exit 1; }
[[ -f "$REPO_DIR/uv.lock" && -f "$REPO_DIR/scripts/serve_vllm.py" ]] || { echo "Incomplete repository snapshot." >&2; exit 1; }

export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$REPO_DIR/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

# Runtime preflight only when the operator starts this script.
"$PROJECT_PYTHON" - <<'PY'
from importlib.metadata import version
import sys
import torch

if sys.version_info < (3, 11):
    raise SystemExit("Python >= 3.11 is required.")
expected = {"vllm": "0.26.0", "compressed-tensors": "0.17.0", "torch": "2.11.0", "transformers": "5.17.0"}
for package, wanted in expected.items():
    actual = version(package)
    if actual.split("+", 1)[0] != wanted:
        raise SystemExit(f"{package}: expected {wanted}, found {actual}")
    print(f"{package}={actual}")
if torch.version.cuda != "13.0":
    raise SystemExit(f"Expected the validated CUDA 13.0 wheel, found {torch.version.cuda}")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("Exactly one visible CUDA GPU is required.")
gpu = torch.cuda.get_device_properties(0)
if "A100" not in gpu.name or "SXM" not in gpu.name or gpu.total_memory < 79 * 1024**3:
    raise SystemExit(f"Expected A100 SXM 80GB, found {gpu.name}")
print(f"GPU={gpu.name}")
PY

validate_checkpoint() {
  "$PROJECT_PYTHON" - "$1" "$2" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
quantized = sys.argv[2] == "awq"
config = json.loads((root / "config.json").read_text())
if config.get("model_type") != "qwen3":
    raise SystemExit("Checkpoint must be Qwen3.")
for name in ("tokenizer.json", "tokenizer_config.json"):
    if not (root / name).is_file():
        raise SystemExit(f"Missing local tokenizer file: {name}")
quantization = config.get("quantization_config")
if quantized and (not isinstance(quantization, dict) or quantization.get("quant_method") not in ("awq", "compressed-tensors")):
    raise SystemExit("Checkpoint has no supported AWQ/compressed-tensors metadata.")
if not quantized and quantization:
    raise SystemExit("BF16 baseline must not contain quantization metadata.")
index = root / "model.safetensors.index.json"
if index.is_file():
    weights = set(json.loads(index.read_text())["weight_map"].values())
else:
    weights = {path.name for path in root.glob("*.safetensors")}
if not weights:
    raise SystemExit("No safetensors model weights found.")
for relative in weights:
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"Missing, empty, or invalid weight shard: {relative}")
PY
}

if [[ ! -e "$AWQ_DIR" ]]; then
  : "${AWQ_GDRIVE_SOURCE:?First bootstrap requires the Google Drive rclone remote:folder}"
  : "${RCLONE_CONFIG:?Provide the credential file outside Terraform and the repository}"
  command -v rclone >/dev/null
  [[ -r "$RCLONE_CONFIG" ]] || { echo "rclone credential file is unreadable." >&2; exit 1; }
  "$PROJECT_PYTHON" - "$RCLONE_CONFIG" "$AWQ_GDRIVE_SOURCE" <<'PY'
import configparser
import sys

config = configparser.ConfigParser(interpolation=None)
config.read(sys.argv[1])
remote, separator, folder = sys.argv[2].partition(":")
if not separator or not folder or any(c in sys.argv[2] for c in "\r\n"):
    raise SystemExit("Use a configured Google Drive remote:folder.")
if config.get(remote, "type", fallback="") != "drive":
    raise SystemExit("The source must be a Google Drive rclone remote.")
PY
  transfer_identity="$AWQ_SOURCE_REVISION"$'\n'"$AWQ_GDRIVE_SOURCE"
  identity_file="$MODEL_ROOT/.awq_w4a16.transfer-source"
  if [[ -e "$identity_file" ]]; then
    [[ "$(cat "$identity_file")" == "$transfer_identity" ]] || { echo "Staging belongs to another checkpoint; inspect it explicitly." >&2; exit 1; }
  else
    [[ ! -e "$STAGING_DIR" ]] || { echo "Unidentified staging directory exists; inspect it explicitly." >&2; exit 1; }
    printf '%s\n' "$transfer_identity" > "$identity_file"
  fi
  mkdir -p "$STAGING_DIR"
  # copy/check read Drive; no sync or delete operation is used.
  rclone copy "$AWQ_GDRIVE_SOURCE" "$STAGING_DIR" --config "$RCLONE_CONFIG" --checksum --transfers 4
  rclone check "$AWQ_GDRIVE_SOURCE" "$STAGING_DIR" --config "$RCLONE_CONFIG"
  validate_checkpoint "$STAGING_DIR" awq
  mv -T "$STAGING_DIR" "$AWQ_DIR"
fi
validate_checkpoint "$AWQ_DIR" awq

source_marker="$MODEL_ROOT/.awq_w4a16.source-revision"
if [[ -f "$MODEL_ROOT/.awq_w4a16.transfer-source" ]]; then
  transfer_revision="$(head -n 1 "$MODEL_ROOT/.awq_w4a16.transfer-source")"
  [[ "$transfer_revision" == "$AWQ_SOURCE_REVISION" ]] || { echo "Downloaded AWQ source revision mismatch." >&2; exit 1; }
fi
if [[ -e "$source_marker" ]]; then
  [[ "$(cat "$source_marker")" == "$AWQ_SOURCE_REVISION" ]] || { echo "Persistent AWQ source revision mismatch." >&2; exit 1; }
else
  # This records the operator's declaration; it does not infer a source SHA.
  printf '%s\n' "$AWQ_SOURCE_REVISION" > "$source_marker"
fi

case "$MODEL_VARIANT" in
  awq_w4a16)
    export MODEL_PATH="$AWQ_DIR"
    export MODEL_REVISION=""
    ;;
  bf16)
    export MODEL_PATH="$MODEL_ROOT/bf16"
    [[ -d "$MODEL_PATH" ]] || { echo "Pre-stage the matching BF16 snapshot at $MODEL_PATH." >&2; exit 1; }
    validate_checkpoint "$MODEL_PATH" bf16
    [[ -f "$MODEL_PATH/.source-revision" ]] || { echo "BF16 snapshot needs its operator-verified .source-revision marker." >&2; exit 1; }
    [[ "$(cat "$MODEL_PATH/.source-revision")" == "$AWQ_SOURCE_REVISION" ]] || { echo "BF16 must match the AWQ source revision." >&2; exit 1; }
    export MODEL_REVISION="$AWQ_SOURCE_REVISION"
    ;;
  *) echo "MODEL_VARIANT must be awq_w4a16 or bf16." >&2; exit 1 ;;
esac

export SERVED_MODEL_NAME=qwen3-14b
export MAX_MODEL_LEN=8192
export GPU_MEMORY_UTILIZATION=0.90
export TENSOR_PARALLEL_SIZE=1
export DTYPE=bfloat16
export SEED=42
export HOST=0.0.0.0
export PORT=8000

mkdir -p "$REPO_DIR/results/serving/runpod"
flock -u 9
cd "$REPO_DIR"
echo "Starting existing serving wrapper; run the benchmark in a second terminal on this GPU Pod using 127.0.0.1:8000."
exec "$PROJECT_PYTHON" "$REPO_DIR/scripts/serve_vllm.py"
