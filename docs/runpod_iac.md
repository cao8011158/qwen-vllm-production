# RunPod：官方镜像、Network Volume IaC 与运行时配置

本实现采用：

~~~text
official image = 软件运行环境
Terraform      = Network Volume IaC
RunPod Console = 手工创建三个 Pod
runtime config = 连接三个服务
~~~

默认 Terraform 执行路径为 Local VS Code → GitHub → Google Colab → private Google Drive state。
详见 [Google Colab Terraform workflow](#google-colab-terraform-workflow)。

本次只修改代码/文档并进行静态 review。没有执行命令、安装软件、运行镜像、
Terraform、测试、Git 操作或任何资源/API 写入。下面全部是操作者后续手工步骤。

## 1. 镜像与 Provider 边界

三个 Pod 直接使用以下 tag，不构建 Dockerfile 或 custom image：

| Pod | 官方 image | compute | HTTP port |
| --- | --- | --- | --- |
| GPU | vllm/vllm-openai:v0.26.0 | 1 x NVIDIA A100 SXM 80GB，Secure Cloud | 8000 |
| Prometheus | prom/prometheus:v3.5.0 | CPU-only | 9090 |
| Grafana | grafana/grafana:12.1.0 | CPU-only | 3000 |

已经核实的官方 Provider 1.0.8 保持不变。Terraform 只声明真实
runpod_network_volume，默认 name=qwen-vllm-production、size=100 GB，
data_center_id 由操作者提供，prevent_destroy=true。
三个 Pod 均在 Console 创建；Terraform 的 ID、URL、启动 JSON 与环境输出只是
配置约定，不会创建、修改、检查或删除这些 Pod。

上次核实的最新稳定版为 1.0.9，但其 Configure 存在客户端类型不匹配；
1.0.8 的 Pod Create 未完整传递 GPU selection/ports/startup，CPU compute 字段
也不能可靠表达。继续固定 1.0.8，仅使用其 Network Volume resource。

官方证据：

- [1.0.8 Volume schema](https://github.com/runpod/terraform-provider-runpod/blob/v1.0.8/internal/provider/resource_network_volume/network_volume_resource_gen.go)
- [1.0.8 Volume implementation](https://github.com/runpod/terraform-provider-runpod/blob/v1.0.8/internal/provider/resource_network_volume/network_volume_resource.go)
- [1.0.8 Pod implementation](https://github.com/runpod/terraform-provider-runpod/blob/v1.0.8/internal/provider/resource_pod/pod_resource.go)
- [1.0.9 Provider Configure](https://github.com/runpod/terraform-provider-runpod/blob/v1.0.9/main.go)
- [1.0.9 Pod Configure](https://github.com/runpod/terraform-provider-runpod/blob/v1.0.9/internal/provider/resource_pod/pod_resource.go)

Network Volume 必须在 GPU Pod 创建时附加，位于相同 data center，挂载 /workspace。
已有 volume 可通过 network_volume_id 复用；不要直接把 Terraform-managed volume
切换为 external 模式，应先审阅 state 迁移。
参见 [RunPod Network Volumes](https://docs.runpod.io/storage/network-volumes)。

## 2. 官方 runtime 与 Console 启动机制

检查的是具体 tag 的官方源码：

- [vLLM v0.26.0 Dockerfile](https://github.com/vllm-project/vllm/blob/v0.26.0/docker/Dockerfile)：
  最终 vllm-openai 入口为 vllm serve；Python 在 PATH 中，最终镜像安装 Python venv
  支持与 curl，默认用户 root。bootstrap 从 PATH 发现 Python/vllm，不硬编码私有环境。
- [Prometheus v3.5.0 Dockerfile](https://github.com/prometheus/prometheus/blob/v3.5.0/Dockerfile)：
  BusyBox base、/bin/prometheus、USER nobody、默认 /prometheus 数据目录。
  不要求 Python、Git 或项目 repository。
- [Grafana v12.1.0 Dockerfile](https://github.com/grafana/grafana/blob/v12.1.0/Dockerfile) 与
  [官方 /run.sh](https://github.com/grafana/grafana/blob/v12.1.0/packaging/docker/run.sh)：
  保留官方入口，由 GF_* 环境变量配置服务；datasource/dashboard 通过外部 API client 导入。

RunPod 官方 Console 的 **Container Start Command** 支持 JSON：
~~~json
{"entrypoint": ["/bin/sh", "-c"], "cmd": ["实际要执行的完整 shell 命令"]}
~~~
参见 [Manage Pod templates](https://docs.runpod.io/pods/templates/manage-templates)。

必须粘贴完整 JSON，不能仅把 shell 字符串追加给 vllm serve 或 /bin/prometheus。
这是 Console 的手工配置，不是本项目 Provider 的自动 startup 注入。
Template 在这里仅记录官方 image 和运行参数，不构建任何 image。

镜像 tag 固定不代表所有 Python dependency 的精确版本都已得到运行时确认：
[vLLM 官方 requirements](https://github.com/vllm-project/vllm/blob/v0.26.0/requirements/common.txt)
对 transformers 使用下限。GPU bootstrap 将核对项目已经验证的四个版本；
若不一致则停止，不通过 pip 替换官方 serving runtime。本次没有运行镜像，不能声称
该 tag 的实际包、GPU driver 或 Console 集成已经验证通过。

## 3. 网络与存储

~~~text
GPU Pod
  vLLM :8000
  benchmark client -> http://127.0.0.1:8000
  /workspace <- 100 GB Network Volume
       |
       | https://<GPU_ID>-8000.proxy.runpod.net/metrics
       v
Prometheus CPU Pod :9090
       |
       | https://<PROMETHEUS_ID>-9090.proxy.runpod.net
       v
Grafana CPU Pod :3000
~~~

正式 TTFT/TPOT/E2E 不经过 RunPod proxy。benchmark 必须在 GPU Pod 内运行，
三个 Pod ID 必须不同。GPU proxy 仅作为监控来源。

GPU 持久目录：

~~~text
/workspace/
├── qwen-vllm-production/           # runtime Git clone
│   ├── .venv-benchmark/            # 独立 client venv
│   └── results/serving/runpod/
├── models/
│   └── qwen3-14b/
│       ├── awq_w4a16/
│       └── bf16/                  # 同源 BF16，操作者预先准备
└── .secrets/rclone.conf            # 私密运行时文件，不在 repository 内
~~~

两个 CPU Pod 的最小配置为 volume disk=0、不挂载 GPU Network Volume，
使用各自 container disk 中 image 已经准备好权限的 /prometheus 与 /var/lib/grafana。
不要将 root-owned 的空卷直接盖到这些 non-root 数据目录上。
CPU 数据不是本方案的独立持久存储；Pod 重建/容器数据丢失后需重新导入配置和 dashboard。

## 4. Terraform：源码、state 与职责

默认推荐工作流为：**本地 VS Code 编写 IaC → 手工 commit/push 到 GitHub →
Google Colab 执行 Terraform → private Google Drive 保存 state → Console 创建三个 Pod**。
完整 notebook 示例见下一章 Google Colab Terraform workflow。
本次没有执行任何 Git 或 Terraform 命令。

首次复制 terraform.tfvars.example 到私有 terraform.tfvars；已有文件则编辑。
旧配置中的 pod_images、deployment_revision 已移除，改为 repository_url、
repository_revision。三个 image tag 已在输出中固定，无需提供 image digest。
repository_revision 应是包含当前代码的已发布完整 commit；首次只创建 volume 时，
repository 与模型相关输入都可以先保持 null。

**Terraform DOES：**

- 创建/管理 Network Volume。
- 在 Terraform state 中跟踪该 Network Volume。
- 根据输入推导 volume ID、Pod proxy URLs 和部署配置 outputs。

**Terraform DOES NOT：**

- 创建 GPU Pod、Prometheus Pod 或 Grafana Pod。
- 启动 vLLM。
- 运行 benchmark。
- 从 Google Drive 复制模型。

Google Drive → GPU Network Volume 的模型复制仍由 scripts/bootstrap_gpu.sh
在 GPU Pod runtime 中通过 rclone 完成；这部分实现未改变。
Colab 的 Drive mount 用于 Terraform state，与 GPU Pod 的 /workspace 模型卷是两种
独立存储用途。

填写三个真实 Pod ID 后，使用同一份 Drive state 手工 plan/apply 更新 outputs；
不涉及 Pod 管理。之后可读取原有配置输出：

~~~bash
%%bash
set -euo pipefail
cd /content/qwen-vllm-production
terraform -chdir=infra/terraform output -json deployment_contract
terraform -chdir=infra/terraform output -json bootstrap_environment
terraform -chdir=infra/terraform output -raw gpu_start_command
terraform -chdir=infra/terraform output -json configuration_client_environment
~~~

模型卷仍有 prevent_destroy；Terraform 不负责停止三个外部 Pod，卷保留会继续计费。

## Google Colab Terraform workflow

### 执行顺序与前提

~~~text
Local VS Code
  -> GitHub（由操作者手工 commit/push）
  -> Google Colab
  -> mount Google Drive
  -> load RUNPOD_API_KEY from Colab Secrets
  -> clone/pull repository
  -> terraform init（local backend path 指向 Drive）
  -> terraform validate
  -> terraform plan（审阅仅创建一个 Network Volume）
  -> terraform apply
  -> manually create 3 RunPod Pods
~~~

用户提供的 main 起点为 60a11a5f7de9ef59a88701df8f1c64a5659ed838。
本补丁仍需由操作者提交/推送；Colab 应获取包含 backend 与 ignore 修改的新提交，
不要把上述起点当作已包含本补丁的 revision。

以下示例假定 Colab 已手工准备符合 versions.tf 的 Terraform CLI（>=1.5、<2）。
示例不安装 Terraform，也不运行 bootstrap、模型或 benchmark。
每个以 %%bash 开头的 block 是一个 Colab code cell；其命令仅在操作者运行 cell 时执行。
任一步失败应停止，不跳过格式、验证或 plan 审阅直接 apply。

只使用默认 Terraform workspace，并由一个 Colab session 操作这份 state。
不要让本地和多个 notebook 同时写它；Google Drive persistence 不等于远程 SaaS backend。
如果已经创建过受管理的 volume，先找回并备份原 state，再迁移到下面的 Drive path。
不要因 /content 或旧 state 丢失而重新 apply 空 state 来创建第二个卷。

### 1. 挂载 Google Drive

~~~python
from google.colab import drive
drive.mount("/content/drive")
~~~

确认 Drive 挂载成功；若失败则停止，不能把未挂载的 /content/drive 当成普通临时目录使用。

### 2. 从 Colab Secrets 读取 API key

在 Colab Secrets 中添加 RUNPOD_API_KEY，并授权当前 notebook 访问：

~~~python
from google.colab import userdata
import os

os.environ["RUNPOD_API_KEY"] = userdata.get("RUNPOD_API_KEY")
print("RUNPOD_API_KEY loaded:", bool(os.environ.get("RUNPOD_API_KEY")))
~~~

只打印是否已加载，不打印 key 内容，不显示整个 os.environ，也不将 key 写入 tfvars。
Python kernel 设置的环境变量会供后续 notebook 启动的 Terraform 进程使用。

### 3. Clone repository；已有 checkout 则 pull

~~~bash
%%bash
set -euo pipefail
cd /content
if [ -d qwen-vllm-production/.git ]; then
  cd qwen-vllm-production
  git pull --ff-only
elif [ -e qwen-vllm-production ]; then
  echo "Existing directory is not a Git checkout; inspect it before continuing." >&2
  exit 1
else
  git clone https://github.com/cao8011158/qwen-vllm-production.git
  cd qwen-vllm-production
fi
~~~

后续每个 shell cell 都显式 cd 到 /content/qwen-vllm-production，
不依赖上一个 %%bash cell 的临时 shell 工作目录。

### 4. 创建持久 state 目录

~~~bash
%%bash
set -euo pipefail
[ -d /content/drive/MyDrive ]
mkdir -p /content/drive/MyDrive/qwen-vllm-production/terraform
~~~

state 的唯一目标路径为：

~~~text
/content/drive/MyDrive/qwen-vllm-production/terraform/terraform.tfstate
~~~

Colab /content 是临时文件系统。state 不能仅保存在 checkout 下或其他 /content 临时目录。
下一次 Colab session 应重新挂载同一个 Drive，并复用同一路径；不要换成空的新 state。

### 5. 准备私有 tfvars

仅在文件不存在时复制 example；已有文件直接编辑，不覆盖已有部署参数：

~~~bash
%%bash
set -euo pipefail
cd /content/qwen-vllm-production
if [ ! -e infra/terraform/terraform.tfvars ]; then
  cp infra/terraform/terraform.tfvars.example \
     infra/terraform/terraform.tfvars
fi
~~~

terraform.tfvars 是私有本地文件，不提交 Git，也不要加入 notebook 的公开输出。
第一次新建 Network Volume 所需的最小内容：

~~~hcl
data_center_id = "REAL_RUNPOD_DATA_CENTER_ID"

network_volume_id      = null
network_volume_size_gb = 100

gpu_pod_id        = null
prometheus_pod_id = null
grafana_pod_id    = null

repository_url      = null
repository_revision = null
awq_source_revision = null
awq_gdrive_source   = null
~~~

REAL_RUNPOD_DATA_CENTER_ID 是占位符，必须替换为所选真实 RunPod data center ID。
不要把真实 API key 放进 tfvars。新 session 中该文件可能需要重新从私有记录准备，
而不是靠 GitHub 存储私有变量。

### 6. Init：在运行时指定 backend path

Terraform source 只声明 backend "local" {}；Google Drive 路径不写进任何 .tf 文件。
从 repository 根目录执行的准确命令为：

~~~bash
%%bash
set -euo pipefail
cd /content/qwen-vllm-production
terraform -chdir=infra/terraform init \
  -backend-config="path=/content/drive/MyDrive/qwen-vllm-production/terraform/terraform.tfstate"
~~~

这一步使用 local backend，把 state 指向 private Drive 文件，而非临时 /content checkout。
如果 init 提示已有 backend/state 需要迁移，先备份和核对旧 state，不盲目更换配置或丢弃它。

### 7. Formatting 与 validation

~~~bash
%%bash
set -euo pipefail
cd /content/qwen-vllm-production
terraform -chdir=infra/terraform fmt -check
terraform -chdir=infra/terraform validate
~~~

未通过 fmt -check 或 validate 时停止，在本地修正、审阅和提交源码后再继续。
本次没有实际运行这些检查，不能声称已通过 Terraform CLI validation。

### 8. Plan：只允许一个 Network Volume

~~~bash
%%bash
set -euo pipefail
cd /content/qwen-vllm-production
terraform -chdir=infra/terraform plan \
  -out=runpod.tfplan
~~~

在没有现有受管理资源、network_volume_id=null 的第一次正常 plan 中，预期：

~~~text
Plan: 1 to add, 0 to change, 0 to destroy.

唯一新增 resource：
runpod_network_volume.workspace[0]
~~~

**不应出现任何 Pod resource。** 若有 Pod、任何删除动作（destroy > 0）、
unexpected replacement、多个 managed resources 或其他非预期变更，则停止，
不执行 apply。outputs 的派生值不是新增云 resource。

再次使用已存在的同一份 state 时，不应要求再次创建同一个 volume。
如果预期复用旧卷却仍显示新增，应先确认 state 是否正确，不能照搬“首次 1 to add”的结论。

-chdir 使保存的计划位于 infra/terraform/runpod.tfplan。
plan 文件仍在临时 checkout 中且必须保持私有；runtime 丢失后重新 plan 并审阅，
不要把旧计划当成新的批准。

### 9. Apply：人工审阅后单独执行

~~~bash
%%bash
set -euo pipefail
cd /content/qwen-vllm-production
terraform -chdir=infra/terraform apply runpod.tfplan
~~~

这是操作者确认计划后手工执行的资源创建步骤；没有自动 approve。
本次未执行 apply，也没有创建任何真实资源。

### 10. 输出 Network Volume ID

~~~bash
%%bash
set -euo pipefail
cd /content/qwen-vllm-production
terraform -chdir=infra/terraform output network_volume_id
~~~

取得 ID 后再按第 5 节手工创建三个官方 image Pod，
并在 GPU Pod 创建时把这个 Network Volume 挂载到 /workspace。

### State、secrets 与 lock file 安全

| 内容 | 保存位置 |
| --- | --- |
| Terraform source | GitHub repository |
| Terraform state | private Google Drive path |
| secrets | Colab Secrets / Pod runtime secrets |

terraform.tfstate 可能包含 resource IDs 和基础设施信息，不应提交 GitHub。
Drive 保存 state 是为了跨 Colab session 保留资源跟踪；state backup 同样保持私有。
不要公开分享保存 state 的 Drive 文件、私有 tfvars、plan、credential 或 notebook secret 输出。
.gitignore 保护这些本地文件，但不会自动清除已被 Git 跟踪的文件或历史。

首次 terraform init 会生成 infra/terraform/.terraform.lock.hcl。
它锁定当前官方 runpod/runpod 1.0.8 的解析版本与校验信息，
**应当审阅并提交 Git**，可先下载回本地 VS Code 再提交，供以后 clone 使用。
本补丁没有伪造或预先生成该文件。

不要提交 infra/terraform/.terraform/；它是本地工作目录，包含 backend 初始化信息。
不要提交 *.tfstate、*.tfstate.*、*.tfvars、*.tfvars.json、*.tfplan、crash logs、
rclone.conf、*.credentials.json、service-account*.json、.env 或 .env.*。
terraform.tfvars.example 与 .terraform.lock.hcl 明确不被 ignore。

## 5. RunPod Console：三个 Pod 填什么

| 设置 | GPU Pod | Prometheus CPU Pod | Grafana CPU Pod |
| --- | --- | --- | --- |
| Image | vllm/vllm-openai:v0.26.0 | prom/prometheus:v3.5.0 | grafana/grafana:12.1.0 |
| Compute | NVIDIA GPU，1 x A100 SXM 80GB，Secure Cloud | CPU-only，GPU=0 | CPU-only，GPU=0 |
| HTTP port | 8000 | 9090 | 3000 |
| Container disk | 建议 50 GB | 建议 20 GB | 建议 20 GB |
| Network Volume | Terraform 输出的 ID，目标 data center | 无 | 无 |
| Volume mount | /workspace | 最小配置 volume disk=0 | 最小配置 volume disk=0 |
| Start command | 首次用下面的准备模式；之后可用 gpu_start_command JSON | 外部生成的 prometheus-start-command.json 全文 | 留空，保留官方 /run.sh |
| Repository | 仅 GPU runtime clone | 不需要 | 不需要 |

Console 的 HTTP port 字段填写数字；若界面使用统一 ports 表示法，则分别是
8000/http、9090/http、3000/http。

### GPU 环境变量

复制 bootstrap_environment.gpu 的非空值：

- REPOSITORY_URL：真实 GitHub HTTPS repository URL。
- REPOSITORY_REVISION：实际完整 repository commit SHA。
- GPU_BOOTSTRAP_URL：该 commit 的 scripts/bootstrap_gpu.sh raw URL，Terraform 可生成。
- NETWORK_VOLUME_ID：实际挂载的 Terraform volume ID。
- AWQ_SOURCE_REVISION：Phase 1/2 已验证 AWQ 的真实 source-model commit。
- MODEL_VARIANT=awq_w4a16，BF16 对比时改为 bf16。
- 首次复制需要 AWQ_GDRIVE_SOURCE=remote:folder、RCLONE_CONFIG=/workspace/.secrets/rclone.conf。
  已有 AWQ 后不需要 Drive credential/remote，即使保留这些变量也不会触发复制。

私有 GitHub repository 的访问凭据另行配置，不放在 URL 中。
GPU_BOOTSTRAP_URL 的自动下载适用于公开、已发布 commit；
私有仓库采用手工传入脚本并准备 Git credential 的首次模式。

### Prometheus 环境变量

**无必填环境变量。** scrape target 已内嵌在生成的启动 JSON 中；
VLLM_METRICS_URL 只提供给外部配置生成工具，不要求官方 Prometheus image 解析它。

### Grafana 环境变量

~~~text
GF_SECURITY_ADMIN_USER=admin
GF_SECURITY_ADMIN_PASSWORD=<私密管理员密码>
GF_USERS_ALLOW_SIGN_UP=false
GF_SERVER_HTTP_ADDR=0.0.0.0
GF_SERVER_HTTP_PORT=3000
GF_SERVER_ROOT_URL=https://<GRAFANA_POD_ID>-3000.proxy.runpod.net/
~~~

自己的 ID 在创建后才知道；GF_SERVER_ROOT_URL 可在得到 ID 后补填。
其余变量首次创建时设置。GF_SECURITY_ADMIN_PASSWORD 在新 Grafana DB 初始化时
设置管理员密码，不会替你重置已有 DB 的密码。
无需 PROMETHEUS_URL、Python、Git、项目目录或 provisioning 文件；
PROMETHEUS_URL 是外部 Grafana 配置 client 的输入。

## 6. GPU 完整首次启动

### 6.1 准备官方容器与 Drive credential

首次在 Console 的 Container Start Command 填写：

~~~json
{"entrypoint":["/bin/bash","-lc"],"cmd":["exec sleep infinity"]}
~~~

创建 Pod 时挂载 Network Volume、设置上述 GPU 变量。
在 GPU Pod web terminal 中私密准备 rclone config：

~~~bash
install -d -m 700 /workspace/.secrets
# 将已有 rclone.conf 私密传入此目录；type=drive 的 remote 已配置好授权。
chmod 600 /workspace/.secrets/rclone.conf
curl --fail --location "$GPU_BOOTSTRAP_URL" --output /tmp/qwen-bootstrap-gpu.sh
mkdir -p /workspace/models/qwen3-14b
nohup /bin/bash /tmp/qwen-bootstrap-gpu.sh \
  >> /workspace/models/qwen3-14b/bootstrap-gpu.log 2>&1 &
echo $! > /workspace/models/qwen3-14b/serving.pid
~~~

私有仓库可手工传入当前 bootstrap 文件到 /tmp，替代 curl。
不要把 credential 内容粘贴到公开日志或 repository。
nohup 使服务不依赖 web terminal 会话；可查看 bootstrap-gpu.log 等待准备和 vLLM 启动。

### 6.2 bootstrap 做什么

1. 从 PATH 发现官方 Python/vllm，确认 /workspace 为可写 mount，核对 GPU 与原有
   vLLM=0.26.0、compressed-tensors=0.17.0、torch=2.11.0、transformers=5.17.0。
2. 必要时只安装 git/curl/util-linux 等 OS utilities。
3. 首次 clone 实际 GitHub repository 到 /workspace/qwen-vllm-production，
   checkout 声明的 commit；已有 checkout 只核对 HEAD 和 tracked changes，
   不自动 pull/reset/覆盖。旧方案留下的无 .git seed 需要人工迁移。
4. 用官方 Python 的 venv 模块创建 .venv-benchmark，启用 system-site-packages，
   只读取 image 已有核心依赖。
5. 从已有 uv.lock 选出 httpx/PyYAML 与其 HTTP 依赖闭包，在 client venv 中使用
   pip --no-deps 安装明确列出的版本。不会运行 uv sync、安装项目 serving/evaluation
   extras，或下载/重新安装 vLLM、torch、CUDA、transformers。
   项目源码通过 PYTHONPATH=.../src 使用，无需构建/安装整个项目。
6. AWQ 缺失时准备 rclone 并执行下一节的 staging copy。
7. 调用原有 scripts/serve_vllm.py，使用官方 runtime Python/CLI，固定 TP=1、
   max_model_len=8192、GPU utilization=0.90、dtype=bfloat16、seed=42，
   prefix caching 仍由原 wrapper 禁用。

没有硬编码私有 Python 环境、预装 repository 或 image revision marker。
client-env.sh 只设置 client Python 路径和 PYTHONPATH，不改变 PATH，也不激活 venv。
runtime-packages.json 与 benchmark-requirements.txt 是运行时记录。

### 6.3 后续启动

首次准备成功后，日常部署可以把 Console Start Command 改成 Terraform 输出的
gpu_start_command JSON，并重新启动同一个 GPU Pod。该 JSON 通过
GPU_BOOTSTRAP_URL 获取同一 commit 的脚本后 exec bootstrap。
checkpoint 与 checkout 从 Network Volume 复用；容器被重置时仅补齐缺失工具。
正式 BF16/AWQ 对比期间保留 sleep infinity 作为容器 PID 1，通过后台 serving 进程
切换模型，保持同一个已分配 GPU Pod；不要在两个 variant 之间 stop/start 或重新调度。
不要在正式测量期间运行 bootstrap、安装 client 或重新启动服务。

## 7. Google Drive -> Network Volume

来源 folder 直接包含 config.json、tokenizer.json、tokenizer_config.json 和
safetensors 文件。AWQ_GDRIVE_SOURCE 是 rclone config 中 type=drive 的 remote:folder，
不是分享链接。首次缺少 rclone 时，GPU bootstrap 使用
[官方安装方式](https://rclone.org/downloads/#script-download-and-install)，不写 image。

实际执行流程：

~~~text
Google Drive remote:folder
  -> rclone copy --checksum -> /workspace/models/qwen3-14b/.awq_w4a16.download
  -> rclone check + 本地 config/tokenizer/weight shard 存在性检查
  -> 同卷 rename -> /workspace/models/qwen3-14b/awq_w4a16
~~~

两个 rclone 操作都显式使用 RCLONE_CONFIG。没有 sync/delete。
staging source identity 固定，复制失败可从同一来源继续。
**awq_w4a16 已存在时，跳过 rclone 安装、复制、check 与全部 Drive 网络访问。**
若现存 checkpoint 不完整，停止并报告，不覆盖或静默重下载。

source-revision marker 只是操作者的 provenance 声明，不能替代已有 Phase 2 验证。
BF16 从同源 revision 预先准备到 models/qwen3-14b/bf16，并写入真实
bf16/.source-revision；bootstrap 使用 HF offline，不下载模型到 image。

## 8. Prometheus 官方 image 如何获得配置

bootstrap_prometheus.sh 现在是**外部配置生成工具**，仅在 GPU Pod 或操作者机器
运行；CPU image 不运行该脚本，也不需要 Bash/Python/Git/project。

在 GPU Pod client terminal：

~~~bash
cd /workspace/qwen-vllm-production
source results/serving/runpod/client-env.sh
export VLLM_METRICS_URL="https://<GPU_POD_ID>-8000.proxy.runpod.net/metrics"
export PYTHON_BIN="$BENCHMARK_PYTHON"
bash scripts/bootstrap_prometheus.sh
~~~

它只在外部生成：

- results/serving/runpod/config/prometheus.yml
- results/serving/runpod/config/prometheus-start-command.json

将后者**完整 JSON**粘贴进 Prometheus CPU Pod 的 Container Start Command。
JSON 将 entrypoint 改为 image 自带 /bin/sh -c，启动时通过 printf 写
/tmp/qwen-prometheus.yml，再 exec /bin/prometheus。
配置内嵌在 Console 参数中，因此不依赖跨 Pod mount、文件上传服务或 config 下载。

scrape scheme=https、metrics_path=/metrics、
target=<GPU_POD_ID>-8000.proxy.runpod.net；保留 job=vllm、interval=5s、
timeout=4s、默认 TLS 验证。/prometheus 保持官方 image 的 nobody 写权限。
Pod 每次启动都会重新生成自己的配置。

## 9. Grafana 官方 image 如何获得 datasource/dashboard

Grafana Pod 保持官方 /run.sh，使用 GF_* 环境变量启动。
bootstrap_grafana.sh 是**外部 HTTP API 配置 client**，不在 Grafana CPU Pod 上运行。

先在 GPU Pod 第二个 terminal 进行现有 serving smoke，捕获真实 metrics：

~~~bash
cd /workspace/qwen-vllm-production
source results/serving/runpod/client-env.sh
"$BENCHMARK_PYTHON" scripts/check_serving.py \
  --base-url http://127.0.0.1:8000 --model qwen3-14b
"$BENCHMARK_PYTHON" scripts/benchmark_serving.py \
  --base-url http://127.0.0.1:8000 --model qwen3-14b \
  --variant awq_w4a16 --checkpoint-path /workspace/models/qwen3-14b/awq_w4a16 \
  --tokenizer-path /workspace/models/qwen3-14b/awq_w4a16 \
  --concurrency 1 --num-requests 1 --warmup-requests 0 \
  --dtype bfloat16 --serving-seed 42 --model-revision "" \
  --output-dir results/serving/runpod/smoke --run-id awq-single
"$BENCHMARK_PYTHON" scripts/check_serving.py \
  --base-url http://127.0.0.1:8000 --model qwen3-14b \
  --metrics-output results/serving/runpod/observability/vllm-metrics.txt
"$BENCHMARK_PYTHON" scripts/prepare_dashboard.py \
  --metrics-file results/serving/runpod/observability/vllm-metrics.txt \
  --output results/serving/runpod/observability/vllm-serving-overview.json \
  --manifest-output results/serving/runpod/observability/dashboard-metric-manifest.json

export PROMETHEUS_URL="https://<PROMETHEUS_POD_ID>-9090.proxy.runpod.net"
export GRAFANA_URL="https://<GRAFANA_POD_ID>-3000.proxy.runpod.net"
export GRAFANA_USER=admin
# 私密设置 GRAFANA_PASSWORD；或使用有足够权限的 GRAFANA_TOKEN。
export VLLM_DASHBOARD_FILE="$PWD/results/serving/runpod/observability/vllm-serving-overview.json"
export PYTHON_BIN="$BENCHMARK_PYTHON"
bash scripts/bootstrap_grafana.sh
~~~

不要按字面执行尖括号占位符；用真实 Pod ID 替换 URL，现有 run-id 已使用则换新 ID。

外部 client 使用 Grafana 12.1.0 已注册的 HTTP API：

- GET /api/datasources/uid/vllm-prometheus。
- 不存在则 POST /api/datasources，存在则 PUT /api/datasources/uid/vllm-prometheus。
- datasource type=prometheus、access=proxy、url=Prometheus CPU Pod HTTPS proxy。
- POST /api/dashboards/db 导入实际 runtime dashboard；保留稳定 UID，重复调用更新
  同一 dashboard，不创建重复对象。保留既有 folder。
- 使用管理员 Basic auth 或 service-account token、验证 TLS、拒绝重定向，
  不把密码/token 写入文件、CLI 参数或 Terraform。
- 拒绝 checked-in 的 up-only bootstrap dashboard，不修改原有 generator/metrics。

证据：
[Grafana 12.1 API routes](https://github.com/grafana/grafana/blob/v12.1.0/pkg/api/api.go)、
[Datasource API](https://grafana.com/docs/grafana/latest/developer-resources/api-reference/http-api/api-legacy/data_source/)。
这些 API 请求只会在操作者手工执行外部脚本时发生，本次没有调用。

## 10. 正式 benchmark

正式对比使用第 6.1 节的准备模式（sleep infinity 为 PID 1）和后台 serving。
如果曾使用自动启动模式，在整轮对比开始前切回准备模式并核实 GPU；整轮对比中
保持 Pod 运行，不重启或重新分配硬件。监控启动并完成既有 checker 后才开始测量：

~~~bash
"$BENCHMARK_PYTHON" scripts/check_prometheus.py --base-url "$PROMETHEUS_URL"
"$BENCHMARK_PYTHON" scripts/check_grafana.py --base-url "$GRAFANA_URL"
"$BENCHMARK_PYTHON" scripts/benchmark_sweep.py \
  --base-url http://127.0.0.1:8000 --model qwen3-14b \
  --variant awq_w4a16 --checkpoint-path /workspace/models/qwen3-14b/awq_w4a16 \
  --tokenizer-path /workspace/models/qwen3-14b/awq_w4a16 \
  --concurrency-levels 1,2,4,8,16,32,64 \
  --num-requests 1000 --warmup-requests 5 \
  --input-tokens 1000 --output-tokens 256 --seed 42 \
  --max-model-len 8192 --gpu-memory-utilization 0.90 --tensor-parallel-size 1 \
  --dtype bfloat16 --serving-seed 42 --model-revision "" \
  --monitoring-enabled --prometheus-url "$PROMETHEUS_URL" --grafana-url "$GRAFANA_URL" \
  --prometheus-scrape-interval-seconds 5 \
  --output-dir results/serving/runpod/formal --run-id awq-formal
~~~

请求数 1000 / warmup 5 是文档示例，正式协议确定后两个 variant 使用相同值。
BF16 仍在同一 GPU Pod/物理 GPU 上运行：
先停止前台服务或向记录的 serving PID 发送正常终止信号，确认所有引擎退出、显存释放。
再设置 MODEL_VARIANT=bf16 并运行同一 bootstrap。
在同一已运行 Pod 的 terminal 中 export MODEL_VARIANT=bf16，再按第 6.1 节的
nohup 方式启动 bootstrap 并更新 serving.pid；不重启容器，不让两个服务同时运行。

BF16 复用同一 tokenizer 与 saved workload：

~~~bash
"$BENCHMARK_PYTHON" scripts/benchmark_sweep.py \
  --base-url http://127.0.0.1:8000 --model qwen3-14b \
  --variant bf16 --checkpoint-path /workspace/models/qwen3-14b/bf16 \
  --tokenizer-path /workspace/models/qwen3-14b/awq_w4a16 \
  --workload-file results/serving/runpod/formal/awq-formal/workload.json \
  --concurrency-levels 1,2,4,8,16,32,64 \
  --num-requests 1000 --warmup-requests 5 \
  --input-tokens 1000 --output-tokens 256 --seed 42 \
  --max-model-len 8192 --gpu-memory-utilization 0.90 --tensor-parallel-size 1 \
  --dtype bfloat16 --serving-seed 42 --model-revision "$AWQ_SOURCE_REVISION" \
  --monitoring-enabled --prometheus-url "$PROMETHEUS_URL" --grafana-url "$GRAFANA_URL" \
  --prometheus-scrape-interval-seconds 5 \
  --output-dir results/serving/runpod/formal --run-id bf16-formal
~~~

TTFT/TPOT/E2E、SLO、goodput、closed-loop、workload generation、server wrapper、
quality evaluation、root README、Compose 与既有 monitoring configs 均未修改。
UTC/monotonic 时间记录和结果 schema 沿用既有实现，W8A8 不进入正式 serving 对比。

## 11. 静态 review 与人工验证边界

本次 deployment-safety patch 只新增根目录 .gitignore，并修改 versions.tf 与本文档。
三个 bootstrap、Network Volume resource、variables/outputs、benchmark、SLO、serving、
quality evaluation、Compose、observability configs 与根目录 README 均未修改。
没有新增 Dockerfile、image 构建流程、测试或新的 Python module。

静态核对：local backend 不写死 Drive 路径；Provider 仍为 runpod/runpod = 1.0.8；
.gitignore 保护 state/tfvars/plan/credentials，但保留 provider lock 与 tfvars example；
Colab Secrets 示例只输出加载状态；首次 plan 预期仅新增一个 Network Volume。

尚未运行或验证：镜像实际 dependency versions、Console JSON 执行、driver、rclone
权限/复制、Prometheus scrape、Grafana API 与真实 benchmark。
应由操作者按上面步骤验证，不能把静态 review 当成部署成功。
