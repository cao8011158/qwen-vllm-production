# RunPod IaC 与三 Pod 部署

本次实现只创建文件并做静态 review，没有执行 Terraform、脚本、测试、
Docker 或 RunPod 资源 API。以下命令全部由操作者手工执行。

## 1. 官方 Provider 核验与实现边界

核验日期：2026-10-04。官方 namespace 为 runpod/runpod。
GitHub 最新稳定 release 与 Registry 官方版本元数据均包含 1.0.9：

- [官方 Registry](https://registry.terraform.io/providers/runpod/runpod/latest)
- [官方 Registry 版本元数据](https://registry.terraform.io/v1/providers/runpod/runpod/versions)
- [官方 v1.0.9 release](https://github.com/runpod/terraform-provider-runpod/releases/tag/v1.0.9)

不能把第三方 namespace 的 schema 当成官方 Provider schema，也不能仅凭
旧 README 或 terraform-provider-spec.json 断言实际能力。本实现检查了注册的
Go schema、Configure 和 Create 请求构造。

| 能力 | 核实结果与本项目处理 |
| --- | --- |
| Pod resource | 真实名称为 runpod_pod，但存在下面的实现缺口，本配置不声明 Pod resource |
| CPU Pod | 1.0.8/1.0.9 Pod schema 没有可配置的 compute_type、cpu_flavor_ids；vcpu_count 仅 computed。不能伪造 CPU Pod 属性，也不能用 gpu_count=0 代替 CPU 配置 |
| GPU 配置 | 1.0.8 声明 gpu_type_id，但 Create 未传入；1.0.9 传入 GPU id/count，但有客户端类型不匹配等问题。A100 选择由外部部署保证 |
| Network Volume 创建 | 真实 resource 为 runpod_network_volume；1.0.8 的 name、size、data_center_id 为 required，id 为 computed |
| Volume attachment | 1.0.8 的 network_volume_id 与 volume_mount_path 会传入 Pod Create；本项目在外部创建 GPU Pod 时选择 Terraform 输出的 volume |
| HTTP ports | 1.0.8 的 ports 是 string，但 Create 未传入；1.0.9 会拆分逗号分隔字符串。三个外部 Pod 分别暴露 8000/http、9090/http、3000/http |
| image/template | image_name 与 template_id 只能选一个，Create 有明确互斥检查；本项目记录真实不可变镜像，不猜 template ID |
| environment | 1.0.8 env 是 KEY=VALUE 的 string list，Create 转为 map；外部 Pod 使用同样的运行时环境约定 |
| startup | docker_entrypoint、docker_start_cmd 虽在 schema 中，但检查的 Create 未传入。使用镜像固化的启动配置或 Pod 内手工执行 bootstrap |
| container disk | container_disk_in_gb 是 int64，Create 会传入；外部 Pod 的建议值见 deployment_contract |

相关官方源码：

- [1.0.8 Pod schema](https://github.com/runpod/terraform-provider-runpod/blob/v1.0.8/internal/provider/resource_pod/pod_resource_gen.go)
- [1.0.8 Pod Create](https://github.com/runpod/terraform-provider-runpod/blob/v1.0.8/internal/provider/resource_pod/pod_resource.go)
- [1.0.8 Volume schema](https://github.com/runpod/terraform-provider-runpod/blob/v1.0.8/internal/provider/resource_network_volume/network_volume_resource_gen.go)
- [1.0.8 Volume Create](https://github.com/runpod/terraform-provider-runpod/blob/v1.0.8/internal/provider/resource_network_volume/network_volume_resource.go)
- [1.0.8 Provider Configure](https://github.com/runpod/terraform-provider-runpod/blob/v1.0.8/main.go)
- [1.0.9 Provider Configure](https://github.com/runpod/terraform-provider-runpod/blob/v1.0.9/main.go)
- [1.0.9 Pod implementation](https://github.com/runpod/terraform-provider-runpod/blob/v1.0.9/internal/provider/resource_pod/pod_resource.go)

1.0.9 的 Configure 把 RunPodClientWrapper 放入 ResourceData，而 Pod 和 Volume
Configure 将它断言为 RunPodClient。两者类型不一致；此外 Pod Create 未传入
type、startup 和 machine_id 等字段。这些是源码静态发现，不是实际运行结论。
因此本配置明确固定官方 1.0.8，只使用其 Network Volume resource。
它的默认 REST v1 endpoint 与 Volume 请求一致，不混用 v2 字段。

**当前交付不具备三个 Pod 的全自动 Terraform 创建能力。**
Terraform 管理存储；三个独立 Pod 是显式外部边界。没有 null_resource、
local-exec、remote-exec、curl 创建资源或隐藏 API 调用来绕过 Provider。
待官方修复后，应再次核验版本的实际 schema 和请求实现，再增加 Pod 管理。

## 2. 架构与网络边界

~~~text
GPU Pod: 1 x A100 SXM 80GB
  vLLM :8000 + benchmark client
  benchmark -> http://127.0.0.1:8000
  Network Volume -> /workspace
             |
             | HTTPS GPU Pod proxy /metrics
             v
CPU Pod: Prometheus :9090
             |
             | HTTPS Prometheus Pod proxy
             v
CPU Pod: Grafana :3000
~~~

正式 TTFT、TPOT、E2E 请求全部留在 GPU Pod 内。GPU proxy URL 只用于监控；
不要把 Terraform 的 monitoring_urls.vllm_metrics 转成 benchmark base URL。
三个 Pod ID 必须不同。CPU Pod 必须明确选择 CPU compute，不能租 GPU 充当 CPU。

proxy 地址来自 RunPod 官方约定：
https://<pod-id>-<port>.proxy.runpod.net。
bootstrap 中的服务监听 0.0.0.0，benchmark 始终访问本机 127.0.0.1。
参见 [官方端口与连接说明](https://docs.runpod.io/runpodctl/reference/runpodctl-remove-pods)。

暴露 8000/http 同时暴露该端口的 generation API 与 /metrics；
现有 serving wrapper 未新增鉴权。部署前需由操作者配置适用的访问控制。
Grafana 必须设置独立私密管理员密码，Prometheus proxy 也应纳入访问范围管理。

## 3. Terraform 管理的存储

默认创建：

- name：qwen-vllm-production
- size：100 GB
- data center：操作者选择的真实 ID，必须有 Network Volume 与目标 A100 库存
- lifecycle：prevent_destroy=true

Network Volume 与 GPU Pod 必须处于同一 data center。GPU Pod 选择 Secure Cloud，
在创建时附加该 volume，mount path 为 /workspace。不能先创建 Pod 再补挂载。
参见 [官方 Network Volume 文档](https://docs.runpod.io/storage/network-volumes)。

实际流程：

~~~text
Terraform 创建 Network Volume
  -> 输出 network_volume_id
  -> 操作者外部创建 GPU Pod，选择 A100 SXM 80GB 并挂载该 ID
  -> bootstrap_gpu.sh 首次从 Google Drive 复制 AWQ
~~~

自动挂载的 Pod resource 未实现，原因见 Provider 核验表，而非 Volume resource
不存在。若要复用已有 volume，首次配置时设置 network_volume_id；Terraform
不会创建、读取、调整或删除这个外部 volume。操作者负责核实其容量和 data center。

已有的 Terraform-managed volume 不要直接切换为 external 模式：
count 改变会涉及 state 地址，prevent_destroy 会阻止删除。先审阅 state 迁移方案。
保护规则也意味着 terraform destroy 不能直接删除受保护的模型存储。
停止 Pod 不会自动停止独立 Network Volume 的存储计费。

## 4. 镜像、文件与凭据

IaC 不猜镜像 digest，也不构建镜像。pod_images 中记录操作者实际提供的
image@sha256:digest，供外部 Pod 创建使用。不能把现有 Compose 当成三个
RunPod Pod 的自动编排入口。

所有镜像包含同一个仓库 snapshot，放在 /opt/qwen-vllm-production，
包括脚本、src、observability、pyproject.toml 和原有 uv.lock。
镜像构建时在该目录写入 .deployment-revision，内容是实际仓库完整 commit SHA。
不把 .git、.venv、结果、模型或凭据打包进这个 repository seed。
实际依赖放在下面约定的独立环境中：

| Pod | 镜像必须具备 |
| --- | --- |
| GPU | Bash、coreutils、mountpoint、flock、rclone；/opt/venv/bin/python；Python >=3.11、vLLM 0.26.0、compressed-tensors 0.17.0、torch 2.11.0+cu130、transformers 5.17.0、httpx |
| Prometheus CPU | Bash、Python >=3.11（/usr/bin/python3）、Prometheus 3.5.0（/usr/local/bin/prometheus） |
| Grafana CPU | Bash、Python >=3.11（/usr/bin/python3）、Grafana 12.1.0（/usr/share/grafana/bin/grafana）及完整 /usr/share/grafana home |

其他已验证的实际路径可以通过各 bootstrap 的 *_BIN、PROJECT_PYTHON、
GRAFANA_HOME 等环境变量指定。CPU 镜像不需要 torch、vLLM 或 GPU。
裸 prom/prometheus 或 grafana/grafana 镜像未必包含脚本所需 Bash/Python，
不能未核实就套用本 bootstrap。构建 GPU 环境时使用现有 uv.lock；
本次没有安装依赖、修改 pyproject.toml 或重新生成 lock。

镜像需让实际运行用户能够写其 /workspace 子目录。两个 CPU Pod 使用各自的
volume disk 保存 Prometheus TSDB、Grafana DB 与运行时配置；它们不依赖 GPU
Network Volume。CPU Pod 删除后的数据保留由操作者备份安排。

凭据只在 Pod 运行环境或操作者本地注入：

- RUNPOD_API_KEY：Terraform 本地环境变量，不写进 tfvars。
- rclone config：GPU Pod 内的私密文件，例如 /run/secrets/rclone.conf。
  这是文件路径约定，不是 Provider secret-mount resource；由操作者自行传入文件。
- GRAFANA_PASSWORD：Grafana Pod 的运行时环境变量，不写进 Terraform state/output。

不要提交 terraform.tfvars、state、plan 或 credential 文件。当前没有替你增加
gitignore 或改变根目录 README。

## 5. 操作者手工创建存储与三个 Pod

### 5.1 本地 Terraform

首次创建私有 terraform.tfvars 时复制 example；已有该文件则直接编辑。
填写真实 data_center_id。其他 null 字段可以先保留，以便先创建存储。
本例为 PowerShell，在当前仓库根目录执行：

~~~powershell
Copy-Item -LiteralPath infra/terraform/terraform.tfvars.example -Destination infra/terraform/terraform.tfvars
# 编辑 terraform.tfvars；通过私密方式给当前进程设置 RUNPOD_API_KEY。
terraform -chdir=infra/terraform init
terraform -chdir=infra/terraform fmt -check
terraform -chdir=infra/terraform validate
terraform -chdir=infra/terraform plan -out=runpod.tfplan
# 审阅计划：默认应只新增一个 Network Volume，不应出现任何 Pod。
terraform -chdir=infra/terraform apply runpod.tfplan
terraform -chdir=infra/terraform output network_volume_id
~~~

操作者保留 init 生成的 .terraform.lock.hcl，并核对它锁定官方 runpod/runpod
1.0.8。静态 schema 检查不能证明真实库存或部署可用性；本次没有运行这些命令。

### 5.2 外部 Pod 创建

在 RunPod console 手工配置三个独立 Pod，使用真实镜像：

1. GPU Pod：Secure Cloud、1 x A100 SXM 80GB、目标 volume/data center、
   /workspace、container disk 建议 50 GB、8000/http。
2. Prometheus CPU Pod：CPU-only、9090/http、container disk 建议 20 GB，
   可写的自身 /workspace volume disk。按实验规模选择 CPU/RAM。
3. Grafana CPU Pod：CPU-only、3000/http、container disk 建议 20 GB，
   可写的自身 /workspace volume disk。

确保外部启动配置保留可用 terminal，或者在镜像内固化 Bash bootstrap。
不要以 Terraform 中声明但不传入的 docker_start_cmd 替代这一步。
三个 bootstrap 的服务进程最终使用 exec，前台运行，退出状态与信号直接传递。

将实际 Pod IDs、镜像 digests、deployment_revision、awq_source_revision、
awq_gdrive_source 写入私有 tfvars。两个 revision 必须来自实际构建记录；
main 不是不可变模型 revision。本实现未猜测 SHA、data center ID 或 image digest。
然后手工更新输出：

~~~powershell
terraform -chdir=infra/terraform plan -out=runpod.tfplan
terraform -chdir=infra/terraform apply runpod.tfplan
terraform -chdir=infra/terraform output -json deployment_contract
terraform -chdir=infra/terraform output -json bootstrap_environment
terraform -chdir=infra/terraform output -json monitoring_urls
~~~

这些 outputs 是配置约定，不会检查或改变外部 Pod。

## 6. GPU 持久目录与首次 Drive 复制

~~~text
/workspace/
├── qwen-vllm-production/
│   ├── .deployment-revision
│   ├── src/
│   ├── scripts/
│   ├── observability/
│   ├── uv.lock
│   └── results/serving/runpod/
└── models/
    └── qwen3-14b/
        ├── awq_w4a16/
        └── bf16/                  # 同源 BF16 snapshot，由操作者预先准备
~~~

bootstrap 拒绝未单独挂载的 /workspace，使用 volume 内的锁串行处理初始复制。
仓库从镜像 seed 首次复制，已有 checkout 不会自动 pull/reset/覆盖。
镜像和持久 checkout 的 .deployment-revision 必须与声明一致。

首次启动需要在 GPU Pod 中准备已有 Google Drive rclone credential config。
AWQ_GDRIVE_SOURCE 是其中 type=drive 的 remote:folder，不是 Google Drive 分享
链接。folder 必须直接包含 config.json、tokenizer 文件和模型权重。
RCLONE_CONFIG 指向该私密 config 文件，不通过 Terraform 传递文件内容。

bootstrap 的 checkpoint 流程：

1. 对不存在的 AWQ 目录，rclone copy 到 .awq_w4a16.download staging。
2. 固定 staging 的 source identity，失败后允许从同一来源再次执行。
3. rclone check 与本地 config/tokenizer/shard 存在性检查。
4. 校验成功后在同一 volume 内 rename 到 awq_w4a16。
5. 记录操作者声明的 AWQ source revision；后续启动复用 checkpoint，不再次下载。

已有 checkpoint 不会自动覆盖。source marker 不推断模型来源，也不替代 Phase 2
quality validation；操作者必须把已验证的 AWQ artifact 与真实 source revision 配对。
BF16 snapshot 必须从 Qwen/Qwen3-14B 的同一 revision 准备，写入实际验证过的
bf16/.source-revision。bootstrap 保持 HF offline，不下载 BF16。

在 GPU Pod 第一个 terminal 设置 bootstrap_environment.gpu 中的非空值，
准备 rclone config 后手工执行：

~~~bash
bash /opt/qwen-vllm-production/scripts/bootstrap_gpu.sh
~~~

脚本校验 GPU/版本，然后调用现有 scripts/serve_vllm.py；固定 TP=1、context=8192、
GPU utilization=0.90、dtype=bfloat16、seed=42。现有 wrapper 禁用 prefix caching。
不运行 benchmark、quality evaluation，也不安装或更新 package。

## 7. 监控启动与 runtime dashboard

### 7.1 Prometheus CPU Pod

设置 DEPLOYMENT_REVISION 与 VLLM_METRICS_URL，取 Terraform 输出的 GPU proxy
/metrics URL。它必须形如 https://<gpu-id>-8000.proxy.runpod.net/metrics。

~~~bash
bash /opt/qwen-vllm-production/scripts/bootstrap_prometheus.sh
~~~

脚本写自己的 /workspace/prometheus/config/prometheus.yml，
不覆盖原有本机或 Docker 配置。保持 job=vllm、scrape_interval=5s、
scrape_timeout=4s，使用 HTTPS 并保留默认 TLS 校验。

### 7.2 GPU Pod：生成一次 smoke 内容并准备 dashboard

下面在 GPU Pod 的第二个 terminal 执行；不要在 CPU Pod 或本地电脑执行 benchmark。
首次 smoke 是集成检查，不算正式结果：

~~~bash
cd /workspace/qwen-vllm-production
export PYTHONPATH="$PWD/src"
export PROJECT_PYTHON=/opt/venv/bin/python

"$PROJECT_PYTHON" scripts/check_serving.py \
  --base-url http://127.0.0.1:8000 --model qwen3-14b
"$PROJECT_PYTHON" scripts/benchmark_serving.py \
  --base-url http://127.0.0.1:8000 --model qwen3-14b \
  --variant awq_w4a16 --checkpoint-path /workspace/models/qwen3-14b/awq_w4a16 \
  --tokenizer-path /workspace/models/qwen3-14b/awq_w4a16 \
  --concurrency 1 --num-requests 1 --warmup-requests 0 \
  --dtype bfloat16 --serving-seed 42 --model-revision "" \
  --output-dir results/serving/runpod/smoke --run-id awq-single
"$PROJECT_PYTHON" scripts/check_serving.py \
  --base-url http://127.0.0.1:8000 --model qwen3-14b \
  --metrics-output results/serving/runpod/observability/vllm-metrics.txt
"$PROJECT_PYTHON" scripts/prepare_dashboard.py \
  --metrics-file results/serving/runpod/observability/vllm-metrics.txt \
  --output results/serving/runpod/observability/vllm-serving-overview.json \
  --manifest-output results/serving/runpod/observability/dashboard-metric-manifest.json
~~~

查看真实 metric manifest。原有 runtime TYPE 校验和缺失 panel 处理完全保留；
这里不猜新 metric、不加 exporter。GPU 和 CPU 的文件系统独立，操作者需把生成的
dashboard 文件传给 Grafana Pod，例如放到 /workspace/input/vllm-serving-overview.json。

### 7.3 Grafana CPU Pod

设置 DEPLOYMENT_REVISION、PROMETHEUS_URL、GRAFANA_PUBLIC_URL，
并私密设置 GRAFANA_PASSWORD。VLLM_DASHBOARD_FILE 指向刚传入的文件。

~~~bash
export VLLM_DASHBOARD_FILE=/workspace/input/vllm-serving-overview.json
bash /opt/qwen-vllm-production/scripts/bootstrap_grafana.sh
~~~

脚本复用现有 datasource/dashboard provisioning，保留 datasource UID
vllm-prometheus 与 dashboard UID vllm-serving-overview，拒绝只有 up panel 的
bootstrap dashboard。既有源码和 dashboard generator 未修改。
Grafana 使用 Prometheus CPU Pod proxy URL，与 GPU benchmark URL 分开。

在 GPU Pod 的 benchmark terminal 设置输出中的 PROMETHEUS_URL/GRAFANA_URL、
GRAFANA_USER 以及私密 GRAFANA_PASSWORD，然后手工使用现有 checker：

~~~bash
"$PROJECT_PYTHON" scripts/check_prometheus.py --base-url "$PROMETHEUS_URL"
"$PROJECT_PYTHON" scripts/check_grafana.py --base-url "$GRAFANA_URL"
~~~

检查 scrape target health、实际 metric data、datasource query 与 runtime dashboard，
而不是只确认三个进程存在。监控全程开启；UTC 时间窗口与 Prometheus time series
对齐，客户端 monotonic 计时仍是正式性能依据。

## 8. 正式同 Pod 对比

先完成前面的人工集成验证。两个模型共用同一个 GPU Pod、物理 A100、镜像、
tokenizer、saved workload、请求数、warmup、版本及监控设置。
1000 measured requests / 5 warmup 仅为下面的示例数值，按正式协议确定后固定。

GPU Pod 第二个 terminal：

~~~bash
"$PROJECT_PYTHON" scripts/benchmark_sweep.py \
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

停止 GPU 第一个 terminal 的前台 serving，确认进程退出并释放显存；
在同一 GPU Pod 中设置 MODEL_VARIANT=bf16 并再次启动 bootstrap_gpu.sh。
不新建或重新调度 GPU Pod。保持两个 CPU Pod 与监控运行。
第二个 terminal 设置真实 AWQ_SOURCE_REVISION，复用 AWQ saved workload：

~~~bash
"$PROJECT_PYTHON" scripts/benchmark_sweep.py \
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

复核 startup logs 与 raw/aggregate：token counts、finish/stop reason、
output_length_complete、UTC window、dtype、serving seed、revision 和 package versions。
保持既有 TTFT/TPOT/E2E、SLO、success、goodput、closed-loop 和 workload 逻辑；
不合并两个 variant 到同一个 sweep summary，不删除异常短输出。

## 9. 静态检查结论与尚待人工验证

本次只新增五个 Terraform 文件、三个 bootstrap 脚本和本文件。
原有 Python、tests、quality evaluation、workload、wrapper、SLO、Compose、
Prometheus/Grafana 配置、uv.lock 和根目录 README 均未改动。

已做文本/结构 review：官方 resource 名称和 Volume 字段、单独的 Pod 外部边界、
固定本机 benchmark URL、无 Terraform provisioner、默认 100 GB 存储保护、
镜像/模型 revision 输入、Drive staging、现有 runtime dashboard 复用。
没有运行 terraform fmt/validate、bash -n、脚本、tests 或真实服务。

尚待操作者提供或验证：

- 三个真实镜像及正确 binary paths，真实 repository/source revisions。
- A100 SXM 80GB 在所选 data center 的实际可用性。
- 三个外部 Pod 的 CPU/GPU 类型、网络卷挂载、HTTP proxy 连通与访问控制。
- Google Drive credential/remote 的可读性、完整 AWQ artifact。
- Provider 1.0.8 的人工 init/validate/plan 与实际 Volume 创建。
- GPU driver、镜像 stack、Prometheus/Grafana runtime 集成与正式 benchmark。

Terraform state 中没有这三个外部 Pod；销毁 Terraform-managed 资源不会替你停止它们。
实验结束时由操作者在 console 停止/删除相应 Pod，并独立处理模型卷的保留与备份。
