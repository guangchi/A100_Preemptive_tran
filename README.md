# P/D KV cache experiments

本项目在一台机器的两张 GPU 上分别运行 Prefill（P）与 Decode（D），使用 NIXL/UCX 传输 KV cache。窗口排序和传输抢占可以独立启用；项目代码和 vLLM 导入钩子均在 `code/`，无需修改已安装的 vLLM。

## 环境与模型

已验证环境：Linux x86_64、Python 3.11.18、A100-SXM4-80GB × 2、NVIDIA 驱动 595.112.07、vLLM 0.33.0、PyTorch 2.14.0+cu130、NIXL 1.5.2。目标机需要支持 CUDA 13.0 的驱动、GPU 间 P2P/NVLink 和可读的 NVLink 计数器，以及 `bash`、`curl`、`ss`（iproute2）、`nvidia-smi`。如使用容器，P/D 需在能共享 CUDA IPC 的同一环境内运行。

从仓库根目录执行：

```bash
conda create -n pd-experiment python=3.11 pip
conda activate pd-experiment
python -m pip install -r requirements.txt
python -m pip check
python -c 'import torch; print(torch.__version__, torch.version.cuda); assert torch.version.cuda == "13.0"'
```

`requirements.txt` 固定已验证的主要依赖版本；其余传递依赖由安装器解析。若默认软件源没有相应版本，使用可提供这些版本及 CUDA 13.0 构建的软件源或 wheel 存档。不要仅凭包名或版本号假设 CUDA 变体一致。

模型权重不随 Git 仓库分发。取得 Hugging Face 模型访问权限并登录后，下载与参考输出对应的版本：

```bash
hf download meta-llama/Llama-3.1-8B-Instruct \
  --revision a2856192dd7c25b842431f39c179a6c2c2f627d1 \
  --local-dir models/Llama-3.1-8B-Instruct
```

也可设置 `MODEL_PATH=/absolute/path/to/Llama-3.1-8B-Instruct` 使用已有权重。仓库自带固定负载和 [A100 参考输出 CSV](results/reference_outputs_a100.csv)；参考输出用于逐 token 校验。更换 GPU 架构或软件版本时应先检查输出，确需新参考时用单卡直接推理生成，另存为新文件，再给比较脚本传 `--reference` 或设置 `PD_REFERENCE`：

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_BATCH_INVARIANT=1 \
  python code/generate_reference_outputs.py --output results/reference_target.csv
```

## 启动和复现

先确认机器上两张卡空闲且由 NVLink 连接，下面的 `0/1` 是示例，请改为目标机器的物理编号。对照程序和微基准都会用 UUID 绑定设备；正式程序使用调用它的 Python，单独运行 `run_pd_pair.sh` 时使用 PATH 中的 `python`，可用 `PD_PYTHON` 显式覆盖。

```bash
nvidia-smi topo -m
export PD_P_GPU=0 PD_D_GPU=1

# CPU 检查：不需要模型权重，不启动 GPU 服务
python code/run_pd_local_sort_comparison.py --dry-run
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s code -p 'test_pd_*.py'

# 双进程 GPU 传输验证：查看 cuda_ipc 和对应 GPU 的 NVLink Tx/Rx 增量
NIXL_TEST_SIZES_MIB=64 python code/test_nixl_nvlink.py
```

正式复现实验：10ms 入口窗口、C16、单槽、0.1ms 轮询，25ms/50ms 两种批间隔，每组 3 轮且每次正式回放前预热。启动器的默认块大小仍是 C8，因此这里显式设为 C16。

```bash
run_dir="$PWD/results/nvlink-fourway-$(date +%Y%m%d-%H%M%S)"
CHUNK_SIZE_BLOCKS=16 PD_LOCAL_SORT_WINDOW_MS=10 \
  PD_STREAM_MAX_IN_FLIGHT=1 PD_STREAM_POLL_MS=0.1 \
  PD_PIPELINE_TRACE="$run_dir/trace" PD_API_TIMING=1 \
  python code/run_pd_local_sort_comparison.py --result-dir "$run_dir" \
  --schemes baseline sort_only preempt_only new --rounds 3 --warmup

python code/analyze_pd_local_sort_comparison.py --result-dir "$run_dir"
python code/audit_pd_streaming.py "$run_dir"
python code/analyze_pd_pipeline_trace.py "$run_dir"
```

| 参数 | 实验组 |
| --- | --- |
| `baseline` | 原生 NIXL 整请求传输 |
| `sort_only` | 窗口排序 + 流式 FIFO |
| `preempt_only` | 流式抢占 |
| `new` | 窗口排序 + 流式抢占 |
| `stream_fifo` | 无排序、无抢占的流式对照 |

服务默认使用 HTTP 端口 `8000/8100/8200` 和 NIXL 侧信道端口 `5600/5601`，启动前须空闲。HTTP 端口可通过 `ROUTER_PORT`、`P_PORT`、`D_PORT` 覆盖。脚本默认读取仓库中的模型、负载和参考路径；可从任意工作目录用脚本路径启动。

正式运行会把负载和参考 CSV 复制到结果目录，供独立审计。用 `nvlink_validation.json` 核对每次运行的 P Tx 与 D Rx 是否匹配该负载的约 5.8175 GB KV 数据，并检查 UCX 日志里的 `cuda_ipc`。仅能启动服务或看到两张卡有 NVLink 连接，不能证明本次传输实际走了 NVLink。更换硬件后的性能需要重新测量。

提交 Git 时保留 `code/`、本文、依赖清单和参考 CSV。模型、缓存、本机配置和其他 `results/` 输出默认忽略。历史实验报告中的链接指向本地归档，归档不参与新实验启动。
