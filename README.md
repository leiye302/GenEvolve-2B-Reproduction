# GenEvolve · Qwen3-VL-2B · 四卡复现

从公开的 Qwen3-VL-2B-Instruct 开始，依次做多模态 SFT、在线 GRPO 和视觉经验蒸馏，导出训练好的策略，再生成并评测 GenEvolve-Bench 的 594 张图。原始 WISE 的 1,000 题可以单独运行。训练部分根据论文 v2 重建，基座、训练数据、外部工具和评测来源的版本在 `sources.json` 中固定。[论文](https://arxiv.org/html/2605.21605v2) · [GenEvolve 官方代码](https://github.com/MeiGen-AI/GenEvolve/tree/23c847c559ccc0f95bbf4b3d8925898463822f4c)

## 启动

训练机使用 Linux x86-64、Python 3.11、四张支持 BF16 的 NVIDIA 卡、CUDA 12.8 toolkit 和可用的 C/C++ 编译器。32K SFT、长轨迹和本地渲染都需要显存。四张 80 GB 卡是本配置的部署参考，较小显存需要先核实峰值。训练工作目录建议准备约 250 GB，渲染模型的下载和缓存另占空间。

### 1. 启动图像渲染服务

已有兼容的 Qwen-Image-Edit-2511 服务时，将 `QWEN_SERVICE_URL` 设为该服务的根地址。自己部署时在渲染机器解包同一仓库，运行以下命令。服务默认只监听本机，接受官方客户端的 `POST /generate` 请求。

```bash
bash scripts/setup.sh --renderer
CUDA_VISIBLE_DEVICES=0 bash scripts/serve_renderer.sh --cpu-offload
```

服务首次启动会下载固定版本的 Qwen-Image-Edit-2511。训练和渲染同机时，可以在四张训练卡之一使用 `--cpu-offload`，同时需要给扩散模型和策略的峰值显存留空间。跨机器部署时让训练机能访问渲染地址，按需使用 `--host`。渲染用官方客户端的 40 步、CFG 4.0、seed 0 和最多两张参考图。[官方服务接口](https://github.com/MeiGen-AI/GenEvolve/blob/23c847c559ccc0f95bbf4b3d8925898463822f4c/genevolve/generator.py)

### 2. 在四卡机器跑完主流程

配置自己的 Serper Key 和能够访问 `gemini-3.1-pro-preview` 的 OpenAI 兼容端点。若使用 Google 官方 Gemini API，其兼容端点如下。搜索、在线奖励和评测会消耗对应服务额度。

```bash
bash scripts/setup.sh

export CUDA_VISIBLE_DEVICES=0,1,2,3
export QWEN_SERVICE_URL=http://127.0.0.1:8001
export JUDGE_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
read -rsp "Serper API key: " SERPER_API_KEY
printf '\n'
read -rsp "Gemini API key: " JUDGE_API_KEY
printf '\n'
export SERPER_API_KEY JUDGE_API_KEY

bash scripts/run_all.sh --work /path/to/genevolve-work
```

`setup.sh` 会拉取两个公开的固定提交，并在 `.venv` 中安装训练依赖。`run_all.sh` 依次运行 `prepare`、`sft`、`evolve` 和 `eval`。下载、模型和训练产物在 `--work` 指定的位置。Hugging Face 下载限流时可在本机配置自己的 `HF_TOKEN` 后重跑。

各阶段也能单独启动。复用同一工作目录和参数，SFT 从已保存的 ZeRO-3 断点恢复，自进化从最近完整断点恢复，评测跳过已落盘的题目。

```bash
source .venv/bin/activate
python run.py --work /path/to/genevolve-work prepare
python run.py --work /path/to/genevolve-work sft
python run.py --work /path/to/genevolve-work evolve
python run.py --work /path/to/genevolve-work eval
```

自进化默认覆盖 2,575 个训练请求，共 322 次更新。每 10 步保存一次可续训状态，结束时另保存。训练只保留最近两份自进化状态，逐步轨迹和日志会保留。最终策略用 HF safetensors 格式导出。

| 产物 | 工作目录下的位置 |
| --- | --- |
| SFT 权重、训练日志、275 和 550 步断点 | `sft/` |
| RL 轨迹、渲染结果、奖励和损失 | `evolve/rollouts/`、`evolve/metrics/` |
| RL 续训状态和最终 HF 权重 | `evolve/checkpoint-*/`、`evolve/final/` |
| GenEvolve-Bench 生成图和逐题评分 | `eval/bench-final/images/`、`eval/bench-final/results_eval.json` |
| KScore 汇总 | `eval/bench-final/summary.json`、`eval/bench-final/summary.csv` |

需要检查 SFT 权重或少量题时，`eval` 支持 `--policy /path/to/genevolve-work/sft` 和 `--max-cases 4`。`eval --split rl-val` 使用公开的 600 条自进化验证样本。

### 3. 评测原始 WISE

WISE 使用旧版数据和论文指定的 `gpt-4o-2024-05-13` 判分。需要一个仍能提供该模型的端点。设置 `WISE_API_KEY`，若用兼容服务再设置 `WISE_BASE_URL`。

```bash
bash scripts/setup.sh --wise-sources
read -rsp "WISE judge key: " WISE_API_KEY
printf '\n'
export WISE_API_KEY
.venv/bin/python run.py --work /path/to/genevolve-work eval --split wise
```

输出在 `eval/wise-final/`。代理只接收 WISE 原始请求，评审使用官方提示和评分函数，生成失败计零。[WISE 原版](https://github.com/PKU-YuanGroup/WISE/tree/d884ff1bcf4daa01f3a7913a1caea97a5aa393ae/WISE_legacy) · [论文评测协议](https://arxiv.org/html/2605.21605v2)

## 训练口径

- **SFT** 使用公开的 8,800 条训练和 200 条验证轨迹。用 LLaMA-Factory 的 Qwen3-VL-Instruct 模板保留每轮 assistant 标签，屏蔽系统、用户、工具反馈和视觉 token 的损失。四卡各放 2 个样本、累积 4 次，全局 batch 为 \(4\times2\times4=32\)。训练两轮，共 550 次更新。
- **在线自进化** 每步 8 个请求，每个请求 6 条轨迹。生成时保留论文的温度 0.7、top-p 0.95、最多 11 次模型调用和 30,000 个生成 token。图像 KScore 与五档程序充分性各占一半。策略更新使用非对称 GRPO 裁剪和系数 0.001 的参考策略 KL 惩罚。
- **视觉经验蒸馏** 对奖励差至少 0.20 且程序有效的同题最佳与最差轨迹提取经验。按原始请求嵌入的余弦相似度检索历史条目，把经验放在 teacher 的工具定义前。teacher 用同一批已采样 token 打分并停止梯度。每条轨迹从工具调用和最终程序的 token 中选差异最大的 10%，SDL 权重 2.0。
- **工程取舍** 作者的自进化训练入口没有公开。本实现使用 PyTorch FSDP 全分片和 CPU offload，每卡用同步后的 Transformers BF16 副本采样。参考策略固定为 SFT 权重。未给定的随机种子取 42，RL 训练覆盖一遍公开训练集，最后一批从开头补齐。每条轨迹先对 token 损失求和再按轨迹取平均。token 蒸馏损失上限取 10，经验容量 500，先淘汰奖励差较小的条目，同分按先进先出。

$$
R=0.5R_{\mathrm{img}}+0.5R_{\mathrm{text}},\qquad
k_3(\ell)=\exp(-\ell)-1+\ell,\qquad
\mathcal L=\mathcal L_{\mathrm{GRPO}}+2\mathcal L_{\mathrm{SDL}}.
$$

公开仓库的工具循环、参考图解析、生成客户端和 KScore 评分器由安装脚本原样取用。多轮 SFT 数据注册和标注检查沿用了已有复现工程的做法，其许可见 `LICENSES/stage1-MIT.txt`。[论文附录 B、C、F](https://arxiv.org/html/2605.21605v2)

## 离线验证

本地小模型用假工具、渲染模型和 judge 响应，执行真实 token log-prob、GRPO/SDL 反向传播、权重更新、断点恢复、HF 导出和评分汇总。运行测试需要训练环境和 FastAPI。

```bash
.venv/bin/python -m pip install fastapi==0.121.1
.venv/bin/python -m unittest discover -s tests -v
```

四卡 CUDA 全量训练和真实外部服务需在目标机器上运行。这里的 2B 设置与论文发布的 8B 分数对应不同模型。
