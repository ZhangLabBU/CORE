# CORE 案例系统 — 环境部署与启动指南

> 适用范围：本文件夹（`CORE_paper_case/`）整理的代码，即论文 arXiv:2601.21822v1 第 IV 节 Case Study 的完整系统。
> 阅读顺序：**先看本文"零、总览"理解组件关系，再按 一 → 六 的顺序逐步部署启动。**

---

## 零、总览：系统由哪几部分组成，谁先谁后

系统一共需要 **4 个组件**，启动顺序有严格依赖关系：

```
┌──────────────────────────────────────────────────────────────┐
│ ① Elasticsearch        必须最先启动（角色路由的能力注册表）      │
│ ② LLM 推理服务          必须在 Agent 执行前可用                 │
│      ├─ 方案 A：DashScope 云端 API（最简，无需 GPU）           │
│      └─ 方案 B：vLLM 自部署（论文配置：8×A40 跑 70B 调度器，     │
│               3090/4090 跑 MiniCPM-v2.6 边缘模型）             │
│ ③ 能力注册表写入        依赖 ①，把 5 个 Agent 的能力灌进 ES      │
│ ④ CORE 主服务           依赖 ①②③，之后才能接受用户请求          │
└──────────────────────────────────────────────────────────────┘

启动顺序：① → ② → ③ → ④ → （客户端测试）
```

**为什么是这个顺序：**
- `main.py` 启动时就会初始化 ES 客户端（连不上会告警，路由直接不可用）；
- ④ 的每次请求都要查 ES（步骤 ③ 写入的 `capability_registry` 索引）；
- Agent 执行时调用 LLM（② 必须就绪），Embedding 也走 DashScope。

---

## 一、安装 Python 依赖

代码仓库没有 `requirements.txt`，按源码实际 import 整理如下（建议 Python 3.10+）：

```bash
python3 -m venv .venv && source .venv/bin/activate

pip install fastapi uvicorn websockets python-multipart \
            "elasticsearch>=8.0,<9" python-dotenv httpx pydantic \
            dashscope
# Agent 执行引擎（ReAct / LLMCompiler）与检索依赖 langchain 全家桶
pip install langchain langchain-core langchain-openai langchain-community
```

> 论文对比实验（CrewAI 基线）另需：`pip install crewai`（可选，仅复现基线时需要）。

## 二、部署 Elasticsearch（步骤 ①）

代码按 **ES 8.x**（basic auth + https）编写。Docker 一键启动：

```bash
docker run -d --name es-core \
  -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=true" \
  -e "ELASTIC_PASSWORD=你的ES密码" \
  -e "ES_JAVA_OPTS=-Xms4g -Xmx4g" \
  docker.elastic.co/elasticsearch/elasticsearch:8.13.4

# 验证
curl -k -u elastic:你的ES密码 https://localhost:9200
```

### 2.1 创建 `capability_registry` 索引

检索器 `4_capability_registry/langchain_es_bm25_demo.py` 同时使用 **BM25 + KNN 向量检索**，索引必须含 `dense_vector` 字段，否则 KNN 查询报错。用下面命令创建：

```bash
curl -k -u elastic:你的ES密码 -X PUT "https://localhost:9200/capability_registry" \
  -H "Content-Type: application/json" -d '
{
  "mappings": {
    "properties": {
      "agent_name":     {"type": "keyword"},
      "capability":     {"type": "text"},
      "keys":           {"type": "text"},
      "description":    {"type": "text"},
      "tools":          {"type": "text"},
      "embedding":      {"type": "dense_vector", "dims": 1536}
    }
  }
}'
```

> `dims=1536` 对应 DashScope `text-embedding-v2` 的输出维度。若换 embedding 模型，维度要同步修改。

## 三、准备 LLM 推理服务（步骤 ②）

### 方案 A：DashScope 云端 API（零 GPU，推荐先用这个跑通）

1. 到阿里云百炼开通 DashScope，拿到 API Key；
2. 在 `.env` 里写 `DASHSCOPE_API_KEY=sk-xxx` 即可。
   - Agent 对话模型：`qwen-plus`（`8_framework_runtime/utils/model_utils.py` 的 `openai` 分支已写死 DashScope 兼容端点）
   - Embedding 模型：`text-embedding-v2`

### 方案 B：vLLM 自部署（论文实验配置）

论文部署环境：**调度器** DeepSeek-R1-distill-llama-70B（8×A40，48GB 显存）；**边缘 Agent** MiniCPM-v2.6-8B（3090/4090）。用 `yzt_source/vllm/` 仓库起服务：

```bash
# 调度器节点（8×A40）
python -m vllm.entrypoints.openai.api_server \
  --model deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
  --tensor-parallel-size 8 --port 8000

# 边缘节点（3090/4090）
python -m vllm.entrypoints.openai.api_server \
  --model openbmb/MiniCPM-V-2_6 --port 8000
```

**注意两处需要改代码的地方**（自部署时）：
- `8_framework_runtime/utils/model_utils.py` 第 62 行 vLLM 分支的 API 地址写死为 `http://10.200.1.35:{port}/v1`，改成你的服务器 IP；
- 各 Agent 的 `configs.py` 中 `default_model`（当前为 `gpt-4-1106-preview`）改成你部署的模型名。

## 四、配置环境变量（连接 ①② 的关键）

**没有 .env 会导致所有环节静默失败。** 在项目根目录（与 `main.py` 同级）创建 `.env`：

```bash
# ========== Elasticsearch ==========
ES_URL=https://localhost:9200
ES_USER=elastic
ES_PASSWORD=你的ES密码
CAP_INDEX=capability_registry

# ========== LLM / Embedding ==========
DASHSCOPE_API_KEY=sk-你的DashScope密钥
EMBED_MODEL=text-embedding-v2

# ========== 角色亲和调度权重（论文 III-D，可调）==========
AGENT_THRESHOLD=0.5     # Agent 语义过滤阈值
W_SEM=0.70              # 语义分权重
W_LAT=0.20              # 时延分权重
W_Q=0.10                # 排队分权重

# ========== 主服务 ==========
PORT=8001
KEYFRAME_DIR=/data/keyframes        # 关键帧输出目录（自动创建）
UPLOAD_DIR=/data/uploads            # 用户上传视频目录（自动创建）
DEBUG_PAYLOAD=0                     # 调试置 1 可看路由/执行明细
ENABLE_SMALLTALK=1
```

> **重要坑**：`4_capability_registry/load_agents_to_es.py` 和 `langchain_es_bm25_demo.py` 里 `.env` 路径写死为服务器路径 `/home/yzt/AgentDNS/src/react/.env`。在本地部署时请把这两处 `ENV_PATH` 改为你的实际 `.env` 路径（或改成 `load_dotenv()` 默认从当前目录查找）。

## 五、写入能力注册表（步骤 ③，一次性）

把 5 个 Agent（VideoAgent / KeyframeAgent / MapAgent / MeteorologyAgent / ReportAgent）的能力描述向量化后写入 ES：

```bash
# 在项目根目录执行
export SEED_MD=$PWD/src/Pre_processing/k-v.md   # 能力种子文件（k-v.md）
python src/Pre_processing/load_agents_to_es.py
```

执行成功会打印写入条数。验证：

```bash
curl -k -u elastic:你的ES密码 \
  "https://localhost:9200/capability_registry/_count"
```

> 能力种子数据来自 `src/Pre_processing/k-v.md`（Key-Value 格式，每条 Value 是一个 JSON capability 描述）。要增删 Agent 能力，编辑该文件后重新执行本步骤。

## 六、启动 CORE 主服务并测试（步骤 ④）

```bash
python main.py          # 即 uvicorn 0.0.0.0:8001，见 main.py 末尾
```

### 6.1 按顺序自检

```bash
# 1) 健康检查
curl http://localhost:8001/health
# → {"status":"ok","service":"ws_only_server_refactored_smalltalk"}

# 2) 上传测试视频（Case 中的起火视频）
curl -X POST http://localhost:8001/upload/video -F "file=@fire.mp4"
# → 返回 {"path": "/data/uploads/xxx.mp4", ...}

# 3) WebSocket 发起任务（或直接用仓库自带的 ws_test_video.py）
python ws_test_video.py
```

### 6.2 WebSocket 请求协议（前端/测试脚本需按此发消息）

```json
{
  "type": "run",
  "request_id": "req_001",
  "user_input": "分析这段高速车辆起火视频，结合路况和气象给出救援建议",
  "top_k": 5,
  "uploaded_video_path": "/data/uploads/xxx.mp4"
}
```

服务端会依次推送：`ack` → `routing`（角色候选与打分）→ `thought` / `status`（执行进度）→ `final`（最终报告文本 + 图片）。一次 Case 完整执行 = `wf_video_keyframe_report` 工作流：KeyframeAgent → VideoAgent → ReportAgent。

### 6.3 单个 Agent 离线调试（不经过平台）

```bash
# start.sh 中的示例：单独跑 VideoAgent
python run_single_agent.py \
    --benchmark_names "VideoAgent" \
    --store ./result \
    --logging
```

可用的 `--benchmark_names`：`VideoAgent`、`KeyframeAgent`、`MapAgent`、`MeteorologyAgent`、`ReportAgent`。

---

## 七、常见问题排查

| 现象 | 原因 | 解决 |
|---|---|---|
| 启动时打印 `[WARN] Elasticsearch init failed` | ES 未启动或 `.env` 未加载 | 检查步骤 ① 和 `.env` |
| 请求返回"未匹配到可用 Agent" | 步骤 ③ 没执行，ES 索引是空的 | 重新执行 `load_agents_to_es.py` |
| 请求返回"未匹配到可用 Agent"（索引有数据） | 索引缺 `dense_vector` 映射，KNN 失败 | 按 2.1 重建索引再重跑步骤 ③ |
| Agent 执行报 401 / model not found | DashScope key 无效，或模型名没改 | 检查 `DASHSCOPE_API_KEY`、`configs.py` 的 `default_model` |
| 前端拿不到关键帧图片 | `KEYFRAME_DIR` 与 Agent 实际输出目录不一致 | 两边统一为同一路径 |
| 本地跑 `load_agents_to_es.py` 读不到环境变量 | `.env` 路径硬编码（见第四节注意事项） | 改 `ENV_PATH` 为本地路径 |
| 一切正常但路由结果总是不对 | embedding 维度与索引 `dims` 不匹配 | 重建索引，`dims` 对齐所用 embedding 模型 |

## 八、论文实验复现提示

- **基线对比**：ReAct、LLMCompiler 的执行引擎已包含在 `7_execution_engines/`；CrewAI 基线用 `yzt_source/crewai/` 仓库。
- **指标**：任务完成率（easy 0–3 工具 / medium 4–6 / hard 7–9）与延迟（scheduling latency + execution latency）的统计入口在 `8_framework_runtime/callbacks/`（StatsCallbackHandler）与 `utils/evaluation_utils.py`（`arun_and_time`）。
- **数据集**：FURG fire dataset（车辆起火异常视频）+ 300 条 LLM 生成并人工校验的任务指令，以 `datasets/*.json` 形式被 `run_single_agent.py` 读取（`agent_task_files` 定义）。
