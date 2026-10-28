# CORE 论文 Case Study 代码集

> **论文**：《CORE: Toward Ubiquitous 6G Intelligence Through Collaborative Orchestration of Large Language Model Agents Over Hierarchical Edge》
> (Yu Zitong et al., BUPT, arXiv:2601.21822v1)
>
> 本文件夹整理了与论文 **第 IV 节 Case Study**（图 3：无人机高速车辆起火应急救援）直接相关的全部代码，按论文框架结构编号组织，便于对照阅读。**所有代码均为原样拷贝，未做任何逻辑修改。**

---

> **部署与启动顺序**：见 [DEPLOYMENT.md](./DEPLOYMENT.md)（ES → LLM 服务 → 能力注册 → 主服务 → 测试）。

## 一、论文 Case 与代码的总体对应关系

论文 Case 描述的场景是：**高速公路车辆起火事故**的实时应急分析。CORE 平台接收用户上传的起火视频后，进行多模态感知（视频、地理、气象）→ 任务分解（DAG）→ 多个专职 Agent 并行/串行协作执行 → 生成综合应急响应报告。

```
用户(起火视频+请求)
        │
        ▼
┌─────────────────────────────────────────────┐
│  平台服务层 (1_platform_server/main.py)      │  ← WebSocket 编排服务器，论文 Fig.5 平台界面后端
│  ├─ 闲聊/低置信度兜底                          │
│  ├─ 任务分解与工作流匹配 (2_orchestration)     │  ← 论文 Fig.3 步骤②③
│  └─ 角色亲和调度 (2_orchestration/routing.py) │  ← 论文 III-D Role Affinity Scheduling
├─────────────────────────────────────────────┤
│  专职角色 Agent (3_agent_configs)             │
│  ├─ KeyframeAgent  抽取关键帧                │  ← 视频感知
│  ├─ VideoAgent      视频理解/异常检测         │  ← 车辆起火识别
│  ├─ MapAgent        地理路线分析              │  ← 事故定位/救援路径
│  ├─ MeteorologyAgent 气象信息分析            │  ← 风向/天气对火势影响
│  └─ ReportAgent     生成应急报告              │  ← 最终响应方案
├─────────────────────────────────────────────┤
│  执行引擎 (7_execution_engines)               │
│  ├─ llm_compiler/  流水线并行执行 (DAG)      │  ← 论文 III-C Pipeline-parallel Execution
│  └─ react/          ReAct 基线引擎           │  ← 论文对比基线 [1]
└─────────────────────────────────────────────┘
```

## 二、目录详解（按编号）

### `1_platform_server/` — CORE 平台服务入口（论文 Fig.5、Section IV 部署）

| 文件 | 说明 | 对应论文内容 |
|---|---|---|
| `main.py` | FastAPI + WebSocket 编排服务器主入口。完整实现：闲聊快速通道 → Workflow 任务分解匹配 → ES 角色路由 → Query 改写 → Agent 执行 → 结果归一化 → 流式推送 | Primary Service Layer（论文 III-A）；实时异常检测平台（论文 Fig.5） |
| `run_single_agent.py` | 单 Agent 调试/执行入口，封装 `process_agent_request()`，支持 ReAct 与 LLMCompiler 两种引擎 | Agent 执行运行时 |
| `openclaw_client.py` | 对接 OpenClaw 技能服务的客户端 | Model Context Protocol (MCP) 的外部工具接入（论文 III-B） |
| `start.sh` | 服务启动脚本示例 | — |

### `2_orchestration/` — 动态角色编排与角色亲和调度（论文 III-C / III-D）

| 文件 | 说明 | 对应论文内容 |
|---|---|---|
| `routing.py` | **Role Affinity Scheduling 核心实现**：从 ES 能力注册表混合检索（BM25+向量）候选 Agent，计算语义分 sem_score 与网络状态分 net_score（0.7×延迟 + 0.3×队列），加权融合后选出最优角色，含低置信度回退 | 论文 III-D Role Affinity Scheduling Algorithm；Fig.3 中 agent 候选打分 |
| `workflows.py` | 5 条案例工作流的 DAG 定义，如 `wf_video_keyframe_report`（关键帧→视频分析→报告）、`wf_map_meteo_report`（地理→气象→报告） | 论文 III-C 任务分解为 DAG；Fig.3 步骤②任务分解 |
| `workflow_router.py` | 根据用户输入与是否上传视频，动态匹配工作流 | Dynamic Role Orchestration（论文 III-A） |
| `normalize.py` / `media.py` / `utils.py` | Agent 输出归一化（抽取最佳文本/图片）、媒体 URL 构建、通用工具 | Feedback and Optimization Layer 的结果评估部分（论文 III-A） |

### `3_agent_configs/` — 五个专职角色 Agent 的配置（论文 Fig.3 Case 的执行角色）

每个子目录包含 `configs.py`（模型与参数）、`tools.py`（可用工具声明）、`gpt_prompts.py` / `llama_prompts.py`（角色提示词）：

| Agent | 职责 | Case 中的作用 |
|---|---|---|
| `VideoAgent/` | 视频理解、异常检测 | 识别车辆起火事故（FURG fire dataset 场景） |
| `KeyframeAgent/` | 关键帧抽取与识别 | 抽取起火关键画面供后续分析 |
| `MapAgent/` | 地理信息与路径分析 | 事故定位、救援路线规划 |
| `MeteorologyAgent/` | 气象数据查询 | 风向/天气对火势与救援的影响评估 |
| `ReportAgent/` | 报告生成 | 汇总上述结果，输出中文应急报告 |

### `4_capability_registry/` — 能力注册表构建（角色亲和调度的数据基础）

| 文件 | 说明 |
|---|---|
| `build_registry.py` | 将各 Agent 的能力描述、工具、模型规格构建为结构化注册表 |
| `langchain_es_bm25_demo.py` | `CapabilityHybridRetriever`：Elasticsearch BM25 + 向量混合检索实现，`routing.py` 直接依赖 |
| `load_agents_to_es.py` | 将能力注册表批量写入 ES 索引 `capability_registry` |

论文 III-D 中"任务需求、网络状况、设备能力"的匹配计算，其检索基础即来自这里构建的注册表。

### `5_services/` — 服务层逻辑

| 文件 | 说明 |
|---|---|
| `query_rewrite.py` | LLM 将用户请求改写为选中 Agent 的执行 prompt（上下文感知的任务下发） |
| `task_processing.py` | 调用 `run_single_agent.process_agent_request()` 执行任务，并从 logs/scratchpad 中抽取最终答案 |
| `rag_retrieval.py` | 基础 RAG 检索（旧版路由兼容） |

### `6_agent_tools/` — Agent 底层工具实现（各角色真正调用的能力）

| 文件 | 对应角色 |
|---|---|
| `Video_Understanding.py` | VideoAgent — 视频理解 |
| `KeyframeExtractor.py` / `KeyframeRecognition.py` | KeyframeAgent — 关键帧抽取/识别 |
| `Map.py` | MapAgent — 地图与路线 |
| `Meteorology.py` | MeteorologyAgent — 气象查询 |
| `Report.py` | ReportAgent — 报告生成/写文件 |
| `TavilySearch.py` / `wikipedia.py` | 外部知识检索（RAG 增强） |
| `media.py` | 媒体文件处理 |

### `7_execution_engines/` — 任务执行引擎

| 目录 | 说明 | 对应论文内容 |
|---|---|---|
| `llm_compiler/` | LLMCompiler 实现：planner（任务规划为 DAG）、task_fetching_unit（依赖满足即并行取任务）、output_parser | 论文 III-C **Pipeline-parallel Execution**（DAG + 并行函数调用），也是论文基线 [2] |
| `react/` | ReAct Agent 实现（推理+行动循环） | 论文对比基线 [1] |

### `8_framework_runtime/` — 框架支撑代码

`agents/`（Agent 封装）、`chains/`（LLM 链）、`executors/`（Agent 执行器）、`tools/`（工具基类）、`callbacks/`（统计回调，用于实验指标收集）、`utils/`（模型接入、日志、评估、计时）。其中 `utils/evaluation_utils.py` 与 `callbacks/callbacks.py` 支撑论文中的**任务完成率与延迟指标统计**。

## 三、论文实验部署环境 与 代码的对应

| 论文描述 | 代码/仓库位置（yzt_source 下） |
|---|---|
| 调度器 DeepSeek-R1-distill-llama-70B（8×A40） | 通过 `vllm/` 仓库部署 vLLM 服务；`8_framework_runtime/utils/model_utils.py` 中 `get_model()` 接入（openai 兼容接口） |
| 边缘端 MiniCPM-v2.6 (8B)（3090/4090） | `MiniCPM-V/` 仓库（模型权重与推理代码） |
| MCP 跨域协作 | `1_platform_server/openclaw_client.py`（OpenClaw 技能对接）+ llm_compiler 的任务传递机制 |
| 基线 LLMCompiler [2] | `LLMCompiler/`、`Raw_llmcompiler/`（开源仓库原版，未拷贝进本文件夹） |
| 基线 CrewAI [3] | `crewai/`（开源仓库，未拷贝进本文件夹） |
| 数据集 FURG fire dataset + 300 条任务指令 | 车辆起火视频经 `KeyframeAgent`/`VideoAgent` 处理（见 `6_agent_tools/`） |

## 四、Case 单次执行的阅读路径（推荐顺序）

1. `1_platform_server/main.py` → `pipeline_stream()`：看请求如何进入
2. `2_orchestration/workflows.py` + `workflow_router.py`：看"起火视频分析"如何被分解为 Keyframe→Video→Report 的 DAG
3. `2_orchestration/routing.py`：看角色亲和调度如何打分选角色
4. `5_services/query_rewrite.py` → `task_processing.py`：看任务如何改写并下发
5. `3_agent_configs/*/` + `6_agent_tools/`：看各角色如何实际执行（视频/地图/气象/报告）
6. `7_execution_engines/llm_compiler/`：看子任务如何流水线并行
7. `1_platform_server/main.py` `normalize_agent_result()`：看结果如何归一化推送回用户

## 五、未拷贝进本文件夹的相关内容

- `LLMCompiler/`、`crewai/`、`vllm/`、`MiniCPM-V/`：体量较大的开源参考仓库，保留在 `yzt_source/` 原位置，见上表对应关系。
- `AgentDNS/`：与本文件夹内容同源的另一份项目副本。
- `SPA/`、`AssistGen/`、`LLM-learning/`、`RL_learning/`、`ReAct/`：与本论文 case 无直接关系。
