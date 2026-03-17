# NVIDIA Nemotron RL Agentic 数据集详细分析报告

> 分析日期: 2026-03-17
> 数据集来源: HuggingFace (NVIDIA)

---

## 一、概述

本报告对 NVIDIA 发布的两个强化学习（RL）Agentic 数据集进行详细对比分析：

| 属性 | Conversational-Tool-Use-Pivot-v1 | Function-Calling-Pivot-v1 |
|------|----------------------------------|---------------------------|
| 全名 | `nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1` | `nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1` |
| 用途 | 对话式工具调用决策训练 | 函数调用决策训练 |
| 许可证 | CC-BY-4.0 | CC-BY-4.0 |
| 商用 | 可商用 | 可商用 |
| 发布日期 | 2026-03-11 | 2026-03-11 |
| 所属框架 | NeMo Gym (RLVR) | NeMo Gym (RLVR) |

两个数据集均为 **NVIDIA NeMo Gym** 框架的一部分，用于通过 **RLVR（Reinforcement Learning from Verifiable Reward）** 方法训练大型语言模型的 Agentic 能力。核心思想是将专家工具使用轨迹（trajectory）的每一个助手决策步骤拆解为独立的 **行为克隆（Behavior Cloning）** 问题。

---

## 二、数据规模对比

| 指标 | Conversational-Tool-Use | Function-Calling |
|------|------------------------|-----------------|
| 训练样本数 | ~96,968 行 (README 标称 170,320) | ~9,620 行 (README 标称 8,458) |
| 文件格式 | JSONL | JSONL |
| 原始文件大小 | ~1.75 GB | ~285 MB |
| Parquet 大小 | ~1.56 GB | ~272 MB |
| 内存占用 | ~1.59 GB | ~264 MB |
| 字段数 | 11 个顶层字段 | 5 个顶层字段 |
| 规模比 | **约 10:1** | 基准 |

**关键发现**：Conversational-Tool-Use 数据集在规模上约是 Function-Calling 的 **10 倍**，反映了对话式工具使用场景的复杂度更高、覆盖面更广。

---

## 三、数据生成方法

### 3.1 Conversational-Tool-Use-Pivot-v1

**数据收集模型**:
- `deepseek-ai/DeepSeek-R1-0528`
- `deepseek-ai/DeepSeek-V3.2`
- `Qwen/Qwen3-235B-A22B-Thinking-2507`
- `Qwen/Qwen3-32B`

**标注/评估模型**:
- `openai/gpt-oss-120b`
- `Qwen/Qwen3-235B-A22B-Instruct-2507`

### 3.2 Function-Calling-Pivot-v1

**数据收集模型**:
- `deepseek-ai/DeepSeek-V3.2`
- `zai-org/GLM-4.6`
- `openai/gpt-oss-120b`
- `moonshotai/Kimi-K2-Instruct`

**关键区别**：两个数据集使用了部分重叠但不完全相同的模型组合来生成专家轨迹，Function-Calling 引入了 GLM-4.6 和 Kimi-K2，而 Conversational-Tool-Use 更多使用了 DeepSeek-R1 和 Qwen 系列。

---

## 四、数据 Schema 对比

### 4.1 Conversational-Tool-Use-Pivot-v1 (11 字段)

```
┌─────────────────────────┬──────────────┬──────────────────────────────────────────┐
│ 字段名                   │ 类型         │ 说明                                      │
├─────────────────────────┼──────────────┼──────────────────────────────────────────┤
│ trajectory_id           │ int64        │ 轨迹唯一标识                               │
│ responses_create_params │ nested obj   │ API 风格的请求参数（含对话历史+工具定义）      │
│ expected_action         │ JSON         │ 期望的动作（消息或函数调用）                   │
│ scenario                │ null         │ 场景标识（当前全为 null）                    │
│ num_unique_actions      │ int64        │ 可选动作数量 (2-16)                         │
│ meta_info               │ object       │ 包含 turn, step, assistant_depth           │
│ qwen_235b_info          │ object       │ Qwen-235B 评估奖励信号                     │
│ agent_ref               │ object       │ Agent 配置类型和名称                        │
│ pass_rate               │ float64      │ 评估通过率 (0 ~ 0.59)                      │
│ pass_rate_total         │ int64        │ 总评估次数 (16-64)                         │
│ pass_rate_passed        │ int64        │ 通过次数 (0-19)                            │
└─────────────────────────┴──────────────┴──────────────────────────────────────────┘
```

### 4.2 Function-Calling-Pivot-v1 (5 字段)

```
┌─────────────────────────┬──────────────┬──────────────────────────────────────────┐
│ 字段名                   │ 类型         │ 说明                                      │
├─────────────────────────┼──────────────┼──────────────────────────────────────────┤
│ trajectory_id           │ int64        │ 轨迹唯一标识                               │
│ info                    │ nested obj   │ 包含 turn, step, depth                    │
│ responses_create_params │ nested obj   │ API 风格的请求参数（含对话历史+工具定义）      │
│ expected_action         │ JSON         │ 期望的动作（消息或函数调用）                   │
│ agent_ref               │ object       │ Agent 配置类型和名称                        │
└─────────────────────────┴──────────────┴──────────────────────────────────────────┘
```

### 4.3 Schema 差异总结

| 特性 | Conversational-Tool-Use | Function-Calling |
|------|------------------------|-----------------|
| 奖励信号 (qwen_235b_info) | **有** | 无 |
| 通过率元数据 (pass_rate) | **有** | 无 |
| 可选动作数 (num_unique_actions) | **有** | 无 |
| 推理链 (reasoning blocks) | 无 | **有** |
| scenario 字段 | 有(但全 null) | 无 |
| 元信息字段名 | meta_info | info |

**核心差异**：Conversational-Tool-Use 包含丰富的 **RL 训练辅助信息**（奖励信号、通过率），适合直接用于 RL 训练流水线；Function-Calling 更精简，但保留了 **推理链（Chain-of-Thought）** 信息。

---

## 五、核心数据结构详解

### 5.1 `responses_create_params`（两数据集共有）

这是两个数据集中最核心的字段，采用与 OpenAI Responses API 兼容的格式：

```json
{
  "input": [
    {"role": "system", "content": "系统提示/Agent 策略文档..."},
    {"role": "user", "content": "用户消息..."},
    {"role": "assistant", "content": "助手回复..."},
    {"type": "function_call", "name": "工具名", "arguments": "...", "call_id": "..."},
    {"type": "function_call_output", "call_id": "...", "output": "..."}
  ],
  "tools": [
    {
      "type": "function",
      "name": "工具名称",
      "description": "工具描述",
      "parameters": { /* JSON Schema */ },
      "strict": true
    }
  ],
  "parallel_tool_calls": true/false
}
```

### 5.2 `expected_action`（两数据集共有）

期望动作有两种形式：

**文本回复**（模型不需要调用工具时）:
```json
{"type": "message", "content": "这是助手的文本回复..."}
```

**函数调用**（模型需要调用工具时）:
```json
{"type": "function_call", "name": "tool_name", "arguments": "{\"param\": \"value\"}"}
```

### 5.3 奖励信号（仅 Conversational-Tool-Use）

```json
{
  "qwen_235b_info": {
    "rewards": [1, 0, 1, 1, 0, ...],  // 每次评估的二值奖励
    "reward_mean": 0.4375,
    "reward_std": 0.496,
    "reward_var": 0.246
  }
}
```

---

## 六、领域覆盖分析

### 6.1 Conversational-Tool-Use-Pivot-v1

覆盖 **838 个不同领域**，包括但不限于：
- 教育科技 (EdTech)
- IT 技术支持
- 旅行预订
- 医疗健康/老年护理
- 体育商品
- 可再生能源
- 有机农业
- 金融服务
- 客户服务

每个领域定义了独立的工具生态系统，通常包含 **10+ 个工具**，涵盖：
- 身份认证 (authenticate_client)
- 数据查询 (check_coverage, get_balance)
- 服务操作 (booking, modifications, discounts)
- 人工升级 (escalation to human agents)

### 6.2 Function-Calling-Pivot-v1

同样覆盖多样化领域：
- 金融分析 (get_balance_sheet, get_earnings, get_income_statement)
- 旅行推荐 (searchengine, searchbook)
- 食品营养 (analyze_recipe, classify_cuisine, wine_pairing)
- 求职搜索 (jobsearch, recruitmentinformation)

---

## 七、对话深度与复杂度

| 指标 | Conversational-Tool-Use | Function-Calling |
|------|------------------------|-----------------|
| 最大助手深度 | 4+ (assistant_depth) | 12+ (depth) |
| 多轮上下文 | 完整对话历史 | 完整对话历史 + 推理链 |
| 决策点类型 | 消息回复 / 工具调用 | 消息回复 / 工具调用 |
| 并行工具调用 | 支持 (parallel_tool_calls) | 支持 (parallel_tool_calls) |
| 系统提示 | 详细的 Agent 策略文档 | 相对简洁 |

---

## 八、适用场景分析

### 8.1 Conversational-Tool-Use-Pivot-v1 适合：

1. **RL 训练管线**：内置奖励信号和通过率元数据，可直接用于 RLVR 训练
2. **对话式 Agent 微调**：涵盖丰富的对话策略和服务协议
3. **多领域通用 Agent**：838 个领域覆盖广泛的业务场景
4. **大规模训练**：近 10 万条样本提供充足的训练数据
5. **难度分级训练**：pass_rate 提供自然的课程学习（curriculum learning）信号

### 8.2 Function-Calling-Pivot-v1 适合：

1. **函数调用精准训练**：聚焦工具调用准确性
2. **思维链蒸馏**：保留推理链，可训练模型的推理过程
3. **轻量级实验**：数据量较小（~9.6K），适合快速迭代
4. **SFT 基线训练**：精简的 5 字段格式便于直接用于监督微调
5. **多步推理能力**：包含复杂的多步工具调用依赖链

---

## 九、技术特点总结

### 9.1 共同特点

- **合成数据**：全部由顶级 LLM 生成，非人工标注
- **Responses API 兼容**：数据格式与 OpenAI 风格的 Responses API 直接兼容
- **行为克隆范式**：每条记录是轨迹中的单个决策点
- **工具+消息双模式**：模型需学习何时调用工具、何时直接回复
- **NeMo Gym 集成**：与 NVIDIA NeMo Gym 框架无缝配合
- **CC-BY-4.0 可商用**：开放许可，支持商业应用

### 9.2 关键差异

| 维度 | Conversational-Tool-Use | Function-Calling |
|------|------------------------|-----------------|
| **侧重点** | 对话策略 + 工具使用 | 精准函数调用 |
| **规模** | 大规模 (~97K / ~1.75GB) | 轻量 (~9.6K / ~285MB) |
| **RL 信号** | 丰富（奖励+通过率） | 无 |
| **推理链** | 无 | 有 |
| **领域广度** | 838 个领域 | 相对较少 |
| **Schema 复杂度** | 11 字段 | 5 字段 |
| **系统提示** | 详细策略文档 | 相对简洁 |
| **生成模型** | DeepSeek-R1/V3.2, Qwen3 | DeepSeek-V3.2, GLM-4.6, GPT-OSS-120B, Kimi-K2 |

---

## 十、使用建议

### 10.1 训练 Pipeline 建议

```
Phase 1: SFT 预训练
  └─ 使用 Function-Calling 数据集进行监督微调
     （精简格式，含思维链，适合建立基础能力）

Phase 2: RL 强化
  └─ 使用 Conversational-Tool-Use 数据集进行 RLVR 训练
     （利用内置奖励信号和通过率进行强化学习）

Phase 3: 评估
  └─ 两个数据集交叉验证
     （Function-Calling 验证调用准确性，
      Conversational-Tool-Use 验证对话策略）
```

### 10.2 数据使用代码示例

```python
from datasets import load_dataset

# 加载 Conversational Tool Use 数据集
conv_ds = load_dataset(
    "nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1",
    split="train"
)

# 加载 Function Calling 数据集
fc_ds = load_dataset(
    "nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1",
    split="train"
)

# 查看样本
print(conv_ds[0].keys())
print(fc_ds[0].keys())
```

### 10.3 结合 NeMo Gym 使用

```bash
# 克隆 NeMo Gym
git clone https://github.com/NVIDIA-NeMo/Gym

# 两个数据集都与 NeMo Gym 框架兼容
# 具体集成方式参见 NeMo Gym 文档
```

---

## 十一、文件结构

```
data/
├── Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1/
│   ├── README.md          (3.02 KB)
│   └── train.jsonl        (~1.75 GB)
│
├── Nemotron-RL-Agentic-Function-Calling-Pivot-v1/
│   ├── README.md          (2.74 KB)
│   └── train.jsonl        (~285 MB)
│
└── analysis_report.md     (本报告)
```

---

## 十二、参考链接

- NeMo Gym HuggingFace 集合: https://huggingface.co/collections/nvidia/nemo-gym/
- NeMo Gym GitHub: https://github.com/NVIDIA-NeMo/Gym
- NVIDIA NeMo: https://github.com/NVIDIA-NeMo/
- Conversational Tool Use 数据集: https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1
- Function Calling 数据集: https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1

---

## 十三、NeMo Gym Reward 计算机制深度分析

> 基于 NeMo Gym 源码 (`github.com/NVIDIA-NeMo/Gym`) 的实际实现分析。

### 13.1 三服务器架构

NeMo Gym 采用三服务器架构运行 RL 训练：

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────────────────────────────────┐
│ Agent Server │────▶│ Model Server │     │ Resources Server                             │
│ (编排)        │     │ (LLM 推理)   │     │ single_step_tool_use_with_argument_comparison │
│              │◀────│              │     │ (提供工具定义 + 计算 reward)                    │
│              │────────────────────────▶│ /verify 端点                                  │
└──────────────┘                         └──────────────────────────────────────────────┘
```

流程：
1. Agent Server 把 `responses_create_params` 发给 Model Server → 模型生成输出
2. Agent Server 把 `{模型输出, expected_action}` 发给 Resources Server 的 `/verify`
3. Resources Server 返回 `{reward, category}`

### 13.2 核心发现："Pivot" 是伪多轮，每条样本只推理一步

**这是理解这两个数据集最关键的一点。**

"Pivot"（透视）指的是**数据准备策略**：将完整的多轮专家轨迹拆解为 N 个独立的单步决策问题。每条训练样本中的 `responses_create_params.input` 已经包含了从对话开始到当前决策点的**完整历史**（来自预录制的专家轨迹），模型只需要预测**下一个 action**。

#### 源码证据：`tool_simulation_agent` 只调用模型一次

```python
# responses_api_agents/tool_simulation_agent/app.py
async def run(self, body):
    # 第1步：调用模型一次，没有循环
    response = await self.server_client.post(
        server_name=config.name,
        url_path="/v1/responses",
        json=body.responses_create_params,   # 完整历史作为输入
    )

    # 第2步：直接验证，返回 reward，结束
    verify_response = await self.server_client.post(
        server_name=config.resources_server.name,
        url_path="/verify",
        json={responses_create_params, response, expected_action},
    )
    return verify_response  # reward: 0 or 1
```

**没有 while 循环，没有工具模拟执行，没有继续对话。** 对比 NeMo Gym 中真正的多轮 Agent（`simple_agent`）：

```python
# simple_agent/app.py — 真正的多轮 rollout
while True:
    response = call_model(...)
    if has_tool_call(response):
        tool_result = execute_tool(response)  # 模拟执行工具
        history.append(tool_result)           # 加入历史
        continue                              # 继续循环
    else:
        break
```

#### 具体示例：turn=5, step=2 的训练样本

```
专家轨迹（离线预录制）:
  turn 1: user → [assistant 调 tool_A] → tool_A 返回 → assistant 回复
  turn 2: user → [assistant 调 tool_B] → tool_B 返回 → assistant 回复
  turn 3: user → [assistant 回复]
  turn 4: user → [assistant 调 tool_C] → tool_C 返回 → assistant 回复
  turn 5: user → [assistant 调 tool_D] → tool_D 返回 → [assistant 调 tool_E]
                                                        ↑ step=2，要预测的点

Pivot 拆解后的训练样本：
┌──────────────────────────────────────────────────────────┐
│ input: turn1~turn5 完整历史（含所有工具调用和返回值）         │
│        + turn5 step1 的 tool_D 调用和返回值                │
│        （全部来自专家轨迹，预先录制好的）                     │
│                                                          │
│ RL rollout: 模型只需预测一个 action                       │
│                                                          │
│ expected_action: function_call → tool_E(args...)          │
│ reward: 模型输出 vs expected_action → 0 或 1               │
└──────────────────────────────────────────────────────────┘
```

#### 多轮(turn) vs 多步(step) 的含义

- **Turn（轮）**= 用户发了一条新消息，每次用户说话开启一个新 turn
- **Step（步）**= 助手在同一轮里的第几次动作（先调工具 A → 再调工具 B → 最后回复）
- **Depth** = 全局累计深度，整个对话中助手已执行的动作总数

#### 为什么采用 Pivot 单步方案

```
传统多轮 RL rollout 的问题：
  ├── 需要模拟工具执行环境（复杂、不通用）
  ├── 错误累积：turn 1 错了，后面全错（稀疏奖励）
  ├── 长轨迹的信用分配困难
  └── 计算成本高（一条轨迹要推理 N 次）

Pivot 单步方案的优势：
  ├── 不需要工具模拟器（历史中工具返回值是预录的）
  ├── 每步独立评估，避免错误累积
  ├── 信用分配简单（一个 action 对应一个 reward）
  ├── 高度并行（所有样本独立，可大规模 batch）
  └── 本质是"用 RL 框架（GRPO）做行为克隆"
```

### 13.3 Reward 判定逻辑（两个数据集共用）

核心代码在 `resources_servers/single_step_tool_use_with_argument_comparison/app.py`。

#### 判定矩阵（二值奖励，只有 0 和 1，无部分得分）

| 期望动作 | 模型输出 | Reward | Category |
|---------|---------|--------|----------|
| function_call | function_call（匹配） | **1.0** | `EXPECTED_TOOL_CALL` |
| function_call | function_call（工具名错） | 0.0 | `UNEXPECTED_TOOL` |
| function_call | function_call（参数错） | 0.0 | `ARGUMENT_VALUE_DIFFERENT` 等 |
| function_call | message | 0.0 | `NO_EXPECTED_TOOL_CALL` |
| function_call | 无输出 | 0.0 | `NO_ACTION_FOUND` |
| **message** | **message（任意内容）** | **1.0** | `EXPECTED_CHAT_MESSAGE_FOUND` |
| message | function_call | 0.0 | `NO_EXPECTED_CHAT_MESSAGE` |

**重要**：当期望是 message 时，只要模型选择了"回复文本"而非"调用工具"，就直接给 1.0，**完全不比较回复内容是否正确**。Reward 只衡量"动作类型选择 + 工具调用准确性"。

### 13.4 函数调用参数比较规则 (`ToolCallComparator`)

参数比较是递归、类型感知的：

```
1. 工具名 → 精确匹配，错了直接 0.0
2. 参数 JSON 解析 → 解析失败直接 0.0
3. 逐个参数递归比较：
   ├── dict:      key 集合必须完全相同，然后递归比较每个 value
   ├── list:      长度必须相同，然后逐元素递归比较
   ├── int/bool/None: 精确相等
   ├── float:     容差 1e-6 内算匹配
   └── string:    见下方特殊规则
```

**字符串比较（唯一的"模糊"部分）**：

- 单词数 < 2 → 精确匹配
- 单词数 >= 2 → Jaccard 词频相似度

```
相似度 = 交集词数 / (期望总词数 + 实际总词数)
```

两个数据集的字符串匹配阈值均为 **0.1**（非常宽松）。例如 `"Birds are animals."` vs `"The birds fly."` → 交集=1, 总=6, 相似度=0.167 > 0.1 → 通过。

### 13.5 两个数据集的 Reward 差异

| | Function-Calling-Pivot | Conversational-Tool-Use-Pivot |
|---|---|---|
| Reward 验证逻辑 | 完全相同（同一个 Resources Server） | 完全相同 |
| Agent 配置名 | `toolcall_schema_*_agent` | `single_step_*_agent` |
| 字符串匹配阈值 | 0.1 | 0.1 |
| **数据集内预计算 reward** | **无** | **有** (`qwen_235b_info`) |
| Reward 来源 | 训练时由 NeMo Gym 在线计算 | 既有预计算的，也可在线重算 |

#### Conversational-Tool-Use 的预计算 reward

`qwen_235b_info` 是用 Qwen-235B 模型**多次采样**后，对每次采样结果用同样的 verify 逻辑打分得到的预计算统计量。配合 `pass_rate` 字段，提供了该决策点的难度信号，可用于：

- **课程学习**（curriculum learning）：按难度排序训练
- **数据筛选**：过滤掉太难或太简单的样本
- **离线 RL**：直接使用预计算奖励而不需要在线推理

### 13.6 Reward 流向 GRPO

在 NeMo RL 中（`nemo_rl/experience/rollouts.py`），verify 返回的单个 reward 标量直接作为 GRPO 的奖励信号：

```python
"total_reward": torch.tensor([r["full_result"]["reward"] for r in results])
```

GRPO 使用这个 per-sample reward 进行优势估计和策略优化，没有轨迹级别的奖励聚合或折扣。

### 13.7 本质总结

**这两个 Pivot 数据集本质上是"用 GRPO/RLVR 框架做行为克隆"**：

- 专家轨迹被拆解为独立的 (state, action) 对
- state = 完整对话历史（预录制），action = 下一步操作
- 每个样本是独立的单步决策问题
- 模型只推理一次，预测一个 action
- Reward 只比较这一个 action 是否与专家一致（0 或 1）
- 不存在真正的多轮 rollout 或工具执行模拟

---

## 十四、关于“伪多轮”与原始轨迹条数的补充说明

这两个数据集在 Hugging Face 上展示的 `num_rows`，并不是“原始多轮对话/原始任务”的条数，而是把一条 expert trajectory 中的每个 assistant 决策步骤拆成单独样本后的 **step-level 样本数**。因此它们更准确地说是 **伪多轮展开数据**，而不是真正以完整多轮会话为单位组织的数据。

如果用公开 Parquet 中的 `trajectory_id` 做去重，那么可以得到更接近“原始 expert trajectory 数量”的统计：

| 数据集 | 官方总行数 (`num_rows`) | 按 `trajectory_id` 去重后的原始轨迹数 |
|------|-------------------------|--------------------------------------|
| `Nemotron-RL-Agentic-Function-Calling-Pivot-v1` | 9,620 | 3,800 |
| `Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1` | 96,968 | 64,534 |

### 14.1 结论解读

- `Function-Calling-Pivot-v1` 表面上有 9,620 行，但对应的原始 expert trajectories 约为 **3,800 条**。
- `Conversational-Tool-Use-Pivot-v1` 表面上有 96,968 行，但对应的原始 expert trajectories 约为 **64,534 条**。
- 如果问题中的“原始数据有多少条”指的是“原始专家轨迹数”，那么答案应以 `trajectory_id` 去重后的数量为准，而不是 Hugging Face 页面显示的总行数。

### 14.2 统计方法说明

上述“原始轨迹数”并不是数据集卡显式给出的字段，而是基于公开 Parquet 文件执行 `COUNT(DISTINCT trajectory_id)` 得到的推断值。这个口径适合回答“实际原始数据有多少条”，尤其是在数据集过大、不方便完整下载到本地的场景下。

---

*报告更新于 2026-03-17，新增 NeMo Gym 源码级 Reward 机制分析，并补充伪多轮展开与原始 trajectory 去重统计。*
