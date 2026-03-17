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

*报告生成于 2026-03-17，基于 HuggingFace 公开数据分析。*
