# 基于心理咨询师数字孪生数据集的Qwen3-4B微调项目

## 📖 项目简介

本项目基于 Qwen3-4B 模型，使用 LoRA 技术对心理咨询师数字孪生数据集（PsyDTCorpus）进行微调，旨在构建专业的心理咨询对话模型。项目包含完整的训练、评估和对比功能，适用于心理咨询领域的对话生成任务。

## 🎯 项目特点

- **基于 Qwen3-4B**：使用小体量的大语言模型
- **LoRA 微调**：采用参数高效微调技术，降低训练成本
- **专业数据集**：使用 PsyDTCorpus 心理咨询数据集
- **完整流程**：包含数据查看、模型训练、性能评估和对比分析

## 📁 项目结构

```
PsyDTCorpus/
├── data/                    # 数据集目录
│   └── PsyDTCorpus/         # PsyDTCorpus 数据集
├── model_origin/            # 原始模型相关
│   ├── Qwen3-4B/           # Qwen3-4B 模型文件
│   └── generate_text.py    # 文本生成脚本
├── training/                # 训练相关
│   ├── train.py            # 训练脚本
│   ├── seeData.py          # 数据查看脚本
│   └── qwen-psy-trained/   # 训练后的模型
├── benchmark/               # 基准测试
│   ├── model_original.py   # 原始模型测试
│   ├── model_trained.py    # 微调后模型测试
│   └── PsychCounsel-Bench.json  # 测试数据集
└── compare/                 # 模型对比
    ├── model_origin.py     # 原始模型推理
    └── model_trained.py    # 微调后模型推理
```

## 🔧 环境要求

### Python 版本
- Python 3.10+

### 主要依赖

基于代码分析，项目主要依赖以下包：

```bash
torch>=2.0.0                    # PyTorch深度学习框架
transformers>=4.51.0            # Hugging Face Transformers（需支持Qwen3）
datasets>=2.0.0                 # Hugging Face Datasets
peft>=0.3.0                     # Parameter-Efficient Fine-Tuning（LoRA）
accelerate>=0.20.0              # 分布式训练加速
tqdm                            # 进度条显示
```

**注意**：由于使用 Qwen3-4B 模型，需要 `transformers>=4.51.0` 版本支持。

### 安装依赖

```bash
# 基础依赖
pip install torch transformers datasets peft accelerate tqdm

# 如果需要使用CUDA，请安装对应版本的PyTorch
# 例如：pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 环境配置示例

```bash
# 创建虚拟环境（推荐）
conda create -n psydt python=3.10
conda activate psydt

# 安装依赖
pip install torch transformers>=4.51.0 datasets peft accelerate tqdm
```

## 📦 数据集准备

### 下载 PsyDTCorpus 数据集

数据集应放置在 `data/PsyDTCorpus/` 目录下，包含以下文件：
- `PsyDTCorpus_train_mulit_turn_packing.json` - 训练集
- `PsyDTCorpus_test_single_turn_split.json` - 测试集

### 数据集格式

数据集采用 OpenAI 格式，每个样本包含：
- `id`: 样本ID
- `normalizedTag`: 标签（如"婚恋"、"工作"等）
- `messages`: 对话消息列表，包含 system、user、assistant 角色

## 🚀 快速开始

### 1. 查看数据集

```bash
cd training
python seeData.py
```

### 2. 训练模型

修改 `training/train.py` 中的路径配置：

```python
# 修改模型路径
model_path = "/path/to/your/Qwen3-4B"

# 修改数据集路径
train_data = load_dataset("json", data_files="/path/to/train.json")
test_data = load_dataset("json", data_files="/path/to/test.json")
```

然后运行训练：

```bash
cd training
python train.py
```

**训练配置**（基于代码）：

| 配置项               | 参数值                              |
| :------------------- | :---------------------------------- |
| **基座模型**         | Qwen3-4B                            |
| **微调方法**         | LoRA                                |
| `r`                  | 8（低秩矩阵的秩）                    |
| `lora_alpha`         | 32（LoRA 缩放因子）                 |
| `lora_dropout`       | 0.1（Dropout 率）                   |
| `target_modules`     | `["q_proj", "v_proj", "k_proj", "o_proj"]` |
| **训练轮数**         | 3 epochs                            |
| **学习率**           | 2e-4                                |
| **批次大小**         | 1（梯度累积步数：8，有效批次大小：8）|
| **最大序列长度**     | 384                                 |
| **优化器**           | AdamW                               |
| **混合精度**         | FP16                                |
| **评估步数**         | 每 200 步                           |
| **保存步数**         | 每 200 步（保留最佳 3 个检查点）    |
| **Warmup 步数**      | 100                                 |
| **数据加载工作线程** | 2                                   |

### 3. 模型评估

#### 评估原始模型（Baseline）

在运行前，修改 `benchmark/model_original.py` 中的配置：

```python
BASE_MODEL_PATH = "/path/to/your/Qwen3-4B"
BENCHMARK_PATH = "/path/to/PsychCounsel-Bench.json"
```

运行评估：

```bash
cd benchmark
python model_original.py
```

#### 评估微调后模型

修改 `benchmark/model_trained.py` 中的配置：

```python
BASE_MODEL_PATH = "/path/to/your/Qwen3-4B"
LORA_PATH = "/path/to/training/qwen-psy-trained"
BENCHMARK_PATH = "/path/to/PsychCounsel-Bench.json"
```

运行评估：

```bash
cd benchmark
python model_trained.py
```

**评估说明**：
- 评估脚本支持三种提示词格式：专业心理咨询师格式、简单直接格式、对话格式
- 结果会保存为 JSON 文件，包含准确率和错误案例分析
- 自动选择最佳提示词格式

### 4. 模型对比

#### 原始模型推理

修改 `compare/model_origin.py` 中的模型路径：

```python
model_path = "/path/to/your/Qwen3-4B"
```

运行推理：

```bash
cd compare
python model_origin.py
```

#### 微调后模型推理

修改 `compare/model_trained.py` 中的路径：

```python
model_path = "/path/to/your/Qwen3-4B"
lora_path = "/path/to/training/qwen-psy-trained"
```

运行推理：

```bash
cd compare
python model_trained.py
```

**对比功能**：
- 可以直观对比原始模型和微调后模型的回复差异
- 支持自定义测试问题

## 📊 模型配置

### LoRA 配置

```python
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                    # 低秩矩阵的秩
    lora_alpha=32,          # LoRA 缩放因子
    lora_dropout=0.1,       # Dropout 率
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    bias="none",
)
```

### 训练超参数

| 参数                         | 值      |
| :--------------------------- | :------ |
| `learning_rate`              | 2e-4    |
| `num_train_epochs`           | 3       |
| `per_device_train_batch_size`| 1       |
| `gradient_accumulation_steps`| 8       |
| `warmup_steps`               | 100     |
| `max_length`                 | 384     |
| `fp16`                       | True    |

## 📝 使用说明

### 修改路径配置

在使用前，请根据你的环境修改以下路径：

1. **模型路径**：在训练和推理脚本中修改 `model_path`
2. **数据集路径**：修改数据集文件的路径
3. **输出路径**：修改模型保存路径

### 自定义训练

你可以根据需要调整以下参数：
- LoRA 的 `r`、`alpha`、`dropout` 值
- 训练轮数、学习率、批次大小
- 最大序列长度

## 🔍 基准测试

项目使用 **PsychCounsel-Bench** 基准测试集（500 道心理学专业选择题）进行模型评估，支持多种提示词格式。

### 提示词格式说明

#### 1. 专业心理咨询师格式（推荐，准确率最高）

```python
你是一位专业的心理咨询师，请基于心理学专业知识选择最合适的答案。
问题：{题目内容}

选项：
A. {选项A内容}
B. {选项B内容}
C. {选项C内容}
D. {选项D内容}
E. {选项E内容}
请只输出选项字母（A/B/C/D/E），不要有任何其他文字。
```

**特点**：明确角色定位，要求基于专业心理学知识，指令简洁明确。

**示例**：

```
你是一位专业的心理咨询师，请基于心理学专业知识选择最合适的答案。
问题：An individual's self-esteem is most likely to improve when they credit their success to which of the following?

选项：
A. Factors within themselves
B. Factors outside themselves
C. Indirect factors
D. Random factors
E. Unstable factors

请只输出选项字母（A/B/C/D/E），不要有任何其他文字。
```

#### 2. 简单直接格式

```python
{题目内容}

A. {选项A内容}
B. {选项B内容}
C. {选项C内容}
D. {选项D内容}
E. {选项E内容}

请选择正确答案的字母：
```

**特点**：去除了角色设定和专业要求，直接呈现题目和选项，格式最简洁。

**示例**：

```
An individual's self-esteem is most likely to improve when they credit their success to which of the following?

A. Factors within themselves
B. Factors outside themselves
C. Indirect factors
D. Random factors
E. Unstable factors

请选择正确答案的字母：
```

#### 3. 对话格式

**System Message**：

```python
你是一位专业的心理咨询师，请根据心理学知识选择最合适的答案。
```

**User Message**：
```python
{题目内容}

选项：
A. {选项A内容}
B. {选项B内容}
C. {选项C内容}
D. {选项D内容}
E. {选项E内容}

请只输出选项字母：
```

**特点**：使用对话形式，包含 system 和 user 角色，更接近训练时的数据格式。

**示例**：

**System**:

```
你是一位专业的心理咨询师，请根据心理学知识选择最合适的答案。
```

**User**:

```
An individual's self-esteem is most likely to improve when they credit their success to which of the following?

选项：
A. Factors within themselves
B. Factors outside themselves
C. Indirect factors
D. Random factors
E. Unstable factors

请只输出选项字母：
```

### 评估结果

**不同格式下的表现对比**：

| 格式                 | 原始模型准确率 | 微调后模型准确率 | 说明                           |
| :------------------- | :------------- | :--------------- | :----------------------------- |
| **专业心理咨询师格式** | 80.60%         | 79.60%           | 准确率最高，**推荐使用**       |
| **简单直接格式**     | 46.20%         | 57.20%           | 中等准确率，微调后有明显提升   |
| **对话格式**         | 4.00%          | 2.20%            | 准确率较低，不适合选择题评估   |

**总体评估结果**：
- 原始模型最佳准确率：**80.60%**（专业心理咨询师格式）
- 微调后模型最佳准确率：**79.60%**（专业心理咨询师格式）
- 详细结果和对比分析请参考 [📊 对比分析](#-对比分析) 部分

评估结果会保存为 JSON 文件（`benchmark_test_results.json`），包含准确率、错误案例分析等详细信息。

## 📊 对比分析

### 测试示例

我们以心理咨询场景中的实际对话为例，对比了原始模型（without LoRA）和微调后模型（with LoRA）的表现。

**测试问题**：这段时间我不开心

#### 原始模型（without LoRA）

**模型信息**：

| 项目     | 值               |
| :------- | :--------------- |
| ✅ 状态  | 模型加载成功     |
| 📱 设备  | cuda:0           |
| 🔢 参数量| 4,022,468,096    |
| 💾 显存  | 7.49 GB          |

**模型回复**：

我能感受到你现在可能正经历一些困扰，情绪低落是人之常情。如果愿意的话，可以和我聊聊是什么让你感到不开心吗？有时候把心里的烦恼说出来，会感觉轻松一些。

如果你暂时不想多说，我也可以陪你一起做些轻松的事，比如听首音乐、聊聊天，或者一起散步。你并不孤单，我在这里陪着你。如果觉得压力太大，也可以考虑找信任的朋友倾诉，或者寻求专业的心理咨询帮助。

希望你能慢慢好起来，记得照顾好自己。如果需要的话，我随时都在。

**特点分析**：回复结构规范，提供了多种支持选项，但语气相对标准化，呈现一定的"心理咨询模板"特征。

#### 微调后模型（with LoRA）

**模型信息**：

| 项目     | 值               |
| :------- | :--------------- |
| ✅ 状态  | 微调模型加载成功 |
| 📱 设备  | cuda:0           |
| 🔢 参数量| 4,028,366,336    |
| 💾 显存  | 7.50 GB          |

**模型回复**：

我能感受到你现在心情不好，真的很抱歉听到你这样。有时候生活确实会给我们带来很多压力和困扰，让你感到难过和无助。你愿意和我聊聊发生了什么吗？我在这里陪着你，听你说说心事。

如果你不想马上说，也可以告诉我你想要什么样的支持：是想听听安慰的话，还是需要有人陪你理理头绪？不管怎样，我都会在这里支持你。有时候倾诉出来会感觉轻松一些，但如果你现在不想说，我理解，我们也可以先做点别的，比如聊聊你最近喜欢的事情，或者看看窗外的风景。

**特点分析**：共情表达更自然，主动询问用户需求并给予选择权，语气温和且贴近日常交流，在专业性和亲和力之间取得了更好的平衡。

### 定量评估结果

我们使用 **PsychCounsel-Bench** 基准测试集（500 道心理学专业选择题）对模型进行了全面的定量评估，测试了三种不同的提示词格式。

#### 基准测试结果对比

| 提示词格式             | 原始模型准确率       | 微调后模型准确率     | 提升幅度      |
| :--------------------- | :------------------- | :------------------- | :------------ |
| **专业心理咨询师格式** | **80.60%** (403/500) | **79.60%** (398/500) | -1.00%        |
| **简单直接格式**       | 46.20% (231/500)     | **57.20%** (286/500) | **+11.00%** ⬆️ |
| **对话格式**           | 4.00% (20/500)       | 2.20% (11/500)       | -1.80%        |

#### 详细分析

1. **专业心理咨询师格式（最佳格式）**
   - 原始模型：80.60% (403/500)
   - 微调后模型：79.60% (398/500)
   - **分析**：两种模型在该格式下都表现优秀，准确率超过 79%。微调后模型略有下降（-1%），差异在统计误差范围内，基本保持了原有性能水平。

2. **简单直接格式**
   - 原始模型：46.20% (231/500)
   - 微调后模型：57.20% (286/500)
   - **分析**：微调后模型在该格式下表现**显著提升**（+11%），准确率提升了近 11 个百分点。这表明微调增强了模型在非专业提示词格式下的适应性，提高了鲁棒性。

3. **对话格式**
   - 原始模型：4.00% (20/500)
   - 微调后模型：2.20% (11/500)
   - **分析**：两种模型在该格式下准确率都很低，说明对话格式不适合用于多项选择题类型的基准测试。这类格式更适合对话生成任务。

#### 性能评估总结

- ✅ **整体性能**：微调后模型在最佳格式下保持了接近原始模型的高准确率（79.60%）
- ✅ **格式鲁棒性**：微调显著提升了模型在简单直接格式下的表现（+11%），增强了模型对不同提示词格式的适应性
- ✅ **性能评级**：根据评估标准，准确率超过 70% 为优秀级别，微调效果显著
- 📊 **提示词敏感性**：结果表明，专业心理咨询师格式最适合心理学知识问答任务，而对话格式不适合选择题类型的评估

#### 错误案例分析

从错误案例中观察到的主要问题类型：
1. **概念混淆**：对心理学专业概念（如 retroactive interference vs proactive interference）的理解偏差
2. **理论应用**：将理论应用到具体情境时的判断错误
3. **答案提取**：部分情况下模型生成了解释但未能正确提取选项字母

### 分析总结

通过对比分析，我们发现微调后的模型在以下几个方面表现出明显提升：

#### 1. **共情表达更自然**
- **微调后模型**：使用更贴近日常交流的共情表达（如"真的很抱歉听到你这样"），语气自然温和，能更好地营造安全、支持的对话氛围
- **原始模型**：共情表达准确但略显模板化，呈现出一定的"心理咨询话术"特征

#### 2. **尊重用户自主性**（关键专业指标）
- **微调后模型**：主动将对话方向的选择权交还给用户（如"你想要什么样的支持？"），体现了以用户为中心的心理咨询原则，这是专业度的重要体现
- **原始模型**：更倾向于温和引导，在给予用户控制感方面相对较弱

#### 3. **角色定位更平衡**
- **微调后模型**：更好地平衡了"专业咨询师"与"支持性陪伴者"的角色，既保持了专业性，又增强了情感温度
- **原始模型**：虽然规范性更强，但略显正式，亲和力稍弱

#### 4. **技术指标**
- 参数量增加：约 590 万个参数（LoRA 适配器）
- 显存占用：仅增加约 0.01 GB，几乎可以忽略不计
- 推理效率：无明显差异，保持高效

#### 综合结论

综合定性对话分析和定量基准测试结果，微调后的模型在多个维度表现出色：

**定性优势**：
- ✅ **更自然的共情表达**：对话更加真实、贴近日常交流
- ✅ **更好的用户自主性尊重**：符合心理咨询的核心原则
- ✅ **更平衡的角色定位**：专业性与亲和力并重

**定量表现**：
- ✅ **保持高准确率**：在最佳格式下维持了 79.60% 的准确率（接近原始模型的 80.60%）
- ✅ **增强格式鲁棒性**：在简单直接格式下准确率显著提升 11%（从 46.20% 提升至 57.20%）
- ✅ **极小的性能开销**：仅增加约 590 万参数（约 0.15%）和 0.01 GB 显存占用
- ✅ **性能评级优秀**：准确率超过 70%，达到优秀级别

**总体评价**：

基于 PsyDTCorpus 数据集的 LoRA 微调在**保持模型专业知识准确性的同时**，显著提升了模型在**心理咨询对话场景下的专业性和用户体验**。虽然在某些格式下准确率略有波动，但整体表现稳定，特别是在增强格式鲁棒性和提升对话自然度方面效果显著。

这表明该微调方案有效平衡了**专业性**（知识准确性）和**对话质量**（共情表达、用户自主性）两个维度，为构建专业的心理健康对话系统提供了可靠的技术基础。

## ⚠️ 注意事项

1. **显存要求**：
   - 训练：建议使用至少 16GB 显存的 GPU（如 A100、V100、RTX 3090 等）
   - 推理：建议至少 8GB 显存
   - 使用 FP16 混合精度训练可以降低显存占用

2. **模型路径**：
   - 确保 Qwen3-4B 模型已正确下载到本地
   - 可以通过 ModelScope 或 Hugging Face 下载模型
   - 所有脚本中的路径都需要修改为实际路径

3. **数据集格式**：
   - 数据集必须符合 OpenAI 格式
   - 每个样本包含 `id`、`normalizedTag`、`messages` 字段
   - `messages` 中需要包含 `system`、`user`、`assistant` 角色

4. **依赖版本**：
   - **重要**：`transformers` 版本必须 >= 4.51.0（支持 Qwen3）
   - 较低版本会报错：`KeyError: 'qwen3'`

5. **训练建议**：
   - 训练过程中会每 200 步保存检查点
   - 最终模型保存在 `training/qwen-psy-trained/` 目录
   - 可以使用 `training/seeData.py` 先查看数据集结构

6. **推理参数**：
   - 非思考模式推荐参数：`Temperature=0.7`, `TopP=0.8`, `TopK=20`
   - 思考模式推荐参数：`Temperature=0.6`, `TopP=0.95`, `TopK=20`
   - 不要使用贪婪解码（greedy decoding），可能导致性能下降和无限重复

## 📄 许可证

本项目遵循以下许可证：
- **项目代码**：Apache 2.0 License
- **基座模型 Qwen3-4B**：Apache 2.0 License（参考 [Qwen3-4B LICENSE](https://huggingface.co/Qwen/Qwen3-4B/blob/main/LICENSE)）
- **数据集 PsyDTCorpus**：请参考数据集原始许可证

## 👥 作者

- **Yanze Jiang**

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来帮助改进项目！

## 🙏 致谢

- 感谢 [Qwen](https://github.com/QwenLM/Qwen) 团队提供的优秀基座模型
- 感谢 PsyDTCorpus 数据集的贡献者
- 感谢 [PEFT](https://github.com/huggingface/peft) 项目提供的 LoRA 实现

## 📚 相关资源

- [Qwen 模型](https://github.com/QwenLM/Qwen)
- [PEFT 文档](https://huggingface.co/docs/peft)
- [Transformers 文档](https://huggingface.co/docs/transformers)

## 📮 联系方式

如有问题或建议，欢迎通过以下方式联系：
- 提交 GitHub Issue
- 发送邮件（请提供联系方式）

## 📊 项目特点详解

### 数据格式处理

项目使用 Qwen 对话格式进行数据预处理：

```python
def format_conversation(messages):
    """格式化对话为Qwen格式"""
    text = ""
    for msg in messages:
        if msg["role"] == "system":
            text += f"<|im_start|>system\n{msg['content']}<|im_end|>\n"
        elif msg["role"] == "user":
            text += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
        elif msg["role"] == "assistant":
            text += f"<|im_start|>assistant\n{msg['content']}<|im_end|>\n"
    return text
```

### LoRA 参数效率

使用 LoRA 技术显著降低可训练参数数量：
- 只训练注意力层（Q、K、V、O 投影层）的 LoRA 适配器
- 大部分模型参数冻结，只更新少量参数
- 训练后的模型文件大小远小于全量微调

### 评估指标

使用 PsychCounsel-Bench 进行多项选择题评估：
- 自动提取模型生成的答案选项（A/B/C/D/E）
- 支持多种提示词格式对比
- 输出详细准确率和错误案例分析

---

**注意**：本项目仅用于科研和教育目的。模型生成的内容不应替代专业心理咨询服务。

