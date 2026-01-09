<div align="center">

# Qwen3-4B Fine-tuning for Psychological Counseling with RAG

**基于心理咨询师数字孪生数据集的Qwen3-4B微调项目**

[English](README_EN.md) | [中文](README.md)

</div>

---

## 📖 Project Overview

This project fine-tunes the Qwen3-4B model using LoRA (Low-Rank Adaptation) on the Psychological Counseling Digital Twin Corpus (PsyDTCorpus) to build a professional psychological counseling dialogue model. The project integrates RAG (Retrieval-Augmented Generation) technology, combining a professional knowledge base (containing 5 psychology books) to retrieve relevant professional knowledge and enhance the professionalism and accuracy of model responses. The project includes complete training, evaluation, and comparison functionality, suitable for dialogue generation tasks in psychological counseling.

## 🎯 Key Features

- **Based on Qwen3-4B**: Uses a lightweight large language model
- **LoRA Fine-tuning**: Employs parameter-efficient fine-tuning to reduce training costs
- **Professional Dataset**: Uses the PsyDTCorpus psychological counseling dataset
- **RAG Enhancement**: Combines vector retrieval technology to retrieve relevant information from a professional psychology knowledge base
- **Complete Pipeline**: Includes data viewing, model training, performance evaluation, and comparative analysis

## 📁 Project Structure

```
PsyDTCorpus/
├── model_origin/            # Original model related
│   ├── Qwen3-4B/           # Qwen3-4B model files
│   └── generate_text.py    # Text generation script
├── lora/                    # LoRA fine-tuning module
│   ├── data/                # Dataset directory
│   │   ├── PsyDTCorpus/    # PsyDTCorpus dataset
│   │   └── seeData.py      # Data viewing script
│   ├── training/            # Training related
│   │   ├── train.py        # Training script
│   │   └── qwen-psy-trained/  # Trained model
│   ├── benchmark/           # Benchmark testing
│   │   ├── model_original.py    # Original model testing
│   │   ├── model_trained.py     # Fine-tuned model testing
│   │   └── PsychCounsel-Bench.json  # Test dataset
│   └── compare/             # Model comparison
│       ├── model_origin.py     # Original model inference
│       └── model_trained.py    # Fine-tuned model inference
└── rag/                     # RAG enhancement module
    ├── indexing.py         # Build knowledge base vector index
    ├── retrieval.py        # Knowledge retriever
    ├── compare_all.py      # Three-model comparison script
    ├── knowledge_base/     # Professional knowledge base (PDF files)
    │   ├── CCMD-3中国精神障碍分类与诊断标准.pdf
    │   ├── 中国心理学会临床与咨询心理学工作伦理守则解读.pdf
    │   ├── 国家职业资格培训教程 心理咨询师 三级.pdf
    │   ├── 心理咨询的理论与实务.pdf
    │   └── 认知疗法基础与应用.pdf
    └── vector_store/       # Vector storage directory
        └── psychology_db/  # FAISS vector database
```

## 🔧 Environment Requirements

### Python Version
- Python 3.10+

### Main Dependencies

Based on code analysis, the project mainly depends on the following packages:

```bash
torch>=2.0.0                    # PyTorch deep learning framework
transformers>=4.51.0            # Hugging Face Transformers (requires Qwen3 support)
datasets>=2.0.0                 # Hugging Face Datasets
peft>=0.3.0                     # Parameter-Efficient Fine-Tuning (LoRA)
accelerate>=0.20.0              # Distributed training acceleration
tqdm                            # Progress bar display

# RAG related dependencies (optional, if using RAG functionality)
langchain>=0.1.0                # LangChain framework
langchain-community>=0.0.20     # LangChain community components
langchain-huggingface>=0.0.1    # LangChain HuggingFace integration
faiss-cpu>=1.7.4                # FAISS vector database (CPU version)
# faiss-gpu>=1.7.4              # Or use GPU version (if GPU available)
pypdf>=3.0.0                    # PDF file processing
```

**Note**: Since we use the Qwen3-4B model, `transformers>=4.51.0` is required.

### Installing Dependencies

```bash
# Basic dependencies
pip install torch transformers datasets peft accelerate tqdm

# If you need to use CUDA, please install the corresponding version of PyTorch
# For example: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Environment Setup Example

```bash
# Create virtual environment (recommended)
conda create -n psydt python=3.10
conda activate psydt

# Install dependencies
pip install torch transformers>=4.51.0 datasets peft accelerate tqdm
```

## 📦 Dataset Preparation

### Download PsyDTCorpus Dataset

The dataset should be placed in the `data/PsyDTCorpus/` directory, containing the following files:
- `PsyDTCorpus_train_mulit_turn_packing.json` - Training set
- `PsyDTCorpus_test_single_turn_split.json` - Test set

### Dataset Format

The dataset uses OpenAI format, with each sample containing:
- `id`: Sample ID
- `normalizedTag`: Label (e.g., "marriage", "work", etc.)
- `messages`: List of dialogue messages, containing system, user, and assistant roles

## 🚀 Quick Start

### 1. View Dataset

```bash
cd lora/data
python seeData.py
```

### 2. Train Model

Modify the path configuration in `lora/training/train.py`:

```python
# Modify model path
model_path = "/path/to/your/Qwen3-4B"

# Modify dataset path
train_data = load_dataset("json", data_files="/path/to/train.json")
test_data = load_dataset("json", data_files="/path/to/test.json")
```

Then run training:

```bash
cd lora/training
python train.py
```

**Training Configuration** (based on code):

| Configuration Item          | Parameter Value                              |
| :-------------------------- | :------------------------------------------- |
| **Base Model**              | Qwen3-4B                                     |
| **Fine-tuning Method**      | LoRA                                         |
| `r`                         | 8 (rank of low-rank matrix)                  |
| `lora_alpha`                | 32 (LoRA scaling factor)                     |
| `lora_dropout`              | 0.1 (Dropout rate)                           |
| `target_modules`            | `["q_proj", "v_proj", "k_proj", "o_proj"]`   |
| **Training Epochs**         | 3 epochs                                     |
| **Learning Rate**           | 2e-4                                         |
| **Batch Size**              | 1 (gradient accumulation steps: 8, effective batch size: 8) |
| **Max Sequence Length**    | 384                                          |
| **Optimizer**               | AdamW                                        |
| **Mixed Precision**         | FP16                                         |
| **Evaluation Steps**        | Every 200 steps                              |
| **Save Steps**              | Every 200 steps (keep best 3 checkpoints)    |
| **Warmup Steps**            | 100                                          |
| **Data Loading Workers**   | 2                                            |

### 3. Model Evaluation

#### Evaluate Original Model (Baseline)

Before running, modify the configuration in `lora/benchmark/model_original.py`:

```python
BASE_MODEL_PATH = "/path/to/your/Qwen3-4B"
BENCHMARK_PATH = "/path/to/PsychCounsel-Bench.json"
```

Run evaluation:

```bash
cd lora/benchmark
python model_original.py
```

#### Evaluate Fine-tuned Model

Modify the configuration in `lora/benchmark/model_trained.py`:

```python
BASE_MODEL_PATH = "/path/to/your/Qwen3-4B"
LORA_PATH = "/path/to/lora/training/qwen-psy-trained"
BENCHMARK_PATH = "/path/to/PsychCounsel-Bench.json"
```

Run evaluation:

```bash
cd lora/benchmark
python model_trained.py
```

**Evaluation Notes**:
- The evaluation script supports three prompt formats: professional psychological counselor format, simple direct format, and dialogue format
- Results are saved as JSON files, containing accuracy and error case analysis
- Automatically selects the best prompt format

### 4. Model Comparison

#### Original Model Inference

Modify the model path in `lora/compare/model_origin.py`:

```python
model_path = "/path/to/your/Qwen3-4B"
```

Run inference:

```bash
cd lora/compare
python model_origin.py
```

#### Fine-tuned Model Inference

Modify the paths in `lora/compare/model_trained.py`:

```python
model_path = "/path/to/your/Qwen3-4B"
lora_path = "/path/to/lora/training/qwen-psy-trained"
```

Run inference:

```bash
cd lora/compare
python model_trained.py
```

**Comparison Features**:
- Can visually compare the response differences between the original model and the fine-tuned model
- Supports custom test questions

### 5. RAG-Enhanced Inference (Retrieval-Augmented Generation)

This project integrates RAG (Retrieval-Augmented Generation) functionality, which can retrieve relevant information from a professional knowledge base to enhance the professionalism and accuracy of model responses.

#### Build Knowledge Base Index

First, you need to build a vector index from PDF files:

```bash
cd rag
python indexing.py
```

**Notes**:
- The script automatically scans all PDF files in the `rag/knowledge_base/` directory
- Uses `BAAI/bge-small-zh-v1.5` as the embedding model
- Splits documents into small chunks (chunk_size=300, overlap=40) suitable for the Qwen3-4B model
- Builds FAISS vector index and saves to `rag/vector_store/psychology_db/`

#### Use RAG-Enhanced Inference

```python
from rag.retrieval import PsyRetriever

# Initialize retriever
retriever = PsyRetriever()

# Retrieve relevant context
query = "What are the main steps of cognitive therapy?"
context = retriever.get_relevant_context(query)

# Combine retrieved context with user question and input to model
```

#### Run Complete Comparison Test

You can use the `rag/compare_all.py` script to compare responses from three models simultaneously:

```bash
cd rag
python compare_all.py
```

This script automatically loads the original model, fine-tuned model, and RAG-enhanced model, and compares their responses to the same questions, making it easy to understand the effects of different configurations.

**Professional Books in Knowledge Base**:
1. CCMD-3 Chinese Classification and Diagnostic Criteria of Mental Disorders
2. Interpretation of the Code of Ethics for Clinical and Counseling Psychology of the Chinese Psychological Society
3. National Vocational Qualification Training Tutorial: Psychological Counselor Level 3
4. Theory and Practice of Psychological Counseling
5. Cognitive Therapy: Basics and Applications (2nd Edition)

**RAG Workflow**:
1. **Retrieval Phase**: Based on user questions, retrieve the most relevant document fragments from the professional knowledge base
2. **Enhancement Phase**: Use the retrieved context as part of the prompt and input it to the fine-tuned model
3. **Generation Phase**: The model generates responses based on the retrieved professional knowledge and its own trained knowledge

#### RAG Technical Details

**Vector Retrieval Configuration**:
- **Embedding Model**: `BAAI/bge-small-zh-v1.5` (lightweight embedding model optimized for Chinese)
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Document Chunking Strategy**:
  - `chunk_size=300`: Each document chunk is 300 characters, suitable for Qwen3-4B's context length
  - `chunk_overlap=40`: 40 characters overlap between chunks to maintain semantic continuity
- **Retrieval Strategy**: Similarity search, returns the most relevant 1 document chunk

**Advantages**:
- ✅ **Enhanced Professionalism**: Directly retrieves relevant knowledge from authoritative psychology books
- ✅ **Improved Accuracy**: Combines retrieved professional content to reduce model hallucinations
- ✅ **Knowledge Updates**: Can supplement the latest professional knowledge by updating the knowledge base
- ✅ **Explainability**: Can trace back to specific document sources retrieved

## 📊 Model Configuration

### LoRA Configuration

```python
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                    # Rank of low-rank matrix
    lora_alpha=32,          # LoRA scaling factor
    lora_dropout=0.1,       # Dropout rate
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    bias="none",
)
```

### Training Hyperparameters

| Parameter                      | Value |
| :---------------------------- | :---- |
| `learning_rate`               | 2e-4  |
| `num_train_epochs`            | 3     |
| `per_device_train_batch_size`| 1     |
| `gradient_accumulation_steps`| 8     |
| `warmup_steps`                | 100   |
| `max_length`                  | 384   |
| `fp16`                        | True  |

## 📝 Usage Instructions

### Modify Path Configuration

Before use, please modify the following paths according to your environment:

1. **Model Path**: Modify `model_path` in training and inference scripts
2. **Dataset Path**: Modify the path to dataset files
3. **Output Path**: Modify the model save path

### Custom Training

You can adjust the following parameters as needed:
- LoRA's `r`, `alpha`, `dropout` values
- Training epochs, learning rate, batch size
- Maximum sequence length

## 🔍 Benchmark Testing

The project uses the **PsychCounsel-Bench** benchmark test set (500 psychology multiple-choice questions) for model evaluation, supporting multiple prompt formats.

### Prompt Format Description

#### 1. Professional Psychological Counselor Format (Recommended, Highest Accuracy)

```python
You are a professional psychological counselor. Please select the most appropriate answer based on professional psychological knowledge.
Question: {question content}

Options:
A. {option A content}
B. {option B content}
C. {option C content}
D. {option D content}
E. {option E content}
Please only output the option letter (A/B/C/D/E), without any other text.
```

**Characteristics**: Clear role positioning, requires professional psychological knowledge, concise and clear instructions.

**Example**:

```
You are a professional psychological counselor. Please select the most appropriate answer based on professional psychological knowledge.
Question: An individual's self-esteem is most likely to improve when they credit their success to which of the following?

Options:
A. Factors within themselves
B. Factors outside themselves
C. Indirect factors
D. Random factors
E. Unstable factors

Please only output the option letter (A/B/C/D/E), without any other text.
```

#### 2. Simple Direct Format

```python
{question content}

A. {option A content}
B. {option B content}
C. {option C content}
D. {option D content}
E. {option E content}

Please select the correct answer letter:
```

**Characteristics**: Removes role setting and professional requirements, directly presents questions and options, simplest format.

#### 3. Dialogue Format

**System Message**:

```python
You are a professional psychological counselor. Please select the most appropriate answer based on psychological knowledge.
```

**User Message**:
```python
{question content}

Options:
A. {option A content}
B. {option B content}
C. {option C content}
D. {option D content}
E. {option E content}

Please only output the option letter:
```

**Characteristics**: Uses dialogue format, includes system and user roles, closer to training data format.

### Evaluation Results

**Performance Comparison Across Different Formats**:

| Format                          | Original Model Accuracy | Fine-tuned Model Accuracy | Notes                                    |
| :------------------------------ | :---------------------- | :------------------------ | :--------------------------------------- |
| **Professional Counselor Format** | 80.60%                  | 79.60%                    | Highest accuracy, **recommended**       |
| **Simple Direct Format**        | 46.20%                  | 57.20%                    | Medium accuracy, significant improvement after fine-tuning |
| **Dialogue Format**              | 4.00%                   | 2.20%                     | Low accuracy, not suitable for multiple-choice evaluation |

**Overall Evaluation Results**:
- Original model best accuracy: **80.60%** (Professional Counselor Format)
- Fine-tuned model best accuracy: **79.60%** (Professional Counselor Format)
- Detailed results and comparative analysis please refer to the [📊 Comparative Analysis](#-comparative-analysis) section

Evaluation results are saved as JSON files (`benchmark_test_results.json`), containing detailed information such as accuracy and error case analysis.

## 📊 Comparative Analysis

### I. LoRA Fine-tuning Effect Comparison

This section compares the performance of the original Qwen3 model and the model fine-tuned on the PsyDTCorpus dataset.

#### 1.1 Qualitative Dialogue Comparison

**Test Question**: I've been feeling unhappy lately

#### Original Model (without LoRA)

**Model Information**:

| Item          | Value            |
| :------------ | :--------------- |
| ✅ Status     | Model loaded successfully |
| 📱 Device     | cuda:0           |
| 🔢 Parameters | 4,022,468,096    |
| 💾 Memory     | 7.49 GB          |

**Model Response**:

I can sense that you may be experiencing some difficulties right now. Feeling down is a normal human emotion. If you're willing, would you like to talk about what's making you feel unhappy? Sometimes expressing your worries can make you feel lighter.

If you don't want to say much right now, I can also accompany you in doing something relaxing, like listening to music, chatting, or taking a walk together. You're not alone; I'm here with you. If you feel the pressure is too great, you can also consider confiding in a trusted friend or seeking professional psychological counseling help.

I hope you can gradually feel better. Remember to take care of yourself. If needed, I'm always here.

**Analysis**: The response structure is standard, provides multiple support options, but the tone is relatively standardized, showing some characteristics of "psychological counseling templates."

#### Fine-tuned Model (with LoRA)

**Model Information**:

| Item          | Value            |
| :------------ | :--------------- |
| ✅ Status     | Fine-tuned model loaded successfully |
| 📱 Device     | cuda:0           |
| 🔢 Parameters | 4,028,366,336    |
| 💾 Memory     | 7.50 GB          |

**Model Response**:

I can sense that you're feeling down right now, and I'm really sorry to hear that. Sometimes life does bring us a lot of pressure and troubles, making you feel sad and helpless. Would you like to talk with me about what happened? I'm here with you, listening to your thoughts.

If you don't want to talk right away, you can also tell me what kind of support you want: do you want to hear words of comfort, or do you need someone to help you sort things out? Whatever it is, I'll be here to support you. Sometimes talking about it can make you feel lighter, but if you don't want to talk now, I understand. We can also do something else first, like talking about things you've been enjoying lately, or looking at the scenery outside the window.

**Analysis**: More natural empathic expression, actively asks about user needs and gives them choice, tone is gentle and closer to daily communication, achieving a better balance between professionalism and affinity.

#### 1.2 Quantitative Evaluation Results Comparison

We conducted a comprehensive quantitative evaluation of the model using the **PsychCounsel-Bench** benchmark test set (500 psychology multiple-choice questions), testing three different prompt formats.

**Benchmark Test Results Comparison**:

| Prompt Format                  | Original Model Accuracy       | Fine-tuned Model Accuracy     | Improvement   |
| :---------------------------- | :---------------------------- | :---------------------------- | :------------ |
| **Professional Counselor Format** | **80.60%** (403/500)          | **79.60%** (398/500)          | -1.00%        |
| **Simple Direct Format**       | 46.20% (231/500)              | **57.20%** (286/500)          | **+11.00%** ⬆️ |
| **Dialogue Format**            | 4.00% (20/500)                | 2.20% (11/500)                | -1.80%        |

#### Detailed Analysis

1. **Professional Counselor Format (Best Format)**
   - Original model: 80.60% (403/500)
   - Fine-tuned model: 79.60% (398/500)
   - **Analysis**: Both models perform excellently in this format, with accuracy exceeding 79%. The fine-tuned model shows a slight decrease (-1%), which is within statistical error range, basically maintaining the original performance level.

2. **Simple Direct Format**
   - Original model: 46.20% (231/500)
   - Fine-tuned model: 57.20% (286/500)
   - **Analysis**: The fine-tuned model shows **significant improvement** (+11%) in this format, with accuracy increasing by nearly 11 percentage points. This indicates that fine-tuning enhances the model's adaptability to non-professional prompt formats, improving robustness.

3. **Dialogue Format**
   - Original model: 4.00% (20/500)
   - Fine-tuned model: 2.20% (11/500)
   - **Analysis**: Both models show very low accuracy in this format, indicating that dialogue format is not suitable for multiple-choice question type benchmark testing. This format is more suitable for dialogue generation tasks.

#### Performance Evaluation Summary

- ✅ **Overall Performance**: The fine-tuned model maintains high accuracy (79.60%) close to the original model in the best format
- ✅ **Format Robustness**: Fine-tuning significantly improves model performance in simple direct format (+11%), enhancing adaptability to different prompt formats
- ✅ **Performance Rating**: According to evaluation standards, accuracy exceeding 70% is excellent level, fine-tuning effect is significant
- 📊 **Prompt Sensitivity**: Results show that Professional Counselor Format is most suitable for psychology knowledge Q&A tasks, while dialogue format is not suitable for multiple-choice type evaluation

#### Error Case Analysis

Main problem types observed from error cases:
1. **Concept Confusion**: Misunderstanding of professional psychology concepts (e.g., retroactive interference vs proactive interference)
2. **Theory Application**: Judgment errors when applying theories to specific situations
3. **Answer Extraction**: In some cases, the model generates explanations but fails to correctly extract option letters

#### 1.3 LoRA Fine-tuning Effect Summary

Through comparative analysis, we found that the fine-tuned model shows significant improvements in the following aspects:

#### 1. **More Natural Empathic Expression**
- **Fine-tuned Model**: Uses more empathic expressions closer to daily communication (e.g., "I'm really sorry to hear that"), with natural and gentle tone, better creating a safe and supportive dialogue atmosphere
- **Original Model**: Empathic expressions are accurate but slightly templated, showing some characteristics of "psychological counseling scripts"

#### 2. **Respecting User Autonomy** (Key Professional Indicator)
- **Fine-tuned Model**: Actively gives users choice in dialogue direction (e.g., "What kind of support do you want?"), reflecting user-centered psychological counseling principles, which is an important manifestation of professionalism
- **Original Model**: More inclined to gentle guidance, relatively weaker in giving users a sense of control

#### 3. **More Balanced Role Positioning**
- **Fine-tuned Model**: Better balances the roles of "professional counselor" and "supportive companion," maintaining professionalism while enhancing emotional warmth
- **Original Model**: Although more standardized, slightly formal, with less affinity

#### 4. **Technical Indicators**
- Parameter increase: Approximately 5.9 million parameters (LoRA adapter)
- Memory usage: Only increased by about 0.01 GB, almost negligible
- Inference efficiency: No significant difference, maintains high efficiency

#### Comprehensive Conclusion

Combining qualitative dialogue analysis and quantitative benchmark test results, the fine-tuned model performs excellently in multiple dimensions:

**Qualitative Advantages**:
- ✅ **More Natural Empathic Expression**: Dialogue is more authentic and closer to daily communication
- ✅ **Better Respect for User Autonomy**: Conforms to core principles of psychological counseling
- ✅ **More Balanced Role Positioning**: Professionalism and affinity combined

**Quantitative Performance**:
- ✅ **Maintains High Accuracy**: Maintains 79.60% accuracy in the best format (close to original model's 80.60%)
- ✅ **Enhanced Format Robustness**: Significant 11% accuracy improvement in simple direct format (from 46.20% to 57.20%)
- ✅ **Minimal Performance Overhead**: Only increased by approximately 5.9 million parameters (about 0.15%) and 0.01 GB memory usage
- ✅ **Excellent Performance Rating**: Accuracy exceeds 70%, reaching excellent level

**Overall Evaluation**:

LoRA fine-tuning based on the PsyDTCorpus dataset **maintains model professional knowledge accuracy** while significantly improving the model's **professionalism and user experience in psychological counseling dialogue scenarios**. Although accuracy fluctuates slightly in some formats, overall performance is stable, with particularly significant effects in enhancing format robustness and improving dialogue naturalness.

This indicates that this fine-tuning solution effectively balances the two dimensions of **professionalism** (knowledge accuracy) and **dialogue quality** (empathic expression, user autonomy), providing a reliable technical foundation for building professional mental health dialogue systems.

---

### II. RAG Enhancement Effect Comparison

This section compares the complete performance of three models: the original model, the LoRA fine-tuned model, and the fine-tuned model combined with RAG.

#### 2.1 Complete Comparison of Three Models

**Test Example 1: Low Mood Problem**

**Test Question**: I'm unhappy and feel life is meaningless

**🟢 Original Qwen3 Response**:

I can sense that you're going through a difficult time right now. This sense of powerlessness about life can indeed make people feel lost and painful. Please remember, your feelings are real and reasonable; you don't need to blame yourself for this.

At such moments, perhaps we can try a few directions together:

1. **Awareness of the Present**: Try recording three "small good things" that happened today with your phone (even if it's just sunlight coming through the window), which can help us reconnect with the positive aspects of life.
2. **Body Signals**: If you're willing, you can share your recent physical feelings with me (such as insomnia, appetite changes, etc.). These physiological reactions are often signal lights of psychological stress.
3. **Breathing Exercises**: When we feel heavy, deep breathing can bring immediate relief. Try this method: inhale for 4 seconds → hold for 4 seconds → exhale for 6 seconds, repeat several times.
4. **Presence Connection**: Imagine a person/thing that makes you feel safe (like a tree, a stone), and speak your current mood to it in words.
5. **Progressive Action**: Start with the simplest, such as getting up and walking for 5 minutes, or sending a short message to someone.

Would you like to talk with me about whether there have been any particularly touching emotional events recently?

**🔵 Fine-tuned Qwen-Psy Response**:

Dear friend, hearing you say you feel unhappy and life has no meaning, I deeply understand your current mood. This feeling is like being trapped in a dark room, where even breathing becomes heavy. Please allow me to give you a warm hug (expressed through words) and tell you: you are not alone, I am here with you.

When we encounter emotional lows, we often need to first acknowledge the fact that "I am in pain right now." Have you noticed what reactions your body has in this state? Such as rapid heartbeat, stomach discomfort, finger trembling... These physiological reactions are actually sending us signals, indicating that our body and mind are experiencing great stress.

Let me try to understand your feelings in a new way. Imagine you're standing by a river, and the river represents your current emotions. When the water is turbulent, we often panic, but if we can calmly observe the direction of the water flow, we can find a path forward. Would you like to explore this river with me?

Finally, I'd like to ask you, what do you need most right now? Someone to confide in, some advice, or quiet companionship? Whatever you need, I'm willing to do my best for you.

**🔥 Fine-tuned + RAG Response**:

I can sense that you're going through some unpleasant days right now. This numbness about life can indeed make people feel frustrated and helpless. Please remember, your feelings are real and exist; you don't need to feel ashamed or blame yourself for this.

When we fall into such emotional lows, it's often because our hearts have accumulated too much unprocessed pressure and confusion. Would you like to talk with me about what things have been particularly troubling you recently? Have you encountered some setbacks, or have there been some changes in life that make you feel uneasy? Sometimes talking about it can help us see another side of the problem.

If you're willing, we can try to find the root causes of these negative emotions together. While waiting for professional help to arrive, you can also try these gentle self-care practices: when breathing deeply, imagine yourself nourishing your soul like watering a plant; arrange a small happiness for yourself every day, even if it's just five minutes (like listening to favorite music, basking in the sun). These small efforts will gradually restore vitality to your body and mind.

You are not alone; I am here with you to find ways out of the gloom. Shall we start by sorting out what has happened recently?

**Comparison Analysis**:

| Dimension | Original Qwen3 | Fine-tuned Qwen-Psy | Fine-tuned + RAG |
|:----------|:--------------|:-------------------|:-----------------|
| **Empathic Expression** | Gentle, understanding, but slightly templated | Deep empathy, uses many metaphors and imagery | Professional and gentle, combines theoretical knowledge |
| **Language Style** | Structured suggestions, clear logic | Psychological counselor style, rich emotional color | Balances professionalism and affinity |
| **Interaction Method** | Provides specific methods, guides action | Uses metaphors, river analogies, more exploratory | Combines professional knowledge and practical suggestions |
| **Professional Level** | Basic mental health knowledge | Reflects professional psychological counseling skills | Integrates professional knowledge base content |

---

**Test Example 2: Cognitive Distortion Concept Explanation**

**Test Question**: What is cognitive distortion?

**🟢 Original Qwen3 Response**:

Cognitive Distortions refer to **irrational, inaccurate, or biased ways of thinking** that people produce when processing information due to the influence of emotions, experiences, beliefs, and other factors. These distorted thinking patterns can exacerbate negative emotions (such as anxiety, depression) and may lead to unreasonable judgments or behaviors. It is one of the core contents studied in **Cognitive Behavioral Therapy (CBT)**.

**Main types include**: All-or-nothing thinking, overgeneralization, catastrophizing, emotional reasoning, intellectualization, personalization, labeling, excessive perfectionism, etc.

**🔵 Fine-tuned Qwen-Psy Response**:

Dear friend, thank you for being willing to explore your inner world and seek understanding. As a psychological counselor, I'm very happy to explore this important topic with you.

**Cognitive Distortions** refer to unreasonable thinking tendencies that individuals exhibit when perceiving reality and evaluating events. These thinking biases often lead to emotional pain and behavioral problems. They are like a double-edged sword, both helping us make quick responses and potentially trapping us in a vortex of negative emotions.

**Practical Path**:
1. **Awareness Training**: Record situations and thoughts that trigger distortions
2. **Critical Thinking Practice**: Question the evidential support of thoughts
3. **Dialectical Testing Method**: View problems from different angles
4. **Reality Verification Technique**: Assess the likelihood of worst-case scenarios
5. **Thought Restructuring Practice**: Replace absolute statements with specific facts

Cognitive Behavioral Therapy (CBT) is the gold standard for treating cognitive distortions. Psychological counselors typically use "Socratic questioning" to help clients examine their thinking.

**🔥 Fine-tuned + RAG Response**:

Dear friend, the question you raised is very meaningful. In the field of mental health, "cognitive distortion" usually refers to people's irrational and biased understanding and judgment of themselves, others, or events under emotional distress or psychological pressure. This thinking bias can exacerbate negative emotions and behavioral reactions.

From a professional knowledge perspective, patients with mental disorders often exhibit symptoms such as **logical thinking confusion**, **decreased comprehension ability**, and **memory abnormalities**, all of which may manifest as characteristics of cognitive distortion. For example:

1. **Black-and-white Thinking**: Simplifying things into "all good/all bad"
2. **Catastrophic Imagination**: Overestimating negative consequences
3. **Overgeneralization**: Inferring universal conclusions from a single event
4. **Reality Distortion**: Distorting objective reality

These irrational thoughts are like filters that blur the clear picture of the real world. Cognitive distortion itself is not a disease, but rather **a psychological reaction mechanism hidden behind the disease**. Professional treatment (such as cognitive behavioral therapy) helps people restore rational understanding and emotional regulation abilities by identifying and correcting these thinking biases.

**Comparison Analysis**:

| Dimension | Original Qwen3 | Fine-tuned Qwen-Psy | Fine-tuned + RAG |
|:----------|:--------------|:-------------------|:-----------------|
| **Knowledge Completeness** | Comprehensive and systematic, detailed explanation of 8 types | Clear structure, tabular presentation, more professional | Combines professional knowledge base, cites clinical features |
| **Expression Method** | Academic, directly gives definitions and classifications | Counselor style, uses metaphors and guidance | Combines professional terminology with popular explanations |
| **Practicality** | Provides specific coping methods | Provides practical paths and professional advice | Emphasizes the importance of professional treatment |
| **Professional Depth** | Basic knowledge explanation | Reflects counseling skills and intervention methods | Combines mental disorder diagnostic criteria |

---

**Test Example 3: CCMD-3 Diagnostic Criteria**

**Test Question**: According to CCMD-3 standards, how long must a depressive episode last?

**🟢 Original Qwen3 Response**:

According to the "Chinese Classification and Diagnostic Criteria of Mental Disorders" 3rd Edition (CCMD-3), **the diagnosis of a depressive episode requires meeting the following conditions**:

1. **Core Symptoms**: **Persistent low mood for at least 2 weeks or more**
2. **Symptom Severity**: At least 2 major depressive symptoms persist for at least 2 weeks
3. **Exclude Other Causes**: Need to exclude depressive-like symptoms caused by organic diseases, drug factors, etc.

**Duration**: **More than 2 weeks** is the key time standard for diagnosing depressive episodes.

**🔵 Fine-tuned Qwen-Psy Response**:

According to the "Chinese Classification and Diagnostic Criteria of Mental Disorders" 3rd Edition (CCMD-3), **the diagnosis of a depressive episode requires meeting the following key conditions**:

**1. Symptom Criteria**: The patient exhibits depressive core symptoms for **more than 2 consecutive weeks** (usually 2 weeks to 2 months) and meets at least 4 related symptoms.

**2. Severity Grading**:

| Grade | Symptom Duration | Functional Impairment |
|:------|:----------------|:---------------------|
| Mild  | ≥2 weeks        | Social function basically normal, self-care ability impaired |
| Moderate | ≥2 weeks    | Social function significantly impaired, mild self-blame or suicidal tendency |
| Severe | ≥2 weeks        | Social function severely impaired, strong self-blame, suicidal ideation or behavior |

**3. Exclusion Criteria**: Need to exclude organic diseases, drug abuse, mixed states, etc.

**Important Note**: CCMD-3 standards are **important references for clinical diagnosis**, but final diagnosis requires **comprehensive assessment** by **psychiatrists or psychological specialists**.

**🔥 Fine-tuned + RAG Response**:

Dear friend, thank you for raising this question. Regarding the duration of depressive episodes, this indeed needs to be treated with caution, as it involves accurate assessment of mental health status.

According to the "Chinese Classification and Diagnostic Criteria of Mental Disorders" 3rd Edition (CCMD-3), the core symptoms of depressive episodes include significant depressive mood, loss of interest, etc. These symptoms need to persist for at least **more than two weeks** to be diagnosed as a depressive episode. However, it should be noted that this time standard is the basis for clinical diagnosis. In practical application, it also needs to be combined with factors such as symptom severity, daily functional impairment, and exclusion of other causes that may lead to depression for comprehensive judgment.

The observation and diagnosis of depression-related disorders require a long follow-up period (such as several years), which indicates that mental health problems often have hidden and recurrent characteristics. If you or someone around you is experiencing symptoms such as low mood and loss of interest, it is recommended to seek help from professional psychiatrists or psychotherapists as soon as possible. They can determine the nature and severity of symptoms through systematic assessment and develop appropriate intervention plans.

**Comparison Analysis**:

| Dimension | Original Qwen3 | Fine-tuned Qwen-Psy | Fine-tuned + RAG |
|:----------|:--------------|:-------------------|:-----------------|
| **Professionalism** | Accurately gives CCMD-3 standards, clear structure | Details symptom criteria, grading, and exclusion criteria | Combines professional knowledge base, emphasizes clinical practice |
| **Accuracy** | Clearly states "more than 2 weeks" time standard | Details various symptoms and grading | Answers based on retrieved professional knowledge |
| **Expression Method** | Direct, academic | Counselor style, tabular presentation | Gentle, professional, combines psychological care |
| **Practicality** | Provides suggestions and screening tools | Provides detailed diagnostic criteria and precautions | Emphasizes the importance of professional assessment |

#### 2.2 RAG Enhancement Effect Summary

**Overall Comparison**:

| Model Type | Advantages | Characteristics | Applicable Scenarios |
|:-----------|:-----------|:----------------|:---------------------|
| **Original Qwen3** | Comprehensive knowledge, clear structure, direct answers | Language slightly templated, lacks professional skills | General Q&A |
| **Fine-tuned Qwen-Psy** | Deep empathy, professional counseling skills | Psychological counselor style, rich emotional color | Daily emotional support, counseling dialogue |
| **Fine-tuned + RAG** | Combines professional knowledge base, more accurate and comprehensive answers | Balances professionalism and affinity | Professional concept explanation, diagnostic criteria queries |

**Application Recommendations**:

1. **Daily Emotional Support**: Recommend using **Fine-tuned Qwen-Psy** model, as its deep empathy and counseling skills are more suitable
2. **Professional Concept Explanation**: Recommend using **Fine-tuned + RAG** model, which can combine professional knowledge base to give more accurate answers
3. **Diagnostic Criteria Queries**: Recommend using **Fine-tuned + RAG** model, which can retrieve relevant diagnostic criteria documents

**RAG Enhancement Advantages**:

- ✅ **Enhanced Professionalism**: Directly retrieves relevant knowledge from authoritative psychology books
- ✅ **Improved Accuracy**: Combines retrieved professional content to reduce model hallucinations
- ✅ **Knowledge Updates**: Can supplement the latest professional knowledge by updating the knowledge base
- ✅ **Explainability**: Can trace back to specific document sources retrieved

## 📊 Project Features Explained

### Data Format Processing

The project uses Qwen conversation format for data preprocessing:

```python
def format_conversation(messages):
    """Format conversation to Qwen format"""
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

### LoRA Parameter Efficiency

Using LoRA technology significantly reduces the number of trainable parameters:
- Only trains LoRA adapters for attention layers (Q, K, V, O projection layers)
- Most model parameters are frozen, only a small number of parameters are updated
- The trained model file size is much smaller than full fine-tuning

### Evaluation Metrics

Uses PsychCounsel-Bench for multiple-choice question evaluation:
- Automatically extracts answer options (A/B/C/D/E) generated by the model
- Supports comparison of multiple prompt formats
- Outputs detailed accuracy and error case analysis

### RAG Retrieval-Augmented Generation

**Working Principle**:
- Split professional knowledge base (PDF documents) into small document chunks
- Use embedding model to convert document chunks into vector representations
- Store vectors in FAISS vector database
- During query, convert user questions into vectors and retrieve the most similar document chunks
- Use retrieved document chunks as context, input together with user questions to the model

**Technology Selection**:
- **LangChain**: Provides complete RAG pipeline including document loading, splitting, vectorization, etc.
- **FAISS**: Efficient vector similarity search library
- **BGE-small-zh-v1.5**: Lightweight embedding model optimized for Chinese

**Applicable Scenarios**:
- Q&A requiring citation of specific professional knowledge
- Professional consultation scenarios requiring high accuracy
- Applications requiring traceable information sources

## ⚠️ Important Notes

1. **Memory Requirements**:
   - Training: Recommend using GPU with at least 16GB memory (such as A100, V100, RTX 3090, etc.)
   - Inference: Recommend at least 8GB memory
   - Using FP16 mixed precision training can reduce memory usage

2. **Model Path**:
   - Ensure the Qwen3-4B model is correctly downloaded locally
   - Models can be downloaded through ModelScope or Hugging Face
   - All paths in scripts need to be modified to actual paths

3. **Dataset Format**:
   - Dataset must conform to OpenAI format
   - Each sample contains `id`, `normalizedTag`, `messages` fields
   - `messages` needs to contain `system`, `user`, `assistant` roles

4. **Dependency Versions**:
   - **Important**: `transformers` version must be >= 4.51.0 (supports Qwen3)
   - Lower versions will error: `KeyError: 'qwen3'`

5. **Training Recommendations**:
   - Checkpoints will be saved every 200 steps during training
   - Final model is saved in `lora/training/qwen-psy-trained/` directory
   - You can use `lora/data/seeData.py` to view the dataset structure first

6. **Inference Parameters**:
   - Non-thinking mode recommended parameters: `Temperature=0.7`, `TopP=0.8`, `TopK=20`
   - Thinking mode recommended parameters: `Temperature=0.6`, `TopP=0.95`, `TopK=20`
   - Do not use greedy decoding, which may cause performance degradation and infinite repetition

7. **RAG Function Usage**:
   - First use requires running `rag/indexing.py` to build vector index
   - Ensure PDF files are in the `rag/knowledge_base/` directory
   - Vector index will be saved in `rag/vector_store/psychology_db/` directory
   - If knowledge base is updated, need to rerun the index building script
   - RAG dependencies (langchain, faiss, etc.) are optional, only install when using RAG functionality

## 📄 License

This project follows the following licenses:
- **Project Code**: Apache 2.0 License
- **Base Model Qwen3-4B**: Apache 2.0 License (refer to [Qwen3-4B LICENSE](https://huggingface.co/Qwen/Qwen3-4B/blob/main/LICENSE))
- **Dataset PsyDTCorpus**: Please refer to the original dataset license

## 👥 Authors

- **Yanze Jiang**

## 🤝 Contributing

Welcome to submit Issues and Pull Requests to help improve the project!

## 🙏 Acknowledgments

- Thanks to the [Qwen](https://github.com/QwenLM/Qwen) team for providing the excellent base model
- Thanks to the contributors of the PsyDTCorpus dataset
- Thanks to the [PEFT](https://github.com/huggingface/peft) project for providing the LoRA implementation

## 📚 Related Resources

- [Qwen Models](https://github.com/QwenLM/Qwen)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Transformers Documentation](https://huggingface.co/docs/transformers)

## 📮 Contact

For questions or suggestions, please contact through:
- Submit GitHub Issue
- Send email (please provide contact information)

---

**Note**: This project is for research and educational purposes only. The content generated by the model should not replace professional psychological counseling services.
