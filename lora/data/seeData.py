import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
from transformers import DataCollatorForLanguageModeling

# 设置模型路径（可以使用 HuggingFace 模型名称或本地路径）
model_path = "Qwen/Qwen3-4B"  # 或改为本地路径

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# 加载数据集并检查结构
print("加载数据集...")
# 使用相对路径（相对于当前脚本位置）
train_dataset = load_dataset("json", data_files="./PsyDTCorpus/PsyDTCorpus_train_mulit_turn_packing.json")
val_dataset = load_dataset("json", data_files="./PsyDTCorpus/PsyDTCorpus_test_single_turn_split.json")

# 查看数据集结构
print("\n训练集结构:")
print(train_dataset)
print("\n验证集结构:")
print(val_dataset)

# 查看可用的keys
print("\n训练集keys:", train_dataset.keys())
print("验证集keys:", val_dataset.keys())

# 查看一个数据样例
print("\n训练集样例:")
print(train_dataset['train'][0] if 'train' in train_dataset else train_dataset[next(iter(train_dataset.keys()))][0])

print("\n验证集样例:")
print(val_dataset['test'][0] if 'test' in val_dataset else val_dataset[next(iter(val_dataset.keys()))][0])
