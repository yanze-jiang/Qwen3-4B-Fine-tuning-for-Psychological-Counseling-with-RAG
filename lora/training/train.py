import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
from transformers import DataCollatorForLanguageModeling

# 设置模型路径（可以使用 HuggingFace 模型名称或本地路径）
# 如果本地有缓存，会自动使用；否则会从 HuggingFace 下载
model_path = "Qwen/Qwen3-4B"  # 或改为本地路径，如 "../../model_origin/Qwen3-4B"

print("加载分词器...")
# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("加载模型...")
# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    use_cache=False
)

print("加载数据集...")
# 加载数据集 - 使用相对路径
# 根据实际数据集文件位置调整路径
train_data = load_dataset("json", data_files="../data/PsyDTCorpus/PsyDTCorpus_train_mulit_turn_packing.json")
test_data = load_dataset("json", data_files="../data/PsyDTCorpus/PsyDTCorpus_test_single_turn_split.json")

print(f"训练集大小: {len(train_data['train'])}")
print(f"测试集大小: {len(test_data['train'])}")

# 数据预处理函数
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

def preprocess_function(examples):
    texts = [format_conversation(msg) for msg in examples["messages"]]
    
    # Tokenize
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=384,
        return_tensors=None
    )
    
    # 创建labels
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

print("\n预处理数据集...")
# 处理训练集
train_dataset = train_data.map(
    preprocess_function,
    batched=True,
    batch_size=4,
    remove_columns=train_data['train'].column_names,
    desc="处理训练集"
)

# 处理测试集
test_dataset = test_data.map(
    preprocess_function,
    batched=True,
    batch_size=4,
    remove_columns=test_data['train'].column_names,
    desc="处理测试集"
)

print(f"训练样本数: {len(train_dataset['train'])}")
print(f"测试样本数: {len(test_dataset['train'])}")

# LoRA配置
print("\n配置LoRA...")
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    bias="none",
)

# 应用LoRA
model = get_peft_model(model, lora_config)
print("\n模型可训练参数统计:")
model.print_trainable_parameters()

# 数据整理器
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# 计算总训练步数
batch_size = 1
gradient_accumulation_steps = 8
effective_batch_size = batch_size * gradient_accumulation_steps
total_train_steps = (len(train_dataset['train']) * 3) // effective_batch_size  # 3个epoch

print(f"\n训练参数计算:")
print(f"  批次大小: {batch_size}")
print(f"  梯度累积步数: {gradient_accumulation_steps}")
print(f"  有效批次大小: {effective_batch_size}")
print(f"  总训练步数 (~): {total_train_steps}")

# 训练参数 - 修正版本：确保save_steps是eval_steps的整数倍
training_args = TrainingArguments(
    output_dir="./qwen-psy-training",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    warmup_steps=100,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    
    # 关键：评估和保存步数设为相同或整数倍关系
    eval_strategy="steps",
    eval_steps=200,          # 每200步评估一次
    save_strategy="steps",
    save_steps=200,          # 改为200，与eval_steps相同
    
    save_total_limit=3,
    load_best_model_at_end=True,  # 现在可以启用了
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    
    report_to="none",
    remove_unused_columns=False,
    optim="adamw_torch",
    dataloader_num_workers=2,
)

print("\n创建Trainer...")
# 创建Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset["train"],
    eval_dataset=test_dataset["train"],  # 使用测试集进行评估
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# 开始训练
print("\n开始训练...")
trainer.train()

# 保存最终模型
print("\n保存模型...")
trainer.save_model("./qwen-psy-trained")
tokenizer.save_pretrained("./qwen-psy-trained")

print("\n训练完成！")

# 测试生成
print("\n测试生成...")
model.eval()

# 准备测试输入
test_prompts = [
    "最近感觉压力很大，睡不着觉，该怎么办？",
    "我和家人关系不好，经常吵架，我该怎么改善？",
]

for i, prompt in enumerate(test_prompts):
    print(f"\n{'='*50}")
    print(f"测试 {i+1}: {prompt}")
    
    test_messages = [
        {"role": "user", "content": prompt}
    ]
    
    # 格式化
    test_text = format_conversation(test_messages)
    
    inputs = tokenizer(test_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"回答:\n{response}")

print(f"\n{'='*50}")
print("所有测试完成！")
