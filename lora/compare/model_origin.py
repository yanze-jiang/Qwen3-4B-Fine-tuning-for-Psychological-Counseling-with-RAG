# local_qwen_inference.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os

def main():
    # ✅ 模型路径（可以使用 HuggingFace 模型名称或本地路径）
    model_path = "Qwen/Qwen3-4B"  # 或改为本地路径，如 "../../model_origin/Qwen3-4B"
    
    if not os.path.exists(model_path):
        print(f"❌ 错误: 模型路径不存在: {model_path}")
        return
    
    print(f"📁 模型路径: {model_path}")
    print("正在加载模型和分词器...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        print("✅ 模型加载成功!")
        print(f"📱 设备: {model.device}")
        print(f"🔢 参数量: {model.num_parameters():,}")
        print(f"💾 显存占用: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return
    
    def generate_response(question):
        messages = [{"role": "user", "content": question}]
        
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except:
            prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=3000,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        return tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
    
    print("\n🔹 测试问题：这段时间我不开心")
    print("-" * 50)
    print(generate_response("这段时间我不开心"))
    print("-" * 50)

if __name__ == "__main__":
    main()