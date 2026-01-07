# local_qwen_inference.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

def main():
    # ✅ 微调后的模型路径（LoRA 适配器）
    base_model_path = "Qwen/Qwen3-4B"  # 基座模型路径
    lora_path = "../training/qwen-psy-trained"  # LoRA 适配器路径
    
    if not os.path.exists(lora_path):
        print(f"❌ LoRA 适配器路径不存在: {lora_path}")
        return
    
    print(f"📁 使用基座模型: {base_model_path}")
    print(f"📁 使用 LoRA 适配器: {lora_path}")
    print("正在加载分词器和模型...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 先加载基座模型
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 然后加载 LoRA 适配器
        model = PeftModel.from_pretrained(base_model, lora_path)
        
        model.eval()
        
        print("✅ 微调模型加载成功!")
        print(f"📱 设备: {model.device}")
        print(f"🔢 参数量: {model.num_parameters():,}")
        print(f"💾 显存占用: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return
    
    def generate_response(question: str):
        messages = [{"role": "user", "content": question}]
        
        # ✅ 关闭 thinking（非常重要）
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            thinking=False
        )
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
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
    print("-" * 60)
    print(generate_response("这段时间我不开心"))
    print("-" * 60)

if __name__ == "__main__":
    main()