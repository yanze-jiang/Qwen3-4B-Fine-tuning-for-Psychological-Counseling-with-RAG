import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from rag.retrieval import PsyRetriever
import os

# --- 路径配置 ---
BASE_MODEL_PATH = "Qwen/Qwen3-4B"  # 或改为本地路径，如 "../model_origin/Qwen3-4B"
LORA_PATH = "../lora/training/qwen-psy-trained"  # LoRA 适配器路径

class ComparisonExperiment:
    def __init__(self):
        print("🚀 正在初始化心理学对比实验...")
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
        
        # 加载基础模型 (使用 bfloat16 提高精度并节省显存)
        print("📦 加载基座模型...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH, 
            torch_dtype=torch.bfloat16, 
            device_map="auto", 
            trust_remote_code=True
        )
        
        # 挂载微调参数
        print("💉 挂载 LoRA 适配器...")
        self.psy_model = PeftModel.from_pretrained(self.base_model, LORA_PATH)
        
        # 初始化 RAG 检索器
        print("🔍 初始化 RAG 检索器...")
        self.retriever = PsyRetriever()

    def clean_output(self, text):
        """移除推理过程，只保留最终回复"""
        if "</think>" in text:
            return text.split("</think>")[-1].strip()
        return text.strip()

    def generate(self, prompt):
        """通用生成函数，配置了长文本参数"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.psy_model.device)
        with torch.no_grad():
            outputs = self.psy_model.generate(
                **inputs, 
                max_new_tokens=1500,      # 深度长回复支持
                temperature=0.8,          # 保持咨询师的语言灵活性
                top_p=0.95,
                repetition_penalty=1.1,   # 适度惩罚重复
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        full_res = self.tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        return self.clean_output(full_res)

    def run_compare(self, user_input):
        print("\n" + "━"*60)
        print(f"🌟 用户提问: {user_input}")
        print("━"*60)

        # 1. 原始 Qwen3 模式
        with self.psy_model.disable_adapter():
            prompt_base = f"<|im_start|>system\n你是一位专业的心理咨询师。<|im_end|>\n<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
            res_base = self.generate(prompt_base)
            print(f"🟢 [1. 原始 Qwen3]:\n{res_base}\n")

        # 2. 微调后 Qwen-Psy 模式
        # 此时 adapter 会自动恢复启用
        res_psy = self.generate(prompt_base)
        print(f"🔵 [2. 微调后 Qwen-Psy]:\n{res_psy}\n")

        # 3. 微调 + RAG 模式
        context = self.retriever.get_relevant_context(user_input)
        
        # 优化后的 RAG 系统提示词：区分事实查询与情感安抚
        rag_system = (
            f"你是一位资深心理学专家。参考资料如下：\n{context}\n"
            "要求：\n1. 如果用户询问专业概念，请优先基于资料给出准确详尽的定义。\n"
            "2. 如果用户表达情绪，请在参考资料的基础上，用长篇幅进行深度共情和温暖引导。"
        )
        
        prompt_rag = f"<|im_start|>system\n{rag_system}<|im_end|>\n<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
        res_rag = self.generate(prompt_rag)
        print(f"🔥 [3. 微调 + RAG]:\n{res_rag}\n")

if __name__ == "__main__":
    exp = ComparisonExperiment()
    
    # 典型测试用例
    test_queries