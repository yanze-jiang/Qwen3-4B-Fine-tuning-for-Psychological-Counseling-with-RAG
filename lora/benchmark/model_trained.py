import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import re

# ================= 配置 =================
BASE_MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-4B"
LORA_PATH = "/root/autodl-tmp/qwen-psy-trained"
BENCHMARK_PATH = "/root/autodl-tmp/PsychCounsel-Bench.json"
MAX_NEW_TOKENS = 50
# ========================================

def load_benchmark(path):
    """加载benchmark数据"""
    print("📥 加载 benchmark 数据...")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"共 {len(data)} 道题")
    return data

def build_psychology_prompt(item):
    """
    构建心理咨询专业提示词
    格式：明确角色 + 专业要求 + 简洁指令
    """
    question = item["question"]
    options = item["options"]

    # 方法1：专业心理咨询师格式（推荐）
    prompt = """你是一位专业的心理咨询师，请基于心理学专业知识选择最合适的答案。
问题：{}

选项：
{}
请只输出选项字母（A/B/C/D/E），不要有任何其他文字。""".format(
        question,
        "\n".join([f"{key.upper()}. {options[key]}" for key in sorted(options.keys())])
    )

    return prompt

def build_simple_prompt(item):
    """
    简单直接的提示词
    """
    question = item["question"]
    options = item["options"]

    prompt = f"{question}\n\n"
    for key in sorted(options.keys()):
        prompt += f"{key.upper()}. {options[key]}\n"

    prompt += "\n请选择正确答案的字母："
    return prompt

def build_chat_format_prompt(item):
    """
    使用对话格式，更接近训练时的格式
    """
    question = item["question"]
    options = item["options"]

    user_content = f"{question}\n\n选项：\n"
    for key in sorted(options.keys()):
        user_content += f"{key.upper()}. {options[key]}\n"
    user_content += "\n请只输出选项字母："

    # 构建messages
    messages = [
        {"role": "system", "content": "你是一位专业的心理咨询师，请根据心理学知识选择最合适的答案。"},
        {"role": "user", "content": user_content}
    ]

    return messages

def extract_answer_v2(text):
    """
    更强大的答案提取函数
    """
    if not text:
        return None

    text = text.strip()

    # 清理常见前缀
    prefixes = ["回答：", "答案是", "选择", "我认为是", "选项", "答：", "answer:", "answer is"]
    for prefix in prefixes:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()

    # 提取括号或点号后的字母
    patterns = [
        r'^([A-Ea-e])[).\s]*',
        r'答案是\s*([A-Ea-e])',
        r'选择\s*([A-Ea-e])',
        r'选项\s*([A-Ea-e])',
        r'我认为是\s*([A-Ea-e])',
        r'正确答案是\s*([A-Ea-e])',
        r'\b([A-Ea-e])\b',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            answer = match.group(1).lower()
            if answer in ['a', 'b', 'c', 'd', 'e']:
                return answer

    # 如果还没找到，检查第一个字符
    if text and text[0].lower() in ['a', 'b', 'c', 'd', 'e']:
        return text[0].lower()

    return None

def load_finetuned_model():
    """加载微调后的模型"""
    print("🤖 加载微调后模型...")

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        use_cache=False
    )

    # 加载LoRA适配器
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.eval()

    print(f"模型加载完成，设备: {model.device}")
    return model, tokenizer

def evaluate_with_prompt_type(model, tokenizer, benchmark, prompt_type="psychology"):
    """
    使用指定类型的prompt进行评估
    """
    print(f"\n🚀 使用 '{prompt_type}' 提示词进行评估...")

    correct = 0
    total = len(benchmark)
    wrong_details = []

    for idx, item in enumerate(tqdm(benchmark, desc="评测进度")):
        # 选择提示词类型
        if prompt_type == "simple":
            prompt = build_simple_prompt(item)
            use_chat_template = False
        elif prompt_type == "chat":
            messages = build_chat_format_prompt(item)
            use_chat_template = True
        else:
            prompt = build_psychology_prompt(item)
            use_chat_template = False

        # 生成回答
        if use_chat_template:
            try:
                prompt_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except:
                prompt_text = build_simple_prompt(item)
        else:
            prompt_text = prompt

        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=0.1,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.0,
            )

        generated_text = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()

        # 提取答案
        pred_answer = extract_answer_v2(generated_text)
        true_answer = item["answer"].lower().strip()

        # 记录结果
        if pred_answer == true_answer:
            correct += 1
        else:
            wrong_details.append({
                "index": idx,
                "question": item["question"][:100],
                "true_answer": true_answer,
                "predicted": pred_answer,
                "generated": generated_text[:200],
                "options": item["options"]
            })

        # 每100题清理一次显存
        if idx % 100 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    accuracy = correct / total * 100
    return accuracy, correct, total, wrong_details

def main():
    print("="*60)
    print("🧠 心理咨询模型benchmark测试")
    print("="*60)

    # 1. 加载数据
    benchmark = load_benchmark(BENCHMARK_PATH)

    # 2. 加载模型
    model, tokenizer = load_finetuned_model()

    # 3. 测试不同提示词的效果
    results = {}

    prompt_types = [
        ("psychology", "专业心理咨询师格式"),
        ("simple", "简单直接格式"),
        ("chat", "对话格式")
    ]

    best_accuracy = 0
    best_prompt_type = ""

    for prompt_type, description in prompt_types:
        print(f"\n🔹 测试提示词: {description}")

        accuracy, correct, total, wrong_details = evaluate_with_prompt_type(
            model, tokenizer, benchmark, prompt_type
        )

        results[prompt_type] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "description": description,
            "wrong_details": wrong_details[:10]
        }

        print(f"   准确率: {accuracy:.2f}% ({correct}/{total})")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_prompt_type = prompt_type

    # 4. 输出最佳结果
    print("\n" + "="*60)
    print("📊 最终结果汇总")
    print("="*60)

    best_result = results[best_prompt_type]
    print(f"🎯 最佳提示词: {best_result['description']}")
    print(f"🏆 最佳准确率: {best_result['accuracy']:.2f}%")
    print(f"✅ 正确数: {best_result['correct']} / {best_result['total']}")

    # 5. 分析错误案例
    if best_result['wrong_details']:
        print(f"\n🔍 前5个错误案例分析:")
        for i, error in enumerate(best_result['wrong_details'][:5]):
            print(f"\n  {i+1}. 问题: {error['question']}...")
            print(f"     正确答案: {error['true_answer'].upper()}")
            print(f"     模型预测: {error['predicted'].upper() if error['predicted'] else '无'}")
            print(f"     模型输出: {error['generated']}")
            print(f"     选项: {error['options']}")

    # 6. 保存结果
    output_file = "./benchmark_test_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "results": results,
            "best_prompt_type": best_prompt_type,
            "best_accuracy": best_accuracy,
            "test_config": {
                "base_model": BASE_MODEL_PATH,
                "lora_path": LORA_PATH,
                "benchmark": BENCHMARK_PATH,
                "max_new_tokens": MAX_NEW_TOKENS
            }
        }, f, ensure_ascii=False, indent=2)

    print(f"\n💾 详细结果已保存到: {output_file}")

    # 7. 性能评估
    print(f"\n📈 性能评估:")
    if best_accuracy > 70:
        print("  🌟 优秀 - 准确率超过70%，微调效果显著")
    elif best_accuracy > 60:
        print("  👍 良好 - 准确率超过60%，有明显提升")
    elif best_accuracy > 50:
        print("  ⚠️ 一般 - 准确率超过50%，但仍有提升空间")
    else:
        print("  ❌ 较差 - 需要重新评估微调策略")

if __name__ == "__main__":
    main()