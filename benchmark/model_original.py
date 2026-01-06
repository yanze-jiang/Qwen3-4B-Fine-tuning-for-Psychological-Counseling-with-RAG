import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

# ================= 配置 =================
BASE_MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-4B"
LORA_PATH = "/root/autodl-tmp/qwen-psy-trained"  # 保留但不使用
BENCHMARK_PATH = "/root/autodl-tmp/PsychCounsel-Bench.json"
MAX_NEW_TOKENS = 50
# ========================================

def load_benchmark(path):
    """加载 benchmark 数据"""
    print("📥 加载 benchmark 数据...")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"共 {len(data)} 道题")
    return data

def build_psychology_prompt(item):
    """专业心理咨询师提示词"""
    question = item["question"]
    options = item["options"]

    prompt = """你是一位专业的心理咨询师，请基于心理学专业知识选择最合适的答案。
问题：{}

选项：
{}
请只输出选项字母（A/B/C/D/E），不要有任何其他文字。""".format(
        question,
        "\n".join([f"{k.upper()}. {options[k]}" for k in sorted(options.keys())])
    )
    return prompt

def build_simple_prompt(item):
    """简单直接提示词"""
    question = item["question"]
    options = item["options"]

    prompt = f"{question}\n\n"
    for k in sorted(options.keys()):
        prompt += f"{k.upper()}. {options[k]}\n"
    prompt += "\n请选择正确答案的字母："
    return prompt

def build_chat_format_prompt(item):
    """对话格式提示词"""
    question = item["question"]
    options = item["options"]

    user_content = f"{question}\n\n选项：\n"
    for k in sorted(options.keys()):
        user_content += f"{k.upper()}. {options[k]}\n"
    user_content += "\n请只输出选项字母："

    messages = [
        {"role": "system", "content": "你是一位专业的心理咨询师，请根据心理学知识选择最合适的答案。"},
        {"role": "user", "content": user_content}
    ]
    return messages

def extract_answer_v2(text):
    """答案提取"""
    if not text:
        return None

    text = text.strip()

    prefixes = ["回答：", "答案是", "选择", "我认为是", "选项", "答：", "answer:", "answer is"]
    for p in prefixes:
        if text.lower().startswith(p.lower()):
            text = text[len(p):].strip()

    patterns = [
        r'^([A-Ea-e])[).\s]*',
        r'答案是\s*([A-Ea-e])',
        r'选择\s*([A-Ea-e])',
        r'\b([A-Ea-e])\b',
    ]

    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).lower()

    if text and text[0].lower() in "abcde":
        return text[0].lower()

    return None

def load_finetuned_model():
    """加载微调前的基础模型"""
    print("🤖 加载微调前基础模型...")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        use_cache=False
    )

    model.eval()
    print(f"模型加载完成，设备: {model.device}")
    return model, tokenizer

def evaluate_with_prompt_type(model, tokenizer, benchmark, prompt_type):
    print(f"\n🚀 使用 '{prompt_type}' 提示词进行评估...")
    correct = 0
    wrong_details = []

    for idx, item in enumerate(tqdm(benchmark, desc="评测进度")):
        if prompt_type == "simple":
            prompt_text = build_simple_prompt(item)
        elif prompt_type == "chat":
            messages = build_chat_format_prompt(item)
            try:
                prompt_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except:
                prompt_text = build_simple_prompt(item)
        else:
            prompt_text = build_psychology_prompt(item)

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
            )

        gen_text = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        pred = extract_answer_v2(gen_text)
        gold = item["answer"].lower()

        if pred == gold:
            correct += 1
        else:
            wrong_details.append({
                "index": idx,
                "question": item["question"][:100],
                "true": gold,
                "pred": pred,
                "gen": gen_text[:200]
            })

        if idx % 100 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    acc = correct / len(benchmark) * 100
    return acc, correct, len(benchmark), wrong_details

def main():
    print("=" * 60)
    print("🧠 心理咨询模型 Benchmark 测试（Base Model）")
    print("=" * 60)

    benchmark = load_benchmark(BENCHMARK_PATH)
    model, tokenizer = load_finetuned_model()

    prompt_types = [
        ("psychology", "专业心理咨询师格式"),
        ("simple", "简单直接格式"),
        ("chat", "对话格式")
    ]

    results = {}
    best_acc = 0
    best_prompt = ""

    for ptype, desc in prompt_types:
        acc, correct, total, wrong = evaluate_with_prompt_type(
            model, tokenizer, benchmark, ptype
        )

        results[ptype] = {
            "accuracy": acc,
            "correct": correct,
            "total": total,
            "description": desc
        }

        print(f"{desc} | 准确率: {acc:.2f}% ({correct}/{total})")

        if acc > best_acc:
            best_acc = acc
            best_prompt = ptype

    print("\n🎯 最佳提示词:", results[best_prompt]["description"])
    print(f"🏆 最佳准确率: {best_acc:.2f}%")

    with open("benchmark_test_results_base.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("💾 结果已保存：benchmark_test_results_base.json")

if __name__ == "__main__":
    main()