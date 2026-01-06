from modelscope import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-4B"

# 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# 准备模型输入
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True  # 切换思考模式，默认为 True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# 执行文本生成
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
)

# 解析生成的文本
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

# 解析思考内容
try:
    # 查找 151668 (</think>) 标记
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

# 打印结果
print("thinking content:", thinking_content)
print("content:", content)
