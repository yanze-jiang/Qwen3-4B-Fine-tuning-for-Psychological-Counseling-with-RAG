import json
from collections import Counter

path = "lora/data/PsyDTCorpus/PsyDTCorpus_train_mulit_turn_packing.json"  # 按实际路径改
data = json.load(open(path, "r", encoding="utf-8"))

c = Counter(x.get("normalizedTag","UNK") for x in data)
print("N =", len(data))
for k,v in c.most_common(20):
    print(k, v)