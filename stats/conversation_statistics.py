import json
import statistics


def avg(values):
    return sum(values) / len(values) if values else 0


def main():
    path = "lora/data/PsyDTCorpus/PsyDTCorpus_train_mulit_turn_packing.json"  # 按实际路径改
    data = json.load(open(path, "r", encoding="utf-8"))

    turns = []
    user_lens, asst_lens = [], []
    for ex in data:
        msgs = ex["messages"]
        turns.append(len(msgs))
        for m in msgs:
            c = m.get("content", "")
            if m.get("role") == "user":
                user_lens.append(len(c))
            elif m.get("role") == "assistant":
                asst_lens.append(len(c))

    print("N =", len(data))
    print(
        "Turns: avg =",
        round(avg(turns), 2),
        "median =",
        float(statistics.median(turns)),
        "min =",
        min(turns),
        "max =",
        max(turns),
    )
    print("User chars: avg =", round(avg(user_lens), 2), "total =", sum(user_lens))
    print("Asst chars: avg =", round(avg(asst_lens), 2), "total =", sum(asst_lens))
    tot = sum(user_lens) + sum(asst_lens)
    print(
        "User share =",
        round(sum(user_lens) / tot * 100, 2),
        "%; Asst share =",
        round(sum(asst_lens) / tot * 100, 2),
        "%",
    )


if __name__ == "__main__":
    main()

