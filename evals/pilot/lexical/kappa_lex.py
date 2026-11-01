import json


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum(a.count(l) * b.count(l) for l in set(a) | set(b)) / n**2
    return (po - pe) / (1 - pe)


rows = [json.loads(line) for line in open("pairs.jsonl")]
r1 = [r["rater_1"] for r in rows]
r2 = [r["rater_2"] for r in rows]
qwen = [r["qwen"] for r in rows]
astra = [r["astra"] for r in rows]

print(f"human-human kappa: {kappa(r1, r2):.2f}")
print(f"human-Qwen kappa: {kappa(r1, qwen):.2f}")
print(f"human-Astra kappa: {kappa(r1, astra):.2f}")
