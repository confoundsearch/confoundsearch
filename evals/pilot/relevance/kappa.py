import json


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) * b.count(l)) for l in set(a) | set(b)) / n**2
    return (po - pe) / (1 - pe)


rows = [json.loads(line) for line in open("passage_annotation.jsonl")]

k = kappa([r["rater_1"] for r in rows], [r["rater_2"] for r in rows])
print(f"passage-level kappa: {k:.2f}")

both = [r for r in rows if r["rater_1"] == r["rater_2"] == "supports"]
po = sum(r["sentence_1"] == r["sentence_2"] for r in both) / len(both)
pe = sum(1 / r["n_sentences"] for r in both) / len(both)
print(f"p(same sentence): {po:.0%}")
print(f"mean sentences/passage: {sum(r['n_sentences'] for r in both) / len(both):.1f}")
print(f"sentence-level kappa: {(po - pe) / (1 - pe):.2f}")
