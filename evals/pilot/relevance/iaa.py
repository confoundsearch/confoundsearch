import json
from itertools import combinations

rows = [json.loads(l) for l in open("passage_sentence_ratings.jsonl")]
raters = ["rater_1", "rater_2", "rater_3"]


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum(a.count(l) * b.count(l) for l in set(a) | set(b)) / n**2
    return (po - pe) / (1 - pe)


def star(n):
    return "*" if n < 10 else ""  # a small sample: fewer than 10 items both raters did


print("Passage-level kappa:")
for j1, j2 in combinations(raters, 2):
    both = [r for r in rows if r[f"{j1}_q1"] and r[f"{j2}_q1"]]
    k = kappa([r[f"{j1}_q1"] for r in both], [r[f"{j2}_q1"] for r in both])
    print(f"  {j1} vs {j2}: {k:.2f}{star(len(both))} (n={len(both)})")

print("\nSentence-level kappa (both said supports):")
agreed = pairs = sentences = 0
for j1, j2 in combinations(raters, 2):
    both = [
        r
        for r in rows
        if r[f"{j1}_q1"] == r[f"{j2}_q1"] == "supports"
        and r[f"{j1}_q2"] is not None
        and r[f"{j2}_q2"] is not None
    ]
    n = len(both)
    po = sum(r[f"{j1}_q2"] == r[f"{j2}_q2"] for r in both) / n
    pe = sum(1 / r["n_sentences"] for r in both) / n
    print(f"  {j1} vs {j2}: {(po - pe) / (1 - pe):.2f}{star(n)} (n={n})")
    agreed += sum(r[f"{j1}_q2"] == r[f"{j2}_q2"] for r in both)
    pairs += n
    sentences += sum(r["n_sentences"] for r in both)

print(f"\nSame sentence, all pairs: {agreed}/{pairs} = {agreed / pairs:.0%}")
print(f"Mean sentences per passage, those pairs: {sentences / pairs:.1f}")
