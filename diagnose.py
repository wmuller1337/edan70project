"""Diagnostic: compare the logistic-regression denial model against the CARDS
model on a small labelled battery of snippets.

Purpose: expose two known weaknesses of the embeddings + logistic-regression
detector, and contrast them with the fine-tuned CARDS classifier:

  1. Topic-over-stance: mainstream statements about denial-adjacent *topics*
     (glaciers, ice, temperature) tend to be flagged as denial.
  2. Format sensitivity: short one-sentence claims (vs. the paragraph-length
     CARDS training data) are mishandled.

For each snippet it prints the true label, the LR denial probability, and the
CARDS denial probability, plus per-model accuracy and a category breakdown.

Usage:
    python diagnose.py                       # uses ./model/denial_clf.joblib
    python diagnose.py --model path.joblib
"""

import argparse

from predict import predict_texts
from labelling import classify_denial

# (text, true_label, category)   label: 1 = denial, 0 = not-denial
BATTERY = [
    # --- clear denial, short ---
    ("Climate change is a natural cycle; the climate has always changed and "
     "humans have nothing to do with it.", 1, "denial/short"),
    ("CO2 is plant food - more carbon dioxide just makes crops grow better.",
     1, "denial/short"),
    ("Global temperatures haven't risen in decades; the warming pause proves "
     "the models are wrong.", 1, "denial/short"),
    ("Climate scientists manipulate their data to keep grant money flowing; "
     "the so-called consensus is manufactured.", 1, "denial/short"),

    # --- clear denial, paragraph ---
    ("There is no climate emergency. The modest warming we have seen is well "
     "within natural variability, driven by the sun and ocean cycles, not by "
     "human CO2. Satellite records show no dangerous trend, and predictions of "
     "catastrophe have repeatedly failed to materialise. Cheap fossil fuels "
     "lift people out of poverty, while costly climate policies do real harm.",
     1, "denial/paragraph"),

    # --- mainstream, glacier/ice topic (the confound test) ---
    ("Scientists report that Arctic glaciers are melting at an accelerating "
     "rate because of human-caused warming.", 0, "mainstream/glacier"),
    ("Rising sea levels driven by melting ice sheets threaten coastal cities "
     "around the world.", 0, "mainstream/glacier"),
    ("New measurements show record low Antarctic sea ice extent, consistent "
     "with greenhouse-gas-driven warming.", 0, "mainstream/glacier"),

    # --- mainstream, short / neutral ---
    ("Glaciers are melting.", 0, "mainstream/short"),
    ("The IPCC warns that urgent emission cuts are needed to limit warming to "
     "1.5 degrees.", 0, "mainstream/short"),

    # --- mainstream, paragraph ---
    ("A new study confirms that global average temperatures continue to climb, "
     "with the past decade the warmest on record. Researchers link the trend "
     "directly to rising concentrations of greenhouse gases from burning "
     "fossil fuels, and warn that glaciers and ice sheets are losing mass "
     "faster than previously projected, accelerating sea-level rise.",
     0, "mainstream/paragraph"),
]


def run_diagnostic(model_path="./model/denial_clf.joblib"):
    texts = [b[0] for b in BATTERY]
    truth = [b[1] for b in BATTERY]
    cats = [b[2] for b in BATTERY]

    # LR path (frozen embeddings + logistic regression)
    lr = predict_texts(texts, model_path=model_path)
    lr_pred = [r["label"] for r in lr]
    lr_prob = [r["denial_prob"] for r in lr]

    # CARDS path (fine-tuned RoBERTa) - battery is English, no translation
    cards_pred, cards_prob = classify_denial(texts)

    print("\nlabel: 1=denial 0=not-denial   prob = denial probability   "
          "[*]=wrong")
    print(f"{'category':20s} {'true':>4s} {'LR':>11s} {'CARDS':>11s}  text")
    print("-" * 100)
    for i in range(len(texts)):
        lr_mark = " " if lr_pred[i] == truth[i] else "*"
        c_mark = " " if cards_pred[i] == truth[i] else "*"
        print(f"{cats[i]:20s} {truth[i]:>4d} "
              f"{lr_prob[i]:9.2f}{lr_mark:2s} "
              f"{cards_prob[i]:9.2f}{c_mark:2s} "
              f"{texts[i][:55]}")

    def acc(pred):
        return sum(int(p == t) for p, t in zip(pred, truth)) / len(truth)

    print("-" * 100)
    print(f"LR accuracy:    {acc(lr_pred):.2f}")
    print(f"CARDS accuracy: {acc(cards_pred):.2f}")

    # Category breakdown (where does each model fail?)
    print("\nper-category accuracy:")
    print(f"{'category':20s} {'LR':>6s} {'CARDS':>6s}")
    for cat in dict.fromkeys(cats):  # preserve order, unique
        idx = [i for i, c in enumerate(cats) if c == cat]
        lr_a = sum(int(lr_pred[i] == truth[i]) for i in idx) / len(idx)
        c_a = sum(int(cards_pred[i] == truth[i]) for i in idx) / len(idx)
        print(f"{cat:20s} {lr_a:6.2f} {c_a:6.2f}")

    return {
        "texts": texts, "truth": truth, "cats": cats,
        "lr_pred": lr_pred, "lr_prob": lr_prob,
        "cards_pred": cards_pred, "cards_prob": cards_prob,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare LR vs CARDS denial models on a labelled battery."
    )
    parser.add_argument("--model", default="./model/denial_clf.joblib")
    args = parser.parse_args()
    run_diagnostic(model_path=args.model)


if __name__ == "__main__":
    main()
