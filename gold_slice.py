"""Improvement #9: human gold-slice export + label-agreement check.

The automatic labels (CARDS / translation) are not ground truth. A *gold slice*
is a small sample you label BY HAND, used to measure how trustworthy the
automatic labels are.

Workflow:
    1. export_gold_slice(dataset, "gold_slice.csv")  -> writes a stratified
       sample with an empty 'human_label' column.
    2. You open the CSV and fill 'human_label' (0 = not-denial, 1 = denial).
    3. load_gold_slice("gold_slice.csv")  -> reports agreement between the
       automatic labels and your human labels (accuracy + macro-F1).
"""

import os

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def export_gold_slice(
    df,
    path,
    n: int = 150,
    text_col: str = "FullText",
    label_col: str = "DenialLabel",
    lang_col: str = "Language",
    random_state: int = 42,
    max_chars: int = 1200,
):
    """Write a stratified (language x label) sample to CSV for hand-labelling.

    Does NOT overwrite an existing file (so you never lose hand labels).
    """
    if os.path.exists(path):
        print(f"[gold] {path} already exists - not overwriting")
        return None

    # stratified sample across language x label cells
    cells = list(df.groupby([lang_col, label_col]))
    per_cell = max(1, n // max(len(cells), 1))
    parts = [g.sample(min(len(g), per_cell), random_state=random_state)
             for _, g in cells]
    sample = (pd.concat(parts)
              .sample(frac=1, random_state=random_state)
              .reset_index(drop=True))

    out = pd.DataFrame({
        "id": range(len(sample)),
        "language": sample[lang_col].values,
        "auto_label": sample[label_col].astype(int).values,
        "human_label": "",  # <-- FILL IN BY HAND: 0 (not-denial) or 1 (denial)
        "text": sample[text_col].astype(str).str.slice(0, max_chars).values,
    })
    out.to_csv(path, index=False)
    print(f"[gold] exported {len(out)} rows -> {path} "
          f"(open it and fill the 'human_label' column with 0/1)")
    return out


def load_gold_slice(path):
    """Read a filled gold-slice CSV and report auto-vs-human agreement."""
    if not os.path.exists(path):
        print(f"[gold] {path} not found")
        return None

    df = pd.read_csv(path)
    filled = df[df["human_label"].notna()
                & (df["human_label"].astype(str).str.strip() != "")]
    if filled.empty:
        print(f"[gold] {path}: no human labels filled in yet")
        return None

    human = filled["human_label"].astype(int).values
    auto = filled["auto_label"].astype(int).values
    acc = accuracy_score(human, auto)
    macro = f1_score(human, auto, average="macro", labels=[0, 1],
                     zero_division=0)

    print(f"\n=== gold-slice label agreement ({len(filled)} human-labelled) ===")
    print(f"auto-vs-human accuracy: {acc:.3f}   macro-F1: {macro:.3f}")
    print("confusion (rows=human, cols=auto):")
    print(confusion_matrix(human, auto, labels=[0, 1]))
    return {"accuracy": acc, "macro_f1": macro, "n": int(len(filled))}
