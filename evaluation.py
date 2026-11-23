"""Stage 3 + Stage 6: dataset diagnostics and multilingual evaluation.

Stage 3  -> ``print_language_label_crosstab``: inspect the language x label
            distribution and flag sparse cells (typically non-English denial).
Stage 6  -> ``evaluate``: overall macro-F1 + confusion matrix + per-language
            macro-F1 on a held-out split.
         -> ``cross_lingual_transfer`` / ``auto_cross_lingual``: train on some
            languages and test on a *held-out* language to measure genuine
            cross-lingual transfer (the headline multilingual result).

Macro-F1 is used throughout because it weights each class equally and therefore
reflects performance on the minority (denial) class rather than being dominated
by the majority.
"""

from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def _make_lr(C: float = 10.0, random_state: int = 42) -> LogisticRegression:
    """Return an unfitted logistic-regression estimator (balanced classes)."""
    return LogisticRegression(
        max_iter=1000,
        C=C,
        class_weight="balanced",
        random_state=random_state,
        solver="lbfgs",
    )


# ---------------------------------------------------------------------------
# Stage 3: dataset diagnostic
# ---------------------------------------------------------------------------
def print_language_label_crosstab(languages, labels, min_denial_per_cell: int = 10):
    """Print a language x label count table and flag sparse denial cells.

    A sparse "non-English denial" cell is the classic failure mode of a
    multilingual denial dataset; if flagged, backfill by machine-translating
    CARDS denial examples into that language.
    """
    languages = np.asarray(languages)
    labels = np.asarray(labels).astype(int)

    print("\n=== Stage 3: language x label cross-tab ===")
    print(f"{'lang':8s} {'not-denial':>11s} {'denial':>8s} {'total':>8s}")
    print("-" * 38)
    sparse = []
    for lang in sorted(set(languages)):
        mask = languages == lang
        n_not = int(np.sum(labels[mask] == 0))
        n_den = int(np.sum(labels[mask] == 1))
        print(f"{lang:8s} {n_not:11d} {n_den:8d} {n_not + n_den:8d}")
        if n_den < min_denial_per_cell:
            sparse.append((lang, n_den))

    total_den = int(np.sum(labels == 1))
    total = len(labels)
    print("-" * 38)
    print(f"{'TOTAL':8s} {total - total_den:11d} {total_den:8d} {total:8d}")
    if sparse:
        print(f"\n[warn] languages with < {min_denial_per_cell} denial examples: "
              f"{sparse}")
        print("       consider backfilling via translated CARDS positives.")


# ---------------------------------------------------------------------------
# Stage 6: overall + per-language evaluation
# ---------------------------------------------------------------------------
def evaluate(
    X,
    y,
    languages,
    test_size: float = 0.2,
    random_state: int = 42,
    C: float = 10.0,
    target_names=("Not Denial", "Denial"),
):
    """Train LR and report overall macro-F1, confusion matrix, per-language F1.

    Returns the fitted model.
    """
    X = np.asarray(X)
    y = np.asarray(y).astype(int)
    languages = np.asarray(languages)

    # Split on indices so we can recover per-row languages for the test set
    indices = np.arange(len(y))
    i_train, i_test = train_test_split(
        indices, test_size=test_size, random_state=random_state, stratify=y
    )

    model = _make_lr(C=C, random_state=random_state)
    model.fit(X[i_train], y[i_train])

    y_true = y[i_test]
    y_pred = model.predict(X[i_test])

    print("\n=== Stage 6: overall evaluation ===")
    print(classification_report(
        y_true, y_pred, target_names=list(target_names), zero_division=0
    ))
    overall_macro = f1_score(y_true, y_pred, average="macro",
                             labels=[0, 1], zero_division=0)
    print(f"Overall macro-F1: {overall_macro:.4f}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

    # Cross-validated macro-F1 for robustness; guard against tiny classes
    min_class = min(Counter(y).values())
    n_folds = min(5, min_class)
    if n_folds >= 2:
        cv_scores = cross_val_score(
            _make_lr(C=C, random_state=random_state),
            X, y, cv=n_folds, scoring="f1_macro",
        )
        print(f"{n_folds}-fold CV macro-F1: {cv_scores.mean():.4f} "
              f"(+/- {cv_scores.std():.4f})")
    else:
        print("[warn] a class has < 2 samples; skipping cross-validation")

    # Per-language macro-F1 on the test split
    print("\n--- per-language macro-F1 (test split) ---")
    test_langs = languages[i_test]
    for lang in sorted(set(test_langs)):
        mask = test_langs == lang
        lang_macro = f1_score(
            y_true[mask], y_pred[mask], average="macro",
            labels=[0, 1], zero_division=0,
        )
        print(f"  {lang:8s} n={int(mask.sum()):4d}  macro-F1={lang_macro:.4f}")

    return model


# ---------------------------------------------------------------------------
# Stage 6: cross-lingual transfer
# ---------------------------------------------------------------------------
def cross_lingual_transfer(
    X, y, languages, train_langs, test_lang, C: float = 10.0, random_state: int = 42
):
    """Train on ``train_langs`` and test on held-out ``test_lang``.

    Returns the transfer macro-F1, or None if the experiment is not runnable.
    """
    X = np.asarray(X)
    y = np.asarray(y).astype(int)
    languages = np.asarray(languages)

    train_mask = np.isin(languages, list(train_langs))
    test_mask = languages == test_lang

    if train_mask.sum() == 0 or test_mask.sum() == 0:
        print(f"[transfer] skip: train={int(train_mask.sum())} "
              f"test={int(test_mask.sum())} rows")
        return None
    if len(set(y[train_mask])) < 2:
        print("[transfer] skip: training languages contain only one class")
        return None

    model = _make_lr(C=C, random_state=random_state)
    model.fit(X[train_mask], y[train_mask])
    y_pred = model.predict(X[test_mask])
    macro = f1_score(y[test_mask], y_pred, average="macro",
                     labels=[0, 1], zero_division=0)

    print("\n=== Stage 6: cross-lingual transfer ===")
    print(f"train on {sorted(str(l) for l in train_langs)} "
          f"(n={int(train_mask.sum())}) "
          f"-> test on '{test_lang}' (n={int(test_mask.sum())})")
    print(f"transfer macro-F1: {macro:.4f}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y[test_mask], y_pred, labels=[0, 1]))
    return macro


def auto_cross_lingual(X, y, languages, min_rows: int = 10):
    """Automatically pick a held-out non-English language and run transfer.

    Chooses the non-English language that has both classes and the most rows,
    trains on all other languages, and tests on it.
    """
    y = np.asarray(y).astype(int)
    languages = np.asarray(languages)

    candidates = []
    for lang in set(languages):
        if lang == "eng":
            continue
        mask = languages == lang
        if mask.sum() >= min_rows and len(set(y[mask])) == 2:
            candidates.append((lang, int(mask.sum())))

    if not candidates:
        print("\n[transfer] no non-English language with both classes and "
              f">= {min_rows} rows; skipping cross-lingual transfer test.")
        return None

    held_out = str(max(candidates, key=lambda t: t[1])[0])
    train_langs = [str(l) for l in set(languages) if str(l) != held_out]
    return cross_lingual_transfer(X, y, languages, train_langs, held_out)
