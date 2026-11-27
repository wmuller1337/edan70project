"""Stage 3 backfill: add climate-denial examples from the CARDS dataset.

The GDELT-derived denial class is typically small (denial is a minority in
mainstream news) and English-heavy. This module strengthens it by pulling the
contrarian paragraphs from the CARDS corpus (Coan, Boussalis, Cook & Nanko,
2021, *Nature Scientific Reports*), and optionally machine-translating them into
the project's target languages so the non-English denial cells are not empty.

The CARDS training data is distributed by the authors via Google Drive
(see https://github.com/traviscoan/cards). We download it on demand and cache
it locally; it is not redistributed here.

CSV schema (``data/training/training.csv``): columns ``text, claim`` where
``claim == "0_0"`` means *no contrarian claim* and any other code is one of the
contrarian super/sub-claims. For our binary task: denial = claim != "0_0".
"""

import glob
import os
import zipfile

import pandas as pd

# CARDS training data (Coan et al. 2021), hosted on the authors' Google Drive
CARDS_GDRIVE_ID = "14exmlYCT3-K2byYHFFrShAIYiemJQroi"
CACHE_DIR = "./cards_data"
NO_CLAIM_CODE = "0_0"


def download_cards_data(cache_dir: str = CACHE_DIR) -> str:
    """Download + unzip the CARDS data (cached). Returns path to training.csv."""
    training_csv = os.path.join(cache_dir, "data", "training", "training.csv")
    if os.path.exists(training_csv):
        return training_csv

    import gdown

    os.makedirs(cache_dir, exist_ok=True)
    zip_path = os.path.join(cache_dir, "cards_data.zip")
    if not os.path.exists(zip_path):
        gdown.download(
            f"https://drive.google.com/uc?id={CARDS_GDRIVE_ID}",
            zip_path,
            quiet=False,
        )
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(cache_dir)

    if not os.path.exists(training_csv):
        # Fall back to searching, in case the archive layout differs
        found = glob.glob(
            os.path.join(cache_dir, "**", "training.csv"), recursive=True
        )
        if not found:
            raise FileNotFoundError(
                f"training.csv not found under {cache_dir} after extraction"
            )
        training_csv = found[0]
    return training_csv


def _rows_to_df(texts, language: str, label: int, source: str,
                claim_ids=None) -> pd.DataFrame:
    """Build a pipeline-compatible dataframe for one (language, label) group.

    claim_ids: optional list (aligned with texts) identifying the ORIGINAL claim
    each row derives from. Translations of the same English claim share a
    claim_id so a grouped train/test split can keep them on the same side.
    If omitted, each row gets a unique claim_id.
    """
    n = len(texts)
    if claim_ids is None:
        claim_ids = [f"{source}/{language}/{label}/{i}" for i in range(n)]
    return pd.DataFrame(
        {
            "FullText": [str(t) for t in texts],
            "Language": language,
            "DenialLabel": int(label),
            "DenialScore": float(label),
            "claim_id": [str(c) for c in claim_ids],
            # Columns the embedding step expects:
            "SourceCommonName": source,
            "DocumentIdentifier": [
                f"cards://{source}/{language}/{label}/{i}" for i in range(n)
            ],
            "DATE": 0,
            "Tone": "0,0,0,0,0,0",
        }
    )


def load_cards_examples(
    cache_dir: str = CACHE_DIR,
    split: str = "training",
    denial: bool = True,
    max_examples: int = None,
    random_state: int = 42,
):
    """Load English CARDS examples of ONE class.

    Args:
        denial: if True, return contrarian paragraphs (claim != '0_0');
            if False, return non-contrarian paragraphs (claim == '0_0').
        max_examples: cap (random sample) if set.

    Returns:
        List of English texts for the requested class.
    """
    csv_path = os.path.join(cache_dir, "data", "training", f"{split}.csv")
    if not os.path.exists(csv_path):
        download_cards_data(cache_dir)
        csv_path = os.path.join(cache_dir, "data", "training", f"{split}.csv")

    df = pd.read_csv(csv_path)
    is_denial = df["claim"].astype(str) != NO_CLAIM_CODE
    sub = df[is_denial] if denial else df[~is_denial]
    if max_examples is not None and len(sub) > max_examples:
        sub = sub.sample(max_examples, random_state=random_state)
    return sub["text"].astype(str).tolist()


def build_cards_dataset(
    target_languages=None,
    per_class_english: int = 500,
    per_class_translated: int = 100,
    translate_backfill: bool = True,
    cache_dir: str = CACHE_DIR,
    random_state: int = 42,
) -> pd.DataFrame:
    """Build a BALANCED, multilingual CARDS dataset (both classes).

    Both the denial and not-denial classes are drawn from the SAME source
    (CARDS) and translated into the SAME target languages, so the classifier
    cannot separate them by source/style - it must learn denial *content*.
    This removes the source/style confound that arises when denial comes only
    from CARDS and not-denial only from GDELT news.

    Args:
        target_languages: GDELT language codes to create coverage for
            (e.g. ['spa', 'deu']); 'eng' is always included natively.
        per_class_english: English examples PER CLASS (denial / not-denial).
        per_class_translated: examples PER CLASS translated into EACH target
            language.
        translate_backfill: if False, only the English CARDS data is returned.

    Returns:
        Balanced, pipeline-ready dataframe (FullText / Language / DenialLabel ...).
    """
    denial_en = load_cards_examples(
        cache_dir=cache_dir, denial=True,
        max_examples=per_class_english, random_state=random_state,
    )
    notden_en = load_cards_examples(
        cache_dir=cache_dir, denial=False,
        max_examples=per_class_english, random_state=random_state,
    )
    # claim_id namespaces: 'd{i}' for denial, 'n{i}' for not-denial.
    denial_ids = [f"d{i}" for i in range(len(denial_en))]
    notden_ids = [f"n{i}" for i in range(len(notden_en))]
    frames = [
        _rows_to_df(denial_en, "eng", 1, "CARDS", claim_ids=denial_ids),
        _rows_to_df(notden_en, "eng", 0, "CARDS", claim_ids=notden_ids),
    ]

    if translate_backfill and target_languages:
        # Imported here so this module stays importable without torch
        from labelling import translate as nllb_translate, gdelt_to_nllb, TARGET_LANG

        denial_pool = denial_en[:per_class_translated]
        notden_pool = notden_en[:per_class_translated]
        # translations reuse the SAME claim_id as the English source claim
        denial_pool_ids = denial_ids[:per_class_translated]
        notden_pool_ids = notden_ids[:per_class_translated]
        for lang in target_languages:
            if lang == "eng":
                continue
            tgt_code = gdelt_to_nllb(lang)
            if tgt_code is None:
                print(f"[cards] no NLLB code for '{lang}', skipping")
                continue
            print(f"[cards] translating {len(denial_pool)} denial + "
                  f"{len(notden_pool)} not-denial -> '{lang}' ({tgt_code})")
            if denial_pool:
                den_tr = nllb_translate(denial_pool, TARGET_LANG, tgt_code)
                frames.append(_rows_to_df(den_tr, lang, 1, "CARDS-translated",
                                          claim_ids=denial_pool_ids))
            if notden_pool:
                not_tr = nllb_translate(notden_pool, TARGET_LANG, tgt_code)
                frames.append(_rows_to_df(not_tr, lang, 0, "CARDS-translated",
                                          claim_ids=notden_pool_ids))

    result = pd.concat(frames, ignore_index=True).reset_index(drop=True)
    n_den = int((result["DenialLabel"] == 1).sum())
    print(f"[cards] balanced dataset: {len(result)} rows "
          f"({n_den} denial / {len(result) - n_den} not-denial) across "
          f"{result['Language'].nunique()} language(s)")
    return result


def combine_balanced(positive_df, negative_dfs, random_state: int = 42):
    """Combine a positive (denial) frame with one or more negative frames,
    downsampling the negative pool to match the positive count, then shuffle.

    Use this to mix negatives from different sources (e.g. CARDS not-denial +
    real GDELT climate news) while keeping the two classes balanced.

    Args:
        positive_df: denial rows (DenialLabel == 1).
        negative_dfs: list of not-denial frames (DenialLabel == 0).
        random_state: sampling/shuffle seed.

    Returns:
        Balanced, shuffled dataframe.
    """
    negatives = pd.concat(list(negative_dfs), ignore_index=True)
    n = min(len(positive_df), len(negatives))
    pos = (positive_df.sample(n, random_state=random_state)
           if len(positive_df) > n else positive_df)
    neg = (negatives.sample(n, random_state=random_state)
           if len(negatives) > n else negatives)
    combined = pd.concat([pos, neg], ignore_index=True)
    # shuffle so classes/sources are interleaved
    combined = combined.sample(frac=1, random_state=random_state).reset_index(drop=True)
    n_den = int((combined["DenialLabel"] == 1).sum())
    print(f"[combine] balanced set: {len(combined)} rows "
          f"({n_den} denial / {len(combined) - n_den} not-denial); "
          f"negatives from {neg['SourceCommonName'].nunique()} source name(s)")
    return combined
