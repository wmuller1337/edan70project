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


def _denial_rows_to_df(texts, language: str, source: str) -> pd.DataFrame:
    """Build a pipeline-compatible denial dataframe from a list of texts."""
    n = len(texts)
    return pd.DataFrame(
        {
            "FullText": [str(t) for t in texts],
            "Language": language,
            "DenialLabel": 1,
            "DenialScore": 1.0,
            # Columns the embedding step expects:
            "SourceCommonName": source,
            "DocumentIdentifier": [f"cards://{language}/{i}" for i in range(n)],
            "DATE": 0,
            "Tone": "0,0,0,0,0,0",
        }
    )


def load_cards_denial(
    cache_dir: str = CACHE_DIR,
    split: str = "training",
    max_examples: int = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Load English CARDS denial (contrarian) paragraphs as a denial dataframe.

    Args:
        cache_dir: where CARDS data is cached/downloaded.
        split: one of 'training', 'validation', 'test'.
        max_examples: cap the number of denial rows (random sample) if set.
        random_state: sampling seed.

    Returns:
        DataFrame with FullText / Language='eng' / DenialLabel=1 (+ metadata).
    """
    csv_path = os.path.join(cache_dir, "data", "training", f"{split}.csv")
    if not os.path.exists(csv_path):
        download_cards_data(cache_dir)
        csv_path = os.path.join(cache_dir, "data", "training", f"{split}.csv")

    df = pd.read_csv(csv_path)
    df["_denial"] = (df["claim"].astype(str) != NO_CLAIM_CODE).astype(int)
    denial = df[df["_denial"] == 1]
    if max_examples is not None and len(denial) > max_examples:
        denial = denial.sample(max_examples, random_state=random_state)

    out = _denial_rows_to_df(denial["text"].tolist(), "eng", "CARDS")
    print(f"[cards] loaded {len(out)} English denial examples from "
          f"CARDS '{split}' split")
    return out.reset_index(drop=True)


def backfill_denial(
    target_languages=None,
    max_english: int = 500,
    per_language: int = 100,
    translate_backfill: bool = True,
    cache_dir: str = CACHE_DIR,
    random_state: int = 42,
) -> pd.DataFrame:
    """Build a CARDS denial backfill set, optionally translated to other langs.

    Args:
        target_languages: iterable of GDELT language codes to create denial
            examples in (e.g. ['spa', 'deu']). 'eng' is ignored (already native).
        max_english: cap on English CARDS denial examples kept.
        per_language: how many English denial examples to translate into EACH
            target language.
        translate_backfill: if False, only the English CARDS denial is returned
            (no translation performed).
        cache_dir: CARDS cache dir.

    Returns:
        Concatenated denial dataframe (English + translated), pipeline-ready.
    """
    english = load_cards_denial(
        cache_dir=cache_dir, max_examples=max_english, random_state=random_state
    )
    frames = [english]

    if translate_backfill and target_languages:
        # Imported here so this module stays importable without torch
        from labelling import translate as nllb_translate, gdelt_to_nllb, TARGET_LANG

        pool = english["FullText"].tolist()
        for lang in target_languages:
            if lang == "eng":
                continue
            tgt_code = gdelt_to_nllb(lang)
            if tgt_code is None:
                print(f"[cards] no NLLB code for '{lang}', skipping backfill")
                continue
            source_texts = pool[:per_language]
            if not source_texts:
                continue
            print(f"[cards] translating {len(source_texts)} denial examples "
                  f"-> '{lang}' ({tgt_code})")
            translated = nllb_translate(source_texts, TARGET_LANG, tgt_code)
            frames.append(
                _denial_rows_to_df(translated, lang, "CARDS-translated")
            )

    result = pd.concat(frames, ignore_index=True)
    print(f"[cards] backfill total: {len(result)} denial examples across "
          f"{result['Language'].nunique()} language(s)")
    return result
