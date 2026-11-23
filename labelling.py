"""Stage 2: binary climate-denial labelling (CARDS + translation).

This module produces binary denial labels for scraped articles WITHOUT needing
any external LLM API. It uses a knowledge-distillation setup:

  * Teacher (labels): the pre-trained English denial classifier
    ``crarojasca/BinaryAugmentedCARDS`` (RoBERTa). English articles are fed
    straight in; non-English articles are first machine-translated to English
    with ``facebook/nllb-200-distilled-600M`` and then classified.
  * The labels produced here are later attached to the *multilingual* embeddings
    of the ORIGINAL-language text, so the downstream logistic-regression
    "student" can classify any language at inference time without translation.

Models are loaded lazily (on first use) and cached, so importing this module is
cheap and unit-testable without downloading multi-GB weights.
"""

from functools import lru_cache
from typing import List

# ---------------------------------------------------------------------------
# Language-code mapping: GDELT srclc (ISO-639-3-ish) -> NLLB FLORES-200 code
# ---------------------------------------------------------------------------
# GDELT's TranslationInfo uses codes like "spa", "deu", "zho"; NLLB expects
# FLORES-200 codes like "spa_Latn", "deu_Latn", "zho_Hans". This covers the
# languages commonly seen in the GDELT climate stream; unknown codes fall back
# to None (handled by the caller: such articles are skipped for translation).
GDELT_TO_NLLB = {
    "eng": "eng_Latn",
    "spa": "spa_Latn",
    "fra": "fra_Latn",
    "deu": "deu_Latn",
    "por": "por_Latn",
    "ita": "ita_Latn",
    "nld": "nld_Latn",
    "swe": "swe_Latn",
    "dan": "dan_Latn",
    "nor": "nob_Latn",
    "fin": "fin_Latn",
    "pol": "pol_Latn",
    "ron": "ron_Latn",
    "hrv": "hrv_Latn",
    "srp": "srp_Cyrl",
    "ces": "ces_Latn",
    "slk": "slk_Latn",
    "ell": "ell_Grek",
    "bul": "bul_Cyrl",
    "rus": "rus_Cyrl",
    "ukr": "ukr_Cyrl",
    "tur": "tur_Latn",
    "ara": "arb_Arab",
    "heb": "heb_Hebr",
    "fas": "pes_Arab",
    "hin": "hin_Deva",
    "jpn": "jpn_Jpan",
    "kor": "kor_Hang",
    "zho": "zho_Hans",
    "tha": "tha_Thai",
    "vie": "vie_Latn",
    "ind": "ind_Latn",
    "msa": "zsm_Latn",
}

TARGET_LANG = "eng_Latn"

# Model identifiers
CARDS_MODEL = "crarojasca/BinaryAugmentedCARDS"
NLLB_MODEL = "facebook/nllb-200-distilled-600M"

# Label constants (match the CARDS model's id2label: 0=Not Contrarian, 1=Contrarian)
LABEL_NOT_DENIAL = 0
LABEL_DENIAL = 1


def gdelt_to_nllb(srclc: str) -> str:
    """Map a GDELT source-language code to an NLLB FLORES-200 code.

    Returns None if the language is not in the mapping (caller should skip
    translation for such rows rather than guess).
    """
    if not srclc:
        return None
    return GDELT_TO_NLLB.get(srclc.strip().lower())


# ---------------------------------------------------------------------------
# Lazy model loaders (cached) - kept out of import time so tests stay cheap
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _load_cards():
    """Load and cache the CARDS tokenizer + model."""
    import torch  # noqa: F401  (ensures torch is available / sets backend)
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
    )

    tokenizer = AutoTokenizer.from_pretrained(CARDS_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(CARDS_MODEL)
    model.eval()
    return tokenizer, model


@lru_cache(maxsize=1)
def _load_nllb():
    """Load and cache the NLLB tokenizer + model."""
    import torch  # noqa: F401
    from transformers import (
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
    )

    tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL)
    model.eval()
    return tokenizer, model


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------
def translate_to_english(
    texts: List[str],
    src_nllb_code: str,
    max_input_chars: int = 1500,
    batch_size: int = 8,
) -> List[str]:
    """Translate a list of same-language texts into English using NLLB.

    Args:
        texts: source-language strings (all in the SAME source language).
        src_nllb_code: the NLLB FLORES-200 code of the source language.
        max_input_chars: truncate each input to this many characters before
            translating (CARDS only reads ~512 tokens downstream, so we don't
            need the full article and this keeps translation fast).
        batch_size: number of texts per model forward pass.

    Returns:
        List of English translations, aligned with ``texts``.
    """
    import torch

    tokenizer, model = _load_nllb()
    tokenizer.src_lang = src_nllb_code
    bos_token_id = tokenizer.convert_tokens_to_ids(TARGET_LANG)

    outputs: List[str] = []
    for start in range(0, len(texts), batch_size):
        batch = [t[:max_input_chars] for t in texts[start:start + batch_size]]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                forced_bos_token_id=bos_token_id,
                max_length=512,
            )
        outputs.extend(
            tokenizer.batch_decode(generated, skip_special_tokens=True)
        )
    return outputs


# ---------------------------------------------------------------------------
# CARDS denial classification
# ---------------------------------------------------------------------------
def classify_denial(
    texts: List[str],
    batch_size: int = 16,
):
    """Classify English texts as denial (1) / not-denial (0) using CARDS.

    Returns:
        (labels, scores) where labels is a list of 0/1 ints and scores is the
        list of denial (class-1) probabilities.
    """
    import torch

    tokenizer, model = _load_cards()

    labels: List[int] = []
    scores: List[float] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        with torch.no_grad():
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)
        denial_probs = probs[:, LABEL_DENIAL]
        batch_labels = (denial_probs >= 0.5).long().tolist()
        labels.extend(batch_labels)
        scores.extend(denial_probs.tolist())
    return labels, scores


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def add_denial_labels(
    df,
    text_col: str = "FullText",
    lang_col: str = "Language",
    label_col: str = "DenialLabel",
    score_col: str = "DenialScore",
):
    """Add binary denial labels to a dataframe of scraped articles.

    Routing:
        * English rows (Language == 'eng')  -> CARDS directly.
        * Non-English rows with a known NLLB code -> translate to English, then
          CARDS.
        * Rows whose language has no NLLB mapping are labelled as NaN (caller
          can drop them); we do not guess.

    Adds two columns in place and returns the dataframe:
        label_col: 0 (not-denial) / 1 (denial) / NaN (unlabellable)
        score_col: denial probability (float) / NaN
    """
    import numpy as np

    # Initialise output columns
    df = df.copy()
    df[label_col] = np.nan
    df[score_col] = np.nan

    # Process language-by-language so each NLLB batch shares one source language
    for lang, group in df.groupby(lang_col):
        idx = group.index
        texts = group[text_col].astype(str).tolist()

        if lang == "eng":
            english_texts = texts
        else:
            nllb_code = gdelt_to_nllb(lang)
            if nllb_code is None:
                # Unknown language: leave as NaN (unlabellable)
                print(f"[labelling] skipping {len(texts)} rows in "
                      f"unmapped language '{lang}'")
                continue
            print(f"[labelling] translating {len(texts)} '{lang}' "
                  f"articles -> English ({nllb_code})")
            english_texts = translate_to_english(texts, nllb_code)

        labels, scores = classify_denial(english_texts)
        df.loc[idx, label_col] = labels
        df.loc[idx, score_col] = scores

    n_labelled = int(df[label_col].notna().sum())
    n_denial = int((df[label_col] == LABEL_DENIAL).sum())
    print(f"[labelling] labelled {n_labelled}/{len(df)} articles; "
          f"{n_denial} denial / {n_labelled - n_denial} not-denial")
    return df
