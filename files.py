import urllib.request
import zipfile
import io
import glob
import os
import re
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from newspaper import Article, ArticleException
import requests
import trafilatura

from labelling import add_denial_labels
from cards_backfill import backfill_denial
from evaluation import (
    print_language_label_crosstab,
    evaluate,
    auto_cross_lingual,
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


def scrape_article_text(url):
    """Extract the main body text of an article.

    Tries newspaper3k first; if that fails or returns too little text,
    falls back to trafilatura. Returns an empty string if both fail.
    """
    # Method 1: newspaper3k
    try:
        article = Article(url, request_timeout=5)
        article.download()
        article.parse()
        # article.text contains ONLY the main body of the article
        if article.text and len(article.text.strip()) > 50:
            return article.text.strip()
    except Exception:
        pass

    # Method 2: fallback to trafilatura (now actually reachable)
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            extracted = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=False,
                fast=True,
            )
            if extracted and len(extracted.strip()) > 50:
                return extracted.strip()
    except Exception:
        pass

    # If both methods fail, return an empty string
    return ""

def add_full_text_to_df(df, limit=20, max_workers=12):
    print(f"Scraping full text for up to {limit} articles "
          f"({max_workers} parallel workers)...")

    # Deduplicate on the article URL so we don't scrape/embed the same
    # article multiple times (and don't waste the limit on duplicates).
    deduped = df.drop_duplicates(subset="DocumentIdentifier")

    # Working on a copy of the sample
    sample_df = deduped.head(limit).copy()

    # Scrape concurrently - scraping is network-bound, so a thread pool gives
    # a large wall-clock speedup. executor.map preserves input order.
    urls = sample_df["DocumentIdentifier"].tolist()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        texts = list(executor.map(scrape_article_text, urls))
    sample_df["FullText"] = texts

    # Keep only articles where scraping actually produced text
    successful = sample_df[sample_df["FullText"].str.len() > 0].reset_index(drop=True)
    print(f"Successfully scraped {len(successful)} out of {len(sample_df)} articles.")

    # Return the filtered successful frame so empty scrapes don't flow downstream
    return successful

def parse_source_language(translation_info):
    """Return the source-language code for a GKG record.

    GDELT's translingual stream stores the original language in the
    TranslationInfo field as "srclc:<lang>;eng:<engine>;...". English-origin
    records (from the English stream) have an empty field, which we map to
    "eng".
    """
    text = str(translation_info) if translation_info is not None else ""
    match = re.search(r"srclc:(\w+)", text)
    return match.group(1) if match else "eng"


def read_df():

    # Build timestamps for every 2 hours across the day, at :00 and :30
    timestamps = [
        f"20240501{hour:02d}{minute:02d}00"
        for hour in range(0, 24, 2)
        for minute in (0, 30)
    ]

    # Download BOTH GKG streams per timestamp:
    #   <ts>.gkg.csv.zip              -> English-only stream
    #   <ts>.translation.gkg.csv.zip  -> translingual stream (65+ languages,
    #                                    machine-translated), which is what
    #                                    makes the corpus actually multilingual.
    base = "http://data.gdeltproject.org/gdeltv2"
    urls = []
    for ts in timestamps:
        urls.append(f"{base}/{ts}.gkg.csv.zip")
        urls.append(f"{base}/{ts}.translation.gkg.csv.zip")

    # Define the primary column names according to the GDELT GKG v2 schema
    COLUMN_NAMES = [
        "GKGRECORDID",
        "DATE",
        "SourceCollectionIdentifier",
        "SourceCommonName",
        "DocumentIdentifier",
        "Counts",
        "V2Counts",
        "Themes",
        "V2Themes",
        "Locations",
        "V2Locations",
        "Persons",
        "V2Persons",
        "Organizations",
        "V2Organizations",
        "Tone",
        "EnhancedDates",
        "GCAM",
        "SharingImage",
        "RelatedImages",
        "SocialImageEmbeds",
        "SocialVideoEmbeds",
        "Quotations",
        "AllNames",
        "Amounts",
        "TranslationInfo",
        "ExtrasXML",
    ]
    
    
    os.makedirs("./gkg_raw_data", exist_ok=True)

    for idx, url in enumerate(urls, 1):
        # The extracted CSV has the same name as the zip minus ".zip".
        expected = os.path.join("./gkg_raw_data", os.path.basename(url)[:-4])
        if os.path.exists(expected):
            # Already downloaded on a previous run; skip to save time/bandwidth.
            continue
        try:
            response = urllib.request.urlopen(url)
            with zipfile.ZipFile(io.BytesIO(response.read())) as zip_file:
                zip_file.extractall("./gkg_raw_data")
        except Exception as e:
            print(f"[{idx}/{len(urls)}] Failed to fetch {url}: {e}")
            
    
    files = glob.glob("./gkg_raw_data/*.gkg.csv")

    df = []
    
    for file_path in files:
        df_single = pd.read_csv(
            file_path,
            sep="\t",
            names=COLUMN_NAMES,
            header=None,
            on_bad_lines="skip",
            low_memory=False,
            )
        df.append(df_single)
    
    df = pd.concat(df, ignore_index=True)

    # Derive a source-language field from the TranslationInfo column.
    # English-origin records have an empty TranslationInfo; translingual
    # records carry "srclc:<lang>;..." indicating the original language.
    df["Language"] = df["TranslationInfo"].apply(parse_source_language)

    print(
        f"Successfully loaded total dataset of {len(df)} raw records."
    )
    lang_counts = df["Language"].value_counts()
    print("Language distribution (top 10):")
    print(lang_counts.head(10).to_string())

    return df


def cretate_climate_df(df):

    CLIMATE_THEMES = r"ENV_CLIMATECHANGE|ENV_GLOBALWARMING|ENV_GREENENERGY|ENV_CARBON|ENV_DROUGHT|ENV_FLOOD"

    all_climate_articles = []

    # creates mask with chosen climate indicators    
    climate_mask = (
            df["V2Themes"].astype(str).str.contains(CLIMATE_THEMES, regex=True)
    )
    filtered = df[climate_mask]
        
    if not filtered.empty:
        all_climate_articles.append(filtered)
        
    if all_climate_articles:
        climate_df = pd.concat(all_climate_articles, ignore_index=True)

        print(f"Dataset Shape: {climate_df.shape[0]} rows, {climate_df.shape[1]} columns\n")

    # Select key columns to inspect
    sample_columns = [
            "DATE",
            "SourceCommonName",
            "DocumentIdentifier",
            "V2Themes",
            "Tone",
            # Tone String = Average Tone, Positive Score, Negative Score, Polarity, Activity Density, Self-Group Density
    ]
    print("--- SAMPLE RECORD PREVIEW ---")
    for idx, row in climate_df[sample_columns].head(2).iterrows():
            print(f"\n[Record {idx + 1}]")
            print(f"Date:       {row['DATE']}")
            print(f"Source:     {row['SourceCommonName']}")
            print(f"URL:        {row['DocumentIdentifier']}")
            print(f"Tone:       {row['Tone']}")
            print(f"Themes:     {str(row['V2Themes'])[:150]}...")  # Truncated for readability
            
    return climate_df     
            
            
def create_embedding(df, rebuild=True):
    # setting up embeddings for articles
    #
    # rebuild=True  -> delete any existing collection and embed from scratch.
    # rebuild=False -> if a non-empty collection already exists, reuse it
    #                  (skips re-embedding, which is the slow/expensive step).

    chroma_client = chromadb.PersistentClient(path="./chroma_db")

    # Use GPU for embedding if available (falls back to CPU cleanly)
    try:
        import torch
        embed_device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        embed_device = "cpu"

    # setup embedding function
    sentence_transformer_ef = (
        embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="microsoft/harrier-oss-v1-270m",
            device=embed_device,
        )
    )

    collection_name = "climate_fulltext"

    # Reuse path: if not rebuilding and a populated collection exists, return it.
    if not rebuild:
        existing = {c.name for c in chroma_client.list_collections()}
        if collection_name in existing:
            collection = chroma_client.get_collection(
                name=collection_name,
                embedding_function=sentence_transformer_ef,
            )
            if collection.count() > 0:
                print(
                    f"Reusing existing embedding collection "
                    f"'{collection_name}' with {collection.count()} records "
                    f"(rebuild=False). Pass rebuild=True to re-embed."
                )
                return collection
            print(
                f"Collection '{collection_name}' exists but is empty; "
                f"rebuilding."
            )

    # Rebuild path: drop any existing collection and re-create it.
    try:
        chroma_client.delete_collection(name=collection_name)
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=collection_name,
        embedding_function=sentence_transformer_ef,
        metadata={"hnsw:space": "cosine"},
    )
    
    documents = []
    metadatas = []
    ids = []

    for idx, row in df.iterrows():
        # Embed the actual scraped article text (first 2000 chars)
        doc_text = str(row["FullText"])[:2000]
        avg_tone = float(str(row["Tone"]).split(",")[0])

        documents.append(doc_text)
        metadatas.append({
            "source": str(row["SourceCommonName"]),
            "date": int(row["DATE"]),
            "url": str(row["DocumentIdentifier"]),
            "tone": avg_tone,
            "language": str(row["Language"]),
            # Real denial label from Stage 2 (CARDS + translation)
            "label": int(row["DenialLabel"]),
            "denial_score": float(row["DenialScore"]),
        })
        ids.append(f"doc_{idx}")

    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print(
        f"Successfully stored full-text embeddings! Total records in Chroma: {collection.count()}"
    )

    return collection


def classification(X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    C: float = 10.0,
    target_names: list = None,):
    
    X = np.array(X)
    y = np.array(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    model = LogisticRegression(
        max_iter=1000,
        C=C,
        class_weight="balanced",
        random_state=random_state,
        solver="lbfgs",
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    
    print("==================================================")
    print("         LOGISTIC REGRESSION EVALUATION          ")
    print("==================================================")
    print(
        f"5-Fold Cross-Validation Accuracy: {cv_scores.mean():.4f} (± {cv_scores.std():.4f})\n"
    )

    names = (
        target_names
        if target_names
        else [str(c) for c in np.unique(y)]
    )
    print(classification_report(y_test, y_pred, target_names=names))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    print("==================================================")

    results = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "cv_scores": cv_scores,
        "confusion_matrix": cm,
    }

    return model, results



if __name__ == "__main__":
    # ----- pipeline configuration -----
    USE_CARDS_BACKFILL = True   # add CARDS denial examples to the denial class
    REBUILD_EMBEDDINGS = True   # False -> reuse existing Chroma collection
    CARDS_MAX_ENGLISH = 500     # cap on English CARDS denial examples
    CARDS_PER_LANGUAGE = 100    # translated denial examples per target language
    USE_CACHE = True            # cache scraped + labelled data to skip recompute
    SCRAPE_WORKERS = 12         # parallel scraping threads
    CACHE_DIR = "./cache"
    SCRAPED_CACHE = os.path.join(CACHE_DIR, "scraped.pkl")
    LABELLED_CACHE = os.path.join(CACHE_DIR, "labelled.pkl")
    os.makedirs(CACHE_DIR, exist_ok=True)

    # ---- data -> scrape -> label -> backfill, with staged caching ----
    if USE_CACHE and os.path.exists(LABELLED_CACHE):
        print(f"[cache] loading labelled data from {LABELLED_CACHE}")
        climate_df_labelled = pd.read_pickle(LABELLED_CACHE)
    else:
        if USE_CACHE and os.path.exists(SCRAPED_CACHE):
            print(f"[cache] loading scraped data from {SCRAPED_CACHE}")
            climate_df_scraped = pd.read_pickle(SCRAPED_CACHE)
        else:
            print("read df")
            df = read_df()
            print("filter on climate df")
            climate_df = cretate_climate_df(df)
            print("scrape for full articles")
            climate_df_scraped = add_full_text_to_df(
                climate_df, limit=100, max_workers=SCRAPE_WORKERS
            )
            if USE_CACHE:
                climate_df_scraped.to_pickle(SCRAPED_CACHE)
                print(f"[cache] saved scraped data -> {SCRAPED_CACHE}")

        # Stage 2: binary denial labels via CARDS (+ NLLB translation for
        # non-English). Drop rows we could not label (unmapped language).
        print("label denial (CARDS + translation)")
        climate_df_labelled = add_denial_labels(climate_df_scraped)
        climate_df_labelled = climate_df_labelled.dropna(
            subset=["DenialLabel"]
        ).reset_index(drop=True)
        climate_df_labelled["DenialLabel"] = climate_df_labelled[
            "DenialLabel"
        ].astype(int)

        # Stage 3 backfill: strengthen the (minority) denial class with CARDS
        # examples, translated into the languages present in our GDELT data.
        if USE_CARDS_BACKFILL:
            print("backfill denial class from CARDS")
            target_langs = sorted(set(climate_df_labelled["Language"]))
            cards_df = backfill_denial(
                target_languages=target_langs,
                max_english=CARDS_MAX_ENGLISH,
                per_language=CARDS_PER_LANGUAGE,
            )
            climate_df_labelled = pd.concat(
                [climate_df_labelled, cards_df], ignore_index=True
            )

        if USE_CACHE:
            climate_df_labelled.to_pickle(LABELLED_CACHE)
            print(f"[cache] saved labelled data -> {LABELLED_CACHE}")

    embs = create_embedding(climate_df_labelled, rebuild=REBUILD_EMBEDDINGS)

    data = embs.get(include=["embeddings", "metadatas"])

    # X is the multilingual embeddings of the original-language text
    X_embeddings = np.array(data["embeddings"])

    # y is now the REAL denial label (0/1) from Stage 2, not the tone proxy
    metadatas = data["metadatas"]
    y_labels = np.array([int(meta["label"]) for meta in metadatas])
    languages = [str(meta["language"]) for meta in metadatas]

    # Stage 3: inspect the language x label distribution (flag sparse cells)
    print_language_label_crosstab(languages, y_labels)

    # Stage 6: overall + per-language macro-F1 evaluation
    trained_model = evaluate(
        X_embeddings,
        y_labels,
        languages,
        target_names=["Not Denial", "Denial"],
    )

    # Stage 6: cross-lingual transfer (train on some languages, test held-out one)
    auto_cross_lingual(X_embeddings, y_labels, languages)



