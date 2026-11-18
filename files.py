import urllib.request
from concurrent.futures import ThreadPoolExecutor
import zipfile
import io
import glob
import numpy as np
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_validate, train_test_split, TunedThresholdClassifierCV
from newspaper import Article, ArticleException
import requests
import trafilatura

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


def scrape_article_text(url):
    
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            extracted = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=False,
            )
            if extracted and len(extracted.strip()) >= 200:
                return extracted.strip()
    except Exception:
        pass

    # Method 2: Fallback to Trafilatura
    try:
        article = Article(url, request_timeout=5)
        article.download()
        article.parse()
        if article.text and len(article.text.strip()) >= 200:
            return article.text.strip()
    except Exception:
        pass

    return ""


def add_full_text_to_df(df, limit=20):
    print(f"Scraping full text for the first {limit} articles...")
    
    # Working on a copy of the sample
    sample_df = df.head(limit).copy() if limit is not None else df.copy()
    
    urls = sample_df["DocumentIdentifier"].tolist()
    with ThreadPoolExecutor(max_workers=10) as executor:
        sample_df["FullText"] = list(executor.map(scrape_article_text, urls))
    
    # Filter out articles where scraping failed (empty text)
    successful = sample_df[sample_df["FullText"].str.len() > 0]
    print(f"Successfully scraped {len(successful)} out of {limit} articles.")
    
    return successful

def read_df():

    #  download for a single 15-minute chunk
    urls = [
            f"http://data.gdeltproject.org/gdeltv2/20240501{hour:02d}{minute:02d}00.gkg.csv.zip"
            for hour in range(0, 24, 2)  # Every 2 hours across the day
            for minute in (0, 30)  # At 00 and 30 minutes
        ]

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
    
    
    for idx, url in enumerate(urls, 1):
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
    
    print(
        f"Successfully loaded total dataset of {len(df)} raw records."
    )
    
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
            
            
def create_embedding(df):     
    # setting up embeddings for articles
            
    chroma_client = chromadb.PersistentClient(path="./chroma_db")

    # setup embedding function
    sentence_transformer_ef = (
        embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="microsoft/harrier-oss-v1-270m"
        )
    )

    collection_name = "climate_fulltext"
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
        })
        ids.append(f"doc_{idx}")

    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print(
        f"Successfully stored full-text embeddings! Total records in Chroma: {collection.count()}"
    )

    return collection


def build_multiclass_labels(metadatas: list, mode: str = "3class_polarity"):
    """
    mode='3class_polarity':
        0 = Crisis / Alarmist (tone <= -2.5)
        1 = Neutral / Procedural (-2.5 < tone < 1.0)
        2 = Solution / Resilience (tone >= 1.0)

    mode='4class_framing':
        0 = Humanitarian & Displacement
        1 = Policy, Law & Governance
        2 = Economic & Infrastructure
        3 = Science, Environment & Disaster Impact (default climate frame)
    """
    if mode == "3class_polarity":
        y = []
        for meta in metadatas:
            t = meta["tone"]
            if t <= -2.5:
                y.append(0)
            elif t < 1.0:
                y.append(1)
            else:
                y.append(2)
        names = ["Crisis / Alarmist", "Neutral / Procedural", "Solution / Resilience"]
        return np.array(y), names

    elif mode == "4class_framing":
        y = []
        for meta in metadatas:
            th = meta["themes"]
            if any(k in th for k in ["REFUGEES", "HUMANITARIAN", "FAMINE", "DISPLACED", "POVERTY"]):
                y.append(0)
            elif any(k in th for k in ["LEGISLATION", "GOV_", "DIPLOMACY", "UNGP_", "ELECTION"]):
                y.append(1)
            elif any(k in th for k in ["ECON_", "AGRICULTURE", "INFRASTRUCTURE", "ENERGY", "TAX_"]):
                y.append(2)
            else:
                y.append(3)
        names = ["Humanitarian", "Policy & Governance", "Economic & Energy", "Environmental Impact"]
        return np.array(y), names

def classification(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    C: float = 1.0,
    target_names: list = None,):
    
    X = np.array(X)
    y = np.array(y)
    
    names = target_names if target_names else [str(c) for c in np.unique(y)]
    print(f"Class distribution: {dict(zip(names, np.bincount(y)))}\n")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    model = LogisticRegression(
        C=C,
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )
    
    cv_results = cross_validate(
        model, X_train, y_train, cv=5, scoring=["accuracy", "f1_macro"]
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    test_macro_f1 = f1_score(y_test, y_pred, average="macro")
    print(f"Test Macro F1: {test_macro_f1:.4f}")
    if y_proba.shape[1] == 2:
        test_auc = roc_auc_score(y_test, y_proba[:, 1])
    else:
        test_auc = roc_auc_score(
            y_test, y_proba, multi_class="ovr", average="macro"
        )
    print(f"Test Macro ROC-AUC: {test_auc:.4f}\n")

    # 3. Per-class breakdown & confusion matrix
    print(classification_report(y_test, y_pred, target_names=names))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

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
        "cv_scores": cv_results,
        "confusion_matrix": cm,
    }

    return model, results



# --- RUN PIPELINE (Bottom of files.py) ---
chroma_client = chromadb.PersistentClient(path="./chroma_db")
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="microsoft/harrier-oss-v1-270m"
)

embs = chroma_client.get_or_create_collection(
    name="climate_fulltext_harrier",
    embedding_function=sentence_transformer_ef,
    metadata={"hnsw:space": "cosine"},
)

# Only scrape and embed if the collection is empty
if embs.count() == 0:
    print("Collection empty — running full scrape and embedding pipeline...")
    df = read_df()
    climate_df = cretate_climate_df(df)
    climate_df_scraped = add_full_text_to_df(climate_df, limit=None)
    embs = create_embedding(climate_df_scraped)
else:
    print(f"Loaded {embs.count()} existing embeddings from ChromaDB.")

data = embs.get(include=["embeddings", "metadatas"])
X_embeddings = np.array(data["embeddings"])
metadatas = data["metadatas"]

# 1. Evaluate 3-Class Polarity
print("\n>>> EVALUATING 3-CLASS POLARITY <<<")
y_3class, names_3class = build_multiclass_labels(metadatas, mode="3class_polarity")
model_3class, results_3class = classification(
    X=X_embeddings,
    y=y_3class,
    target_names=names_3class,
)

# 2. Evaluate 4-Class Substantive Framing
print("\n>>> EVALUATING 4-CLASS SUBSTANTIVE FRAMING <<<")
y_4class, names_4class = build_multiclass_labels(metadatas, mode="4class_framing")
model_4class, results_4class = classification(
    X=X_embeddings,
    y=y_4class,
    target_names=names_4class,
)



