import urllib.request
import zipfile
import io
import glob
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from newspaper import Article

def scrape_article_text(url):
    try:
        # scrape the article text using newspaper3k
        article = Article(url, request_timeout=5)
        article.download()
        article.parse()
        
        # article.text contains ONLY the main body of the article
        return article.text
    except Exception as e:
        # If scraping fails return an empty string
        return ""

def add_full_text_to_df(df, limit=20):
    print(f"Scraping full text for the first {limit} articles...")
    
    # Working on a copy of the sample
    sample_df = df.head(limit).copy()
    
    # Getting the main text for each URL
    sample_df["FullText"] = sample_df["DocumentIdentifier"].apply(scrape_article_text)
    
    # Filter out articles where scraping failed (empty text)
    successful = sample_df[sample_df["FullText"].str.len() > 0]
    print(f"Successfully scraped {len(successful)} out of {limit} articles.")
    
    return sample_df
    


def read_df():

    #  download for a single 15-minute chunk
    url = "http://data.gdeltproject.org/gdeltv2/20240501120000.gkg.csv.zip"
    response = urllib.request.urlopen(url)
    zip_file = zipfile.ZipFile(io.BytesIO(response.read()))
    zip_file.extractall("./gkg_raw_data")

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

    files = glob.glob("./gkg_raw_data/*.gkg.csv")
    file_path = files[0]

    df = pd.read_csv(
            file_path,
            sep="\t",
            names=COLUMN_NAMES,
            header=None,
            on_bad_lines="skip",  # skips occasional malformed lines
            low_memory=False,
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
    # delete the existing collection if it exists, to start fresh
    try:
        chroma_client.delete_collection("climate_denial_discourse")
        print("Rensade den gamla databasen.")
    except Exception:
        pass  # If the collection doesn't exist, ignore the error

    # setup embedding function
    sentence_transformer_ef = (
        embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    )

    collection = chroma_client.get_or_create_collection(
        name="climate_denial_discourse",
        embedding_function=sentence_transformer_ef,
        metadata={"hnsw:space": "cosine"},  # cosine similarity
    )

    documents = []
    metadatas = []
    ids = []

    for idx, row in df.iterrows():
        # Check if scraped full text exists and is not empty
        full_text = str(row.get("FullText", "")).strip()
        
        if full_text:
            # Including the source and the first 1500 chars of full text
            # all-MiniLM-L6-v2 will automatically truncate at 256 tokens anyways
            # first 1500 characters is reasonable for context
            doc_text = f"Source: {row['SourceCommonName']}. Text: {full_text[:1500]}"
        else:
            continue  # Skip this record if no full text is available (change if we want to include records without full text)

        avg_tone = float(str(row["Tone"]).split(",")[0])

        documents.append(doc_text)
        metadatas.append({
            "source": str(row["SourceCommonName"]),
            "date": int(row["DATE"]),
            "url": str(row["DocumentIdentifier"]),
            "tone": avg_tone
        })
        ids.append(f"doc_{idx}")

    if documents:
        collection.add(documents=documents, metadatas=metadatas, ids=ids)
        print(f"Stored {len(documents)} articles with full text. Total in database: {collection.count()}")
    else:
        print("No articles with full text found to save.")

# retrieve the dataframe, filter for climate articles
df = read_df()
climate_df = cretate_climate_df(df)
# Scrape full text andadd it to the dataframe (limit=20 for testing purposes)
climate_df_scraped = add_full_text_to_df(climate_df, limit=20)
# Create embeddings for the scraped articles and store them in the ChromaDB collection
create_embedding(climate_df_scraped)
