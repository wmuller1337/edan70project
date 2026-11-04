import urllib.request
import zipfile
import io
import glob
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions


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

    # setup embedding function
    sentence_transformer_ef = (
        embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    )

    collection = chroma_client.get_or_create_collection(
        name="climate_refugees",
        embedding_function=sentence_transformer_ef,
        metadata={"hnsw:space": "cosine"},  # cosine similarity
    )

    documents = []
    metadatas = []
    ids = []

    for idx, row in df.iterrows():
        # Construct a descriptive text representation for embedding:
        # Combining the source, themes, and identifier gives the vector model rich context.
        # ONLY SAVES PARAMETERS, RATHER THAN FULL TEXT
        doc_text = f"Article from {row['SourceCommonName']}. Themes: {row['V2Themes']}. Link: {row['DocumentIdentifier']}"

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
        f"Successfully stored! Total records in Chroma: {collection.count()}"
    )


df = read_df()
climate_df = cretate_climate_df(df)
create_embedding(climate_df)
