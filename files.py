import urllib.request
import zipfile
import io
import glob
import pandas as pd

# Example download for a single 15-minute chunk
url = "http://data.gdeltproject.org/gdeltv2/20240501120000.gkg.csv.zip"

print("Downloading batch...")
response = urllib.request.urlopen(url)
zip_file = zipfile.ZipFile(io.BytesIO(response.read()))

# Extracts the tab-delimited file (.gkg.csv)
zip_file.extractall("./gkg_raw_data")
print("Download complete.")


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

CLIMATE_THEMES = r"ENV_CLIMATECHANGE|ENV_GLOBALWARMING|ENV_GREENENERGY|ENV_CARBON|ENV_DROUGHT|ENV_FLOOD"

# Find the uncompressed .gkg.csv file in your folder
files = glob.glob("./gkg_raw_data/*.gkg.csv")
all_climate_articles = []

if not files:
    print(
        "No extracted .gkg.csv files found. Check your ./gkg_raw_data folder!"
    )
else:
    file_path = files[0]
    print(f"Loading sample rows from: {file_path}\n")

    # Read the first 5 rows
    df = pd.read_csv(
        file_path,
        sep="\t",
        names=COLUMN_NAMES,
        header=None,
        on_bad_lines="skip",  # skips occasional malformed lines
        low_memory=False,
    )
    
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