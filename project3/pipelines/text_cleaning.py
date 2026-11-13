import json
import re
import unicodedata
import pandas as pd
from project3.config.settings import PROCESSED_DIR

def clean_text(value):
    if pd.isna(value):
        return ""

    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace("ك", "ک").replace("ي", "ی")
    text = re.sub(r"https?://\S+|www\.\S+", " ", text, flags=re.I)
    text = re.sub(r"(?<!\w)@\w+", " ", text)
    text = text.replace("#", " ")
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def process_comments():
    source = PROCESSED_DIR / "comments_verified.csv"
    if not source.exists():
        raise FileNotFoundError("Run Phase 2 Part 4 first.")

    data = pd.read_csv(source, encoding="utf-8-sig")

    data["clean_text"] = data["comment_text"].apply(clean_text)
    data["text_length"] = data["clean_text"].str.len()
    data["word_count"] = data["clean_text"].str.split().str.len()
    data["empty_after_cleaning"] = data["clean_text"].eq("")

    destination = PROCESSED_DIR / "comments_cleaned.csv"
    data.to_csv(destination, index=False, encoding="utf-8-sig")

    summary = {
        "total_comments": int(len(data)),
        "empty_after_cleaning": int(data["empty_after_cleaning"].sum()),
        "unique_cleaned_texts": int(data["clean_text"].nunique()),
        "average_word_count": round(float(data["word_count"].mean()), 2),
        "languages": data["language_type"].value_counts().to_dict()
    }

    output = PROCESSED_DIR / "text_cleaning_summary.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print("Sample cleaned comments:")
    for row in data[["comment_text", "clean_text"]].head(5).itertuples(index=False):
        print("Original:", row.comment_text)
        print("Cleaned :", row.clean_text)

    print("PHASE 3 PART 1 COMPLETED")
    return data

if __name__ == "__main__":
    process_comments()
