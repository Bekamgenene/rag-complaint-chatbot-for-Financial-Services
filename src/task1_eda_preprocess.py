import os
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils.text_cleaning import clean_text

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RAW_CSV = DATA_DIR / "complaints.csv"
FILTERED_CSV = DATA_DIR / "filtered_complaints.csv"
REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"
REPORT_DIR.mkdir(exist_ok=True)

def load_data(nrows: int | None = None) -> pd.DataFrame:
    """Load complaints CSV with selected columns only.
    CFPB file is large; use low_memory=False to avoid dtype guessing issues.
    """
    cols = [
        "complaint_id",
        "product",
        "issue",
        "consumer_complaint_narrative",
        "date_received",
    ]
    df = pd.read_csv(RAW_CSV, usecols=cols, nrows=nrows, low_memory=False)
    return df


def initial_eda(df: pd.DataFrame) -> None:
    """Generate basic plots and stats, saving to reports/ folder."""
    # Distribution of complaints per product
    plt.figure(figsize=(8, 4))
    sns.countplot(y="product", data=df, order=df["product"].value_counts().index)
    plt.title("Complaint count by Product")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "product_distribution.png")

    # Narrative length histogram
    df["narrative_length"] = df["consumer_complaint_narrative"].fillna("").str.split().apply(len)
    plt.figure(figsize=(8, 4))
    sns.histplot(df["narrative_length"], bins=50)
    plt.title("Narrative length distribution (word count)")
    plt.xlabel("Words per narrative")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "narrative_length_hist.png")

    # Null narrative share
    total = len(df)
    nulls = df["consumer_complaint_narrative"].isna().sum()
    with open(REPORT_DIR / "eda_summary.txt", "w", encoding="utf-8") as f:
        f.write(
            f"Total records: {total}\n"
            f"Records without narrative: {nulls} ({nulls/total:.2%})\n"
            f"Median narrative length: {df['narrative_length'].median()} words\n"
        )


def filter_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    allowed_products = {
        "Credit card",
        "Personal loan",
        "Buy Now, Pay Later (BNPL)",
        "Savings account",
        "Money transfers",
    }
    df = df[df["product"].isin(allowed_products)].copy()
    df = df.dropna(subset=["consumer_complaint_narrative"])
    df["clean_narrative"] = df["consumer_complaint_narrative"].apply(clean_text)
    df = df[df["clean_narrative"].str.len() > 0]
    return df


def main(sample: int | None = None):
    df = load_data(nrows=sample)
    initial_eda(df)
    filtered = filter_and_clean(df)
    filtered.to_csv(FILTERED_CSV, index=False)
    print(
        f"Saved cleaned dataset with {len(filtered):,} rows to {FILTERED_CSV.relative_to(Path.cwd())}"
    )


if __name__ == "__main__":
    # For interactive runs one might pass sample=10000 for speed.
    main()
