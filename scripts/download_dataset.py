"""Download and verify the SMS spam dataset.
Usage:
    python scripts/download_dataset.py --url <URL> --out data/sms_spam.csv
"""
import argparse
import os
import pandas as pd


DEFAULT_URL = "https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv"


def download(url: str, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df = pd.read_csv(url, header=None)
    # The dataset is expected to have two columns: label and text
    if df.shape[1] < 2:
        raise SystemExit(f"Unexpected CSV format: expected >=2 columns, got {df.shape[1]}")
    df = df.iloc[:, :2]
    df.columns = ["label", "message"]
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--out", default="data/sms_spam.csv")
    args = p.parse_args()
    download(args.url, args.out)


if __name__ == "__main__":
    main()
