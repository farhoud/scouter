import os
import requests
import pdfplumber
from datasets import load_dataset
from pathlib import Path
import json

CACHE_DIR = Path(".cache/pdfs")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

API_URL = "http://localhost:8000/v1/ingest"


def download_pdf(url, filename):
    cache_path = CACHE_DIR / filename
    if cache_path.exists():
        print(f"Using cached {filename}")
        return cache_path
    print(f"Downloading {filename}")
    response = requests.get(url)
    response.raise_for_status()
    with open(cache_path, "wb") as f:
        f.write(response.content)
    return cache_path


def extract_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text.strip()


def ingest_document(title, content):
    data = {
        "title": title,
        "content": content,
        "metadata": {"source": "open-rag-bench"},
    }
    response = requests.post(API_URL, json=data)
    response.raise_for_status()
    print(f"Ingested {title}")


def seed_db(num_docs=5):
    # Load dataset info
    dataset = load_dataset("vectara/open_ragbench", split="train")
    # Assuming pdf_urls is available, but since it's separate, perhaps hardcode or load from HF
    # For simplicity, use sample URLs or assume
    # Actually, the dataset may not have pdf_urls directly, but from README, pdf_urls.json
    # Let's download the pdf_urls.json from HF
    urls_response = requests.get(
        "https://huggingface.co/datasets/vectara/open_ragbench/resolve/main/pdf_urls.json"
    )
    pdf_urls = urls_response.json()

    count = 0
    for doc_id, url in pdf_urls.items():
        if count >= num_docs:
            break
        filename = f"{doc_id}.pdf"
        try:
            pdf_path = download_pdf(url, filename)
            text = extract_text(pdf_path)
            if text:
                ingest_document(f"Paper {doc_id}", text)
                count += 1
        except Exception as e:
            print(f"Failed to process {doc_id}: {e}")


if __name__ == "__main__":
    seed_db()
