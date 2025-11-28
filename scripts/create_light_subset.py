import json
import random
from pathlib import Path

import requests

random.seed(42)
BASE = "https://huggingface.co/datasets/vectara/open_ragbench/resolve/main/official"


def create_light_subset(
    total_docs: int = 20,  # e.g. 20 → 8 positive + 12 hard-negative
    out_folder: str = "my_light_ragbench",
    *,  # keyword-only args
    download_pdf: bool = True,
):
    out = Path(out_folder)
    (out / "pdfs").mkdir(parents=True, exist_ok=True)
    (out / "corpus").mkdir(parents=True, exist_ok=True)
    (out / "qa").mkdir(parents=True, exist_ok=True)

    print(f"Creating light subset with {total_docs} documents (4:6 ratio)...")

    # 1. Load only the tiny metadata files we need
    pdf_urls = requests.get(f"{BASE}/pdf/arxiv/pdf_urls.json", timeout=30).json()
    paper_to_url = {x["paper_id"]: x["url"] for x in pdf_urls}
    all_papers = list(paper_to_url.keys())

    qrels = requests.get(f"{BASE}/qa/arxiv/qrels.json", timeout=30).json()
    queries_full = requests.get(f"{BASE}/qa/arxiv/queries.json", timeout=30).json()
    answers_full = requests.get(f"{BASE}/qa/arxiv/answers.json", timeout=30).json()

    # 2. Find the official 400 positive papers
    positive_papers = {
        d["doc_id"]
        for rel in qrels.values()
        for d in (rel if isinstance(rel, list) else [rel])
    }
    positive_papers = list(positive_papers)  # exactly 400
    negative_papers = [p for p in all_papers if p not in positive_papers]  # exactly 600

    # 3. Sample preserving 4:6 ratio
    num_pos = min(int(total_docs * 0.4), len(positive_papers))
    num_neg = total_docs - num_pos

    selected_pos = random.sample(positive_papers, num_pos)
    selected_neg = random.sample(negative_papers, num_neg)
    selected_all = selected_pos + selected_neg

    print(f"→ {num_pos} positive + {num_neg} hard-negative documents selected")

    # 4. Find queries that belong to our selected positive documents
    relevant_qids = set()
    for qid, rel in qrels.items():
        docs = rel if isinstance(rel, list) else [rel]
        if any(d["doc_id"] in selected_pos for d in docs):
            relevant_qids.add(qid)

    print(f"→ {len(relevant_qids)} queries kept")

    # 5. Download ONLY the files we actually need
    print("Downloading only the selected files...")
    for paper_id in selected_all:
        # Corpus JSON (tiny, always useful)
        url = f"{BASE}/pdf/arxiv/corpus/{paper_id}.json"
        (out / "corpus" / f"{paper_id}.json").write_bytes(
            requests.get(url, timeout=30).content,
        )

        if download_pdf:
            pdf_url = paper_to_url[paper_id]
            (out / "pdfs" / f"{paper_id}.pdf").write_bytes(
                requests.get(pdf_url, timeout=30).content,
            )

    # 6. Save filtered QA files
    subset_q = {qid: queries_full[qid] for qid in relevant_qids}
    subset_a = {qid: answers_full[qid] for qid in relevant_qids}
    subset_r = {qid: qrels[qid] for qid in relevant_qids}

    (out / "qa" / "queries.json").write_text(json.dumps(subset_q, indent=2))
    (out / "qa" / "answers.json").write_text(json.dumps(subset_a, indent=2))
    (out / "qa" / "qrels.json").write_text(json.dumps(subset_r, indent=2))

    # Save info
    (out / "info.txt").write_text(
        f"Light Open-RAG-Bench subset\n"
        f"Total documents : {len(selected_all)}\n"
        f"Positive (gold) : {num_pos}\n"
        f"Hard negatives  : {num_neg}\n"
        f"Queries         : {len(relevant_qids)}\n"
        f"Preserves official 4:6 ratio\n"
        f"No full download required!",
    )

    print(f"\nDone! Your light subset is ready → {out.resolve()}")
    print("Size: ~150-300 MB (perfect for any laptop)")
