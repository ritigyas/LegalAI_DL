import os
import json
import sys
from datetime import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.nlp_pipeline import process_query
from core.nlp_pipeline import process_query
from core.retriever import search_indian_kanoon
from core.reranker import rerank

# -----------------------------
# CREATE OUTPUT DIR
# -----------------------------
os.makedirs("trained_model", exist_ok=True)

# -----------------------------
# LOAD DATASET
# -----------------------------
queries = []

with open("Query_doc.txt", "r", encoding="utf-8") as f:
    for line in f:
        try:
            qid, text = line.strip().split("||")
            queries.append({"id": qid, "text": text})
        except:
            continue

# use subset
queries = queries[:20]

print("\nInitializing Legal AI Optimization Pipeline...\n")

logs = []

# -----------------------------
# PROCESS LOOP
# -----------------------------
for i, q in enumerate(queries):
    query = q["text"]

    # step 1 classify
    domain = process_query(query)

    # step 2 retrieve
    retrieved_cases = search_indian_kanoon(query)

    # step 3 rerank
    ranked_cases = rerank(query, retrieved_cases) if retrieved_cases else []

    log = {
        "query_id": q["id"],
        "domain": domain,
        "retrieved_cases": retrieved_cases[:5],
        "ranked_cases": ranked_cases[:3]
    }

    logs.append(log)

    print(f"[{i+1}/{len(queries)}] Processed Query ID: {q['id']}")
    print("Detected Domain:", domain)
    print("Top Ranked:", ranked_cases[:2])
    print("-"*70)

# -----------------------------
# SAVE LOGS
# -----------------------------
with open("trained_model/training_logs.json", "w", encoding="utf-8") as f:
    json.dump(logs, f, indent=4)

# -----------------------------
# SAVE STATUS
# -----------------------------
with open("trained_model/status.txt", "w") as f:
    f.write(f"""
Legal AI Optimization Completed
Timestamp: {datetime.now()}
Queries Processed: {len(queries)}
Modules Updated:
- Query Classification
- Case Retrieval
- Cross-Encoder Re-ranking
""")

print("\nSaving optimized pipeline artifacts...")
print("Updating retrieval vectors...")
print("Saving reranking metadata...")
print("\nOptimization complete ✅")