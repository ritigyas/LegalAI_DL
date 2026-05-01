# import os
# import json
# import sys
# from datetime import datetime
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# from core.nlp_pipeline import process_query
# from core.nlp_pipeline import process_query
# from core.retriever import search_indian_kanoon
# from core.reranker import rerank


# os.makedirs("trained_model", exist_ok=True)


# queries = []

# with open("Query_doc.txt", "r", encoding="utf-8") as f:
#     for line in f:
#         try:
#             qid, text = line.strip().split("||")
#             queries.append({"id": qid, "text": text})
#         except:
#             continue

# # use subset
# queries = queries[:20]

# print("\nInitializing Legal AI Optimization Pipeline...\n")

# logs = []


# for i, q in enumerate(queries):
#     query = q["text"]

#     # step 1 classify
#     domain = process_query(query)

#     # step 2 retrieve
#     retrieved_cases = search_indian_kanoon(query)

#     # step 3 rerank
#     ranked_cases = rerank(query, retrieved_cases) if retrieved_cases else []

#     log = {
#         "query_id": q["id"],
#         "domain": domain,
#         "retrieved_cases": retrieved_cases[:5],
#         "ranked_cases": ranked_cases[:3]
#     }

#     logs.append(log)

#     print(f"[{i+1}/{len(queries)}] Processed Query ID: {q['id']}")
#     print("Detected Domain:", domain)
#     print("Top Ranked:", ranked_cases[:2])
#     print("-"*70)

# # -----------------------------
# # SAVE LOGS
# # -----------------------------
# with open("trained_model/training_logs.json", "w", encoding="utf-8") as f:
#     json.dump(logs, f, indent=4)

# # -----------------------------
# # SAVE STATUS
# # -----------------------------
# with open("trained_model/status.txt", "w") as f:
#     f.write(f"""
# Legal AI Optimization Completed
# Timestamp: {datetime.now()}
# Queries Processed: {len(queries)}
# Modules Updated:
# - Query Classification
# - Case Retrieval
# - Cross-Encoder Re-ranking
# """)

# print("\nSaving optimized pipeline artifacts...")
# print("Updating retrieval vectors...")
# print("Saving reranking metadata...")
# print("\nOptimization complete ✅")


import os
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# -----------------------------
# LOAD DATA
# -----------------------------
queries = []

with open("Query_doc.txt", "r", encoding="utf-8") as f:
    for line in f:
        try:
            _, text = line.strip().split("||")
            queries.append(text)
        except:
            continue

# -----------------------------
# LABEL CREATION (simple rules)
# -----------------------------
def label_query(q):
    q = q.lower()
    if "murder" in q or "accused" in q or "police" in q:
        return 0   # Criminal
    elif "salary" in q or "employee" in q:
        return 1   # Labour
    elif "property" in q or "land" in q:
        return 2   # Property
    elif "marriage" in q or "divorce" in q:
        return 3   # Family
    else:
        return 4   # General

labels = [label_query(q) for q in queries]

df = pd.DataFrame({
    "text": queries,
    "label": labels
})

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------
train_df, val_df = train_test_split(df, test_size=0.2)

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# -----------------------------
# TOKENIZER
# -----------------------------
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding=True)

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)

# -----------------------------
# MODEL
# -----------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=5
)

# -----------------------------
# TRAINING CONFIG
# -----------------------------
training_args = TrainingArguments(
    output_dir="./trained_model_real",
    num_train_epochs=1,   # fast training
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    logging_steps=10,
    save_strategy="epoch"
)

# -----------------------------
# TRAINER
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# -----------------------------
# TRAIN
# -----------------------------
print("Starting real fine-tuning...\n")
trainer.train()

# -----------------------------
# SAVE
# -----------------------------
model.save_pretrained("./trained_model_real")
tokenizer.save_pretrained("./trained_model_real")

print("\nFine-tuning completed ✅")