from transformers import pipeline

# classifier = pipeline("zero-shot-classification",
#                       model="valhalla/distilbart-mnli-12-1")

classifier = pipeline(
    "zero-shot-classification",
    model="typeform/distilbert-base-uncased-mnli"
)

labels = [
    "labour law",
    "criminal law",
    "constitutional law",
    "property law",
    "family law"
]

def process_query(query):
    query = query.lower()

    if "salary" in query or "wage" in query:
        return "Labour Law"
    elif "murder" in query or "theft" in query:
        return "Criminal Law"
    elif "property" in query:
        return "Property Law"
    elif "family" in query:
        return "Family Law"
    else:
        result = classifier(query, labels)
        return result["labels"][0]