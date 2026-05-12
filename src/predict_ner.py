import spacy
import json
from pathlib import Path

# Load trained model
nlp = spacy.load("ner_model")
print("Loaded trained model")

# Read clauses
clauses_path = "output/SPONSORSHIP_AGREEMENT_clauses.txt"
with open(clauses_path, "r", encoding="utf-8") as f:
    clauses = [line.strip() for line in f if line.strip()]

print(f"Processing {len(clauses)} clauses...")

results = []

for clause in clauses:
    doc = nlp(clause)
    entities = []
    
    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char
        })
    
    results.append({
        "clause": clause,
        "entities": entities
    })

# Save results
output_path = "output/ner_results.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Inference completed! Results saved to '{output_path}'")
print(f"Total entities extracted: {sum(len(r['entities']) for r in results)}")