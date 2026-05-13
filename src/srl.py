import os
import json
import warnings
from transformers import pipeline

warnings.filterwarnings("ignore")

def get_qa_pipeline():
    model_name = "deepset/minilm-uncased-squad2"
    print(f"Loading Deep Learning QA-SRL model '{model_name}'...")
    qa_pipe = pipeline("question-answering", model=model_name, framework="pt")
    return qa_pipe

def process_srl():
    input_file = "output/ner_results.json"
    output_file = "output/srl_results.json"

    if not os.path.exists(input_file):
        print(f"Error: Could not find '{input_file}'. Please run Task 2.1 first.")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        ner_data = json.load(f)

    qa_pipeline = get_qa_pipeline()

    results = []
    print(f"Running Deep Learning QA-SRL Inference on {len(ner_data)} clauses...")

    role_questions = {
        "Predicate": "What is the main action, requirement, or verb?",
        "Agent": "Who is required to perform the action?",
        "Theme": "What is the object, topic, or document being acted upon?",
        "Recipient": "Who receives the payment, notice, or benefit?",
        "Time": "When does this happen or what is the duration?",
        "Condition": "Under what condition, exception, or requirement does this apply?"
    }

    for item in ner_data:
        clause = item.get("clause", "")
        if not clause.strip() or len(clause.split()) < 4:
            continue

        roles = {}
        predicate = "Unknown"

        for role, question in role_questions.items():
            try:
                ans = qa_pipeline(question=question, context=clause)
                if ans['score'] > 0.05:
                    if role == "Predicate":
                        predicate = ans['answer']
                    else:
                        roles[role] = ans['answer']
            except Exception:
                continue

        if predicate != "Unknown":
            results.append({
                "clause": clause,
                "predicate": predicate,
                "roles": roles,
                "entities_present": [ent["text"] for ent in item.get("entities", [])]
            })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"SRL Task 2.2 Complete! Extracted robust roles for {len(results)} clauses.")
    print(f"Results successfully saved to '{output_file}'.")

if __name__ == "__main__":
    process_srl()