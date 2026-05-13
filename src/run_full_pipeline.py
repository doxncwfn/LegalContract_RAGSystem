import argparse
import json
import re
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import spacy
from transformers import pipeline

warnings.filterwarnings("ignore")


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"
NER_MODEL_DIR = ROOT / "ner_model"


def clean_legal_text(text: str) -> str:
    text = re.sub(r"\[\*\*\*\]", "", text)
    text = re.sub(r"Page\s+\d+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_clauses(raw_text: str, nlp) -> List[str]:
    doc = nlp(clean_legal_text(raw_text))
    clauses: List[str] = []
    start_idx = 0

    for token in doc:
        split_here = False
        if token.text in [".", ";"]:
            split_here = True
        elif token.text == ":":
            if token.i + 1 < len(doc) and doc[token.i + 1].text == "(":
                split_here = True
        elif token.dep_ == "cc":
            head = token.head
            if head.pos_ in ["VERB", "AUX"]:
                for child in head.children:
                    if child.dep_ == "conj" and child.pos_ in ["VERB", "AUX"]:
                        has_subject = any(
                            c.dep_ in ["nsubj", "nsubjpass"] for c in child.children
                        )
                        if has_subject:
                            split_here = True
                            break

        if split_here:
            end_idx = token.i + 1 if token.text in [".", ";", ":"] else token.i
            clause_span = doc[start_idx:end_idx].text.strip()
            if len(clause_span) > 10:
                clause_span = re.sub(r"[,]\s*$", "", clause_span)
                if not clause_span.endswith((".", ";", ":")):
                    clause_span += "."
                clauses.append(clause_span)
            start_idx = end_idx

    if start_idx < len(doc):
        final_span = doc[start_idx:].text.strip()
        if len(final_span) > 10:
            if not final_span.endswith("."):
                final_span += "."
            clauses.append(final_span)

    return clauses


def noun_phrase_chunking(clauses: List[str], nlp) -> List[str]:
    lines: List[str] = []
    excluded_pos = {"SPACE"}
    excluded_tokens = {",", ".", ";", ":", "(", ")"}

    for clause in clauses:
        doc = nlp(clause)
        tags = ["O"] * len(doc)

        for chunk in doc.noun_chunks:
            start_new = True
            for token in chunk:
                if token.pos_ in excluded_pos or token.text in excluded_tokens or token.is_space:
                    tags[token.i] = "O"
                    start_new = True
                else:
                    tags[token.i] = "B-NP" if start_new else "I-NP"
                    start_new = False

        for token, tag in zip(doc, tags):
            if token.text.strip():
                lines.append(f"{token.text}\t{tag}")
        lines.append("")

    return lines


def dependency_analysis(clauses: List[str], nlp) -> List[Dict]:
    payload = []
    for clause in clauses:
        doc = nlp(clause)
        deps = []
        for token in doc:
            if token.text.strip():
                deps.append(
                    {
                        "Token": token.text,
                        "Head": token.head.text,
                        "Dependency": token.dep_,
                    }
                )
        payload.append({"Clause": clause, "Dependencies": deps})
    return payload


def run_ner_inference(clauses: List[str]) -> List[Dict]:
    if not NER_MODEL_DIR.exists():
        raise FileNotFoundError(
            "Missing trained model at 'src/ner_model'. Train first with src/train_ner.py."
        )
    nlp_ner = spacy.load(str(NER_MODEL_DIR))

    results = []
    for clause in clauses:
        doc = nlp_ner(clause)
        entities = [
            {
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
            }
            for ent in doc.ents
        ]
        results.append({"clause": clause, "entities": entities})
    return results


def train_ner_model() -> None:
    train_script = ROOT / "train_ner.py"
    if not train_script.exists():
        raise FileNotFoundError(f"Missing training script: {train_script}")

    print("Training NER model (this may take a few minutes)...")
    subprocess.run([sys.executable, str(train_script)], check=True, cwd=str(ROOT))

    if not NER_MODEL_DIR.exists():
        raise RuntimeError("NER training completed but 'src/ner_model' was not created.")


def run_srl(ner_data: List[Dict]) -> List[Dict]:
    qa = pipeline("question-answering", model="deepset/minilm-uncased-squad2", framework="pt")

    role_questions = {
        "Predicate": "What is the main action, requirement, or verb?",
        "Agent": "Who is required to perform the action?",
        "Theme": "What is the object, topic, or document being acted upon?",
        "Recipient": "Who receives the payment, notice, or benefit?",
        "Time": "When does this happen or what is the duration?",
        "Condition": "Under what condition, exception, or requirement does this apply?",
    }

    results = []
    for item in ner_data:
        clause = item.get("clause", "")
        if not clause.strip() or len(clause.split()) < 4:
            continue

        roles = {}
        predicate = "Unknown"
        for role, question in role_questions.items():
            try:
                ans = qa(question=question, context=clause)
                if ans["score"] > 0.05:
                    if role == "Predicate":
                        predicate = ans["answer"]
                    else:
                        roles[role] = ans["answer"]
            except Exception:
                continue

        if predicate != "Unknown":
            results.append(
                {
                    "clause": clause,
                    "predicate": predicate,
                    "roles": roles,
                    "entities_present": [ent["text"] for ent in item.get("entities", [])],
                }
            )
    return results


def classify_clause_intent(clauses: List[str]) -> List[Dict]:
    rules: List[Tuple[str, str, str]] = [
        (r"\b(shall not|may not|must not|in no event shall)\b", "Prohibition", "prohib_keyword"),
        (r"\b(if|provided that|provided however|in the event)\b", "Condition", "cond_keyword"),
        (r"\b(shall|must|agrees to|will)\b", "Obligation", "oblig_keyword"),
        (r"\b(may|entitled to|reserves the right)\b", "Permission", "perm_keyword"),
        (r"\b(represents|warrants|acknowledges)\b", "Representation/Warranty", "repr_keyword"),
        (r"\b(shall mean|referred to as|defined as|means)\b", "Definition", "def_keyword"),
    ]
    structural_re = re.compile(r"\b(section|exhibit|schedule|appendix)\b", re.IGNORECASE)

    outputs = []
    for idx, clause in enumerate(clauses):
        text_lower = clause.lower()
        if structural_re.search(text_lower):
            outputs.append(
                {
                    "clause_id": idx,
                    "clause": clause,
                    "intent": "Scope/Structure",
                    "trigger_rule": "struct_keyword",
                }
            )
            continue

        best = ("Other", "no_match")
        earliest = float("inf")
        for pattern, intent, trigger in rules:
            m = re.search(pattern, text_lower)
            if m and m.start() < earliest:
                earliest = m.start()
                best = (intent, trigger)

        outputs.append(
            {
                "clause_id": idx,
                "clause": clause,
                "intent": best[0],
                "trigger_rule": best[1],
            }
        )
    return outputs


def read_contract(input_name: str) -> str:
    path = INPUT_DIR / input_name
    if not path.exists():
        raise FileNotFoundError(f"Input contract not found: {path}")
    return path.read_text(encoding="utf-8", errors="ignore")


def write_lines(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Run Assignment 1 + Assignment 2 legal NLP pipeline."
    )
    parser.add_argument(
        "--input",
        default="SPONSORSHIP_AGREEMENT.txt",
        help="Input file name under src/input/",
    )
    parser.add_argument(
        "--skip-srl",
        action="store_true",
        help="Skip SRL stage (useful to avoid model download time).",
    )
    parser.add_argument(
        "--force-train-ner",
        action="store_true",
        help="Always retrain NER before inference.",
    )
    args = parser.parse_args()

    stem = Path(args.input).stem
    nlp_parse = spacy.load("en_core_web_sm")

    print(f"[1/6] Reading contract: {args.input}")
    raw_text = read_contract(args.input)

    print("[2/6] Clause splitting")
    clauses = extract_clauses(raw_text, nlp_parse)
    clause_file = OUTPUT_DIR / f"{stem}_clauses.txt"
    write_lines(clause_file, clauses)

    print("[3/6] Noun phrase chunking (IOB)")
    chunks = noun_phrase_chunking(clauses, nlp_parse)
    chunk_file = OUTPUT_DIR / f"{stem}_chunks.txt"
    write_lines(chunk_file, chunks)

    print("[4/6] Dependency analysis")
    deps = dependency_analysis(clauses, nlp_parse)
    dep_file = OUTPUT_DIR / f"{stem}_dependency.json"
    write_json(dep_file, deps)

    print("[5/6] NER training + inference")
    if args.force_train_ner or not NER_MODEL_DIR.exists():
        train_ner_model()
    ner_results = run_ner_inference(clauses)
    ner_file = OUTPUT_DIR / "ner_results.json"
    write_json(ner_file, ner_results)

    print("[6/6] Intent classification")
    intent_results = classify_clause_intent(clauses)
    write_json(OUTPUT_DIR / "intent_classification_final.json", intent_results)
    write_lines(
        OUTPUT_DIR / "intent_classification_final.txt",
        [f"{row['intent']}\t{row['clause']}" for row in intent_results],
    )

    if args.skip_srl:
        print("SRL skipped (--skip-srl).")
    else:
        print("[Optional] SRL")
        try:
            srl_results = run_srl(ner_results)
            write_json(OUTPUT_DIR / "srl_results.json", srl_results)
        except Exception as exc:
            print(f"SRL stage failed and was skipped: {exc}")
            print("Tip: run with --skip-srl or reinstall requirements in a clean virtualenv.")

    print("Pipeline completed successfully.")
    print(f"- Clauses: {clause_file}")
    print(f"- Chunks: {chunk_file}")
    print(f"- Dependencies: {dep_file}")
    print(f"- NER: {ner_file}")
    if not args.skip_srl:
        print(f"- SRL: {OUTPUT_DIR / 'srl_results.json'}")
    print(f"- Intent: {OUTPUT_DIR / 'intent_classification_final.json'}")


if __name__ == "__main__":
    main()
