import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding
import random
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

MODEL_NAME = "en_core_web_md"
OUTPUT_MODEL = "ner_model"
TRAIN_DATA_PATH = "train.spacy"
JSON_DATA_PATH = "ner_train_spacy.json"

print(f"Loading base model '{MODEL_NAME}'...")
nlp = spacy.load(MODEL_NAME)

if "ner" not in nlp.pipe_names:
    ner = nlp.add_pipe("ner", last=True)
else:
    ner = nlp.get_pipe("ner")

labels = ["PARTY", "AGREEMENT", "SECTION", "EXHIBIT", "DATE", "POLICY"]
for label in labels:
    ner.add_label(label)

print(f"Schema initialized with labels: {labels}")

def convert_to_docbin():
    from spacy.tokens import DocBin
    db = DocBin()
    
    if not Path(JSON_DATA_PATH).exists():
        print(f"Error: {JSON_DATA_PATH} not found.")
        return
        
    with open(JSON_DATA_PATH, "r", encoding="utf-8") as f:
        training_data = json.load(f)
    
    print(f"Converting {len(training_data)} examples for your custom schema...")
    skipped = 0
    for item in training_data:
        text = item.get("text") or item.get("clause")
        entities = item.get("entities", [])
        
        doc = nlp.make_doc(text)
        ents = []
        for ent_data in entities:
            if isinstance(ent_data, list):
                start, end, label = ent_data
            else:
                start, end, label = ent_data["start"], ent_data["end"], ent_data["label"]
            if label in labels:
                span = doc.char_span(start, end, label=label, alignment_mode="expand")
                if span is not None:
                    ents.append(span)
                else:
                    skipped += 1
        
        try:
            doc.ents = ents
            db.add(doc)
        except Exception:
            skipped += 1
            
    db.to_disk(TRAIN_DATA_PATH)
    print(f"Saved training binary to {TRAIN_DATA_PATH}. (Skipped {skipped} misaligned spans)")

convert_to_docbin()

doc_bin = spacy.tokens.DocBin().from_disk(TRAIN_DATA_PATH)
examples = []
for doc in doc_bin.get_docs(nlp.vocab):
    examples.append(Example.from_dict(doc, {"entities": [(e.start_char, e.end_char, e.label_) for e in doc.ents]}))

optimizer = nlp.resume_training()

unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]

EPOCHS = 40
print(f"Starting fine-tuning for {EPOCHS} epochs...")

with nlp.disable_pipes(*unaffected_pipes):
    for epoch in range(EPOCHS):
        random.shuffle(examples)
        losses = {}
        batches = minibatch(examples, size=compounding(4.0, 32.0, 1.001))
        for batch in batches:
            nlp.update(
                batch,
                drop=0.3,
                losses=losses,
                sgd=optimizer
            )
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:2d}/{EPOCHS} | NER Loss: {losses.get('ner', 0):.6f}")

nlp.to_disk(OUTPUT_MODEL)
print(f"Success! Corrected model saved to '{OUTPUT_MODEL}'")