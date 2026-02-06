import os
import xml.etree.ElementTree as ET
import re


import pandas as pd
import matplotlib.pyplot as plt



DATA_DIR = "data"
RESULTS_DIR = "results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

GOLD_FILE = os.path.join(DATA_DIR, "gold.xml")

MODEL_FILES = {
    "ChatGPT": os.path.join(DATA_DIR, "chatgpt.xml"),
    "Gemini": os.path.join(DATA_DIR, "gemini.xml"),
    "Claude": os.path.join(DATA_DIR, "claude.xml"),
    "Copilot": os.path.join(DATA_DIR, "copilot.xml"),
}

ENTITY_TYPES = [
    "placeName",
    "persName",
    "ethnic",
]



def normalize_text(s):
    
    if s is None:
        return ""
    return " ".join(s.strip().split()).lower()

import re  

def parse_xml(path):
  
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    
    text = text.replace("```xml", "").replace("```", "")

    
    text = re.sub(r"<\?xml.*?\?>", "", text, flags=re.DOTALL)

   
    text = text.strip()

    
    wrapped = "<root>\n" + text + "\n</root>"

    
    root = ET.fromstring(wrapped)
    return root



def extract_entities_from_xml(root):
    
    entities = []

    for elem in root.iter():
        tag = elem.tag

        
        if "}" in tag:
            tag = tag.split("}", 1)[1]

        text = elem.text
        if text is None:
            continue

        norm_text = normalize_text(text)
        if not norm_text:
            continue

        if tag == "placeName":
            entities.append({
                "type": "placeName",
                "text": norm_text,
            })
        elif tag == "persName":
            entities.append({
                "type": "persName",
                "text": norm_text,
            })
        elif tag == "rs":
            rs_type = elem.attrib.get("type", "")
            if rs_type == "ethnic":
                entities.append({
                    "type": "ethnic",
                    "text": norm_text,
                })

    return entities


def to_entity_set(entities):
    
    s = set()
    for e in entities:
        s.add((e["type"], e["text"]))
    return s




def precision_recall_f1(tp, fp, fn):
    """Precision, recall, F1 hesapla."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def evaluate_model(gold_entities, pred_entities, entity_types):
    

    gold_set = to_entity_set(gold_entities)
    pred_set = to_entity_set(pred_entities)

    results = {}

    
    for et in entity_types:
        gold_type = {e for e in gold_set if e[0] == et}
        pred_type = {e for e in pred_set if e[0] == et}

        tp = len(gold_type & pred_type)
        fp = len(pred_type - gold_type)
        fn = len(gold_type - pred_type)

        p, r, f1 = precision_recall_f1(tp, fp, fn)
        results[et] = {
            "precision": p,
            "recall": r,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    # Overall
    tp_overall = len(gold_set & pred_set)
    fp_overall = len(pred_set - gold_set)
    fn_overall = len(gold_set - pred_set)

    p_o, r_o, f1_o = precision_recall_f1(tp_overall, fp_overall, fn_overall)
    results["overall"] = {
        "precision": p_o,
        "recall": r_o,
        "f1": f1_o,
        "tp": tp_overall,
        "fp": fp_overall,
        "fn": fn_overall,
    }

    return results




def plot_f1_scores(df, save_path):
    
    overall_df = df[df["entity_type"] == "overall"]

    plt.figure(figsize=(8, 5))
    plt.bar(overall_df["model"], overall_df["f1"])
    plt.ylabel("F1 skoru")
    plt.title("Modellere göre Overall F1")
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()




def main():
    
    gold_root = parse_xml(GOLD_FILE)
    gold_entities = extract_entities_from_xml(gold_root)

    rows = []

    
    for model_name, model_path in MODEL_FILES.items():
        print(f"Model değerlendiriliyor: {model_name}")

        pred_root = parse_xml(model_path)
        pred_entities = extract_entities_from_xml(pred_root)

        results = evaluate_model(
            gold_entities=gold_entities,
            pred_entities=pred_entities,
            entity_types=ENTITY_TYPES,
        )

        for et, metrics in results.items():
            rows.append({
                "model": model_name,
                "entity_type": et,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
            })

    df = pd.DataFrame(rows)
    scores_csv = os.path.join(RESULTS_DIR, "scores.csv")
    df.to_csv(scores_csv, index=False)
    print(f"Skorlar kaydedildi: {scores_csv}")

    plot_path = os.path.join(PLOTS_DIR, "overall_f1_by_model.png")
    plot_f1_scores(df, plot_path)
    print(f"Grafik kaydedildi: {plot_path}")


if __name__ == "__main__":
    main()
