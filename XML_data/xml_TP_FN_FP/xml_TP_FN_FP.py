
import os
import xml.etree.ElementTree as ET
import pandas as pd
import re


DATA_DIR = "." 
GOLD_FILE = os.path.join(DATA_DIR, "gold.xml")
MODEL_FILES = {
    "ChatGPT": os.path.join(DATA_DIR, "chatgpt.xml"),
    "Gemini": os.path.join(DATA_DIR, "gemini.xml"),
    "Claude": os.path.join(DATA_DIR, "claude.xml"),
    "Copilot": os.path.join(DATA_DIR, "copilot.xml"),
}


ENTITY_TYPES = ["placeName", "persName", "rs"] 



def normalize_text(text):
 
    if not text:
        return ""
    
    return " ".join(text.split()).lower()

def extract_entities(xml_path):
 
    try:
       
        with open(xml_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        
        root = ET.fromstring(content)
        
        entities = []
        
        
        for elem in root.iter():
            tag = elem.tag.split('}')[-1] 
            
            if tag in ENTITY_TYPES:
                
                if tag == 'rs' and elem.get('type') != 'ethnic':
                    continue
                
                
                label = "ethnic" if (tag == 'rs' and elem.get('type') == 'ethnic') else tag
                
                raw_text = "".join(elem.itertext())
                norm_text = normalize_text(raw_text)
                
                if norm_text:
                    entities.append({
                        "type": label,
                        "text_raw": raw_text,
                        "text_norm": norm_text
                    })
        return entities
    except Exception as e:
        print(f"Hata ({xml_path}): {e}")
        return []

def compare_models(gold_entities, model_entities, model_name):
  
    results = []
    
    
    gold_set = {(x['type'], x['text_norm']) for x in gold_entities}
    model_set = {(x['type'], x['text_norm']) for x in model_entities}
    
    
    for g in gold_entities:
        key = (g['type'], g['text_norm'])
        if key in model_set:
            results.append({
                "Model": model_name,
                "Durum": "TP (Doğru Tespit)",
                "Etiket": g['type'],
                "Metin (Gold)": g['text_raw'],
                "Metin (Model)": g['text_raw'], 
                "Açıklama": "Model ve Gold tam eşleşti."
            })
        else:
            results.append({
                "Model": model_name,
                "Durum": "FN (Gözden Kaçan)",
                "Etiket": g['type'],
                "Metin (Gold)": g['text_raw'],
                "Metin (Model)": "-",
                "Açıklama": "Gold verisetinde var ama Model bulamadı."
            })
            
    
    for m in model_entities:
        key = (m['type'], m['text_norm'])
        if key not in gold_set:
            results.append({
                "Model": model_name,
                "Durum": "FP (Yanlış Alarm)",
                "Etiket": m['type'],
                "Metin (Gold)": "-",
                "Metin (Model)": m['text_raw'],
                "Açıklama": "Model buldu ama Gold verisetinde yok."
            })
            
    return results


all_results = []

print("Gold Standard işleniyor...")
gold_data = extract_entities(GOLD_FILE)

for model_name, path in MODEL_FILES.items():
    print(f"{model_name} analiz ediliyor...")
    model_data = extract_entities(path)
    if model_data:
        comparison = compare_models(gold_data, model_data, model_name)
        all_results.extend(comparison)


if all_results:
    df = pd.DataFrame(all_results)
    
    df.to_csv("Detayli_Hata_Analizi.csv", index=False, encoding='utf-8-sig') 
    print("\n✅ Analiz tamamlandı! 'Detayli_Hata_Analizi.csv' dosyası oluşturuldu.")
    print("Bu dosyayı açıp hakem için örnekleri seçebilirsiniz.")
    
    
    print("\n--- Özet İstatistikler ---")
    print(df.groupby(["Model", "Durum"]).size())
else:
    print("Hata: Veri çıkarılamadı.")




    import os
import xml.etree.ElementTree as ET
import pandas as pd
import re


GOLD_FILE = "gold.xml"
MODEL_FILES = {
    "ChatGPT": "chatgpt.xml",
    "Gemini": "gemini.xml",
    "Claude": "claude.xml",
    "Copilot": "copilot.xml",
}



def normalize_text(text):
    """
    Normalization Rule:
    1. Strip whitespace from ends.
    2. Collapse multiple spaces to one.
    3. Lowercase.
    """
    if not text:
        return ""
    return " ".join(text.split()).lower()

def extract_entities(xml_path):
    """
    Parses XML and returns a list of dictionaries.
    Handles 'placeName', 'persName', and 'rs type=ethnic'.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        entities = []
        
        for elem in root.iter():
            tag = elem.tag.split('}')[-1] 
            entity_type = None
            
            if tag == "placeName":
                entity_type = "placeName"
            elif tag == "persName":
                entity_type = "persName"
            elif tag == "rs" and elem.get("type") == "ethnic":
                entity_type = "ethnic"
            
            if entity_type:
                raw_text = "".join(elem.itertext())
                norm_text = normalize_text(raw_text)
                if norm_text:
                    entities.append({
                        "type": entity_type,
                        "text_raw": raw_text,
                        "text_norm": norm_text
                    })
        return entities
    except Exception as e:
        print(f"Error parsing {xml_path}: {e}")
        return []

def compare_models(gold_entities, model_entities, model_name):
    """
    Compares Gold vs Model entities using Set approach for unique errors.
    """
    results = []
    gold_set = {(x['type'], x['text_norm']) for x in gold_entities}
    model_set = {(x['type'], x['text_norm']) for x in model_entities}
    
   
    gold_raw_map = {(x['type'], x['text_norm']): x['text_raw'] for x in gold_entities}
    model_raw_map = {(x['type'], x['text_norm']): x['text_raw'] for x in model_entities}

    
    for g_type, g_norm in gold_set:
        g_raw = gold_raw_map[(g_type, g_norm)]
        if (g_type, g_norm) in model_set:
            results.append({
                "Model": model_name, "Error_Type": "TP", "Tag": g_type,
                "Gold_Text": g_raw, "Model_Text": g_raw, 
                "Description": "Correctly identified."
            })
        else:
            results.append({
                "Model": model_name, "Error_Type": "FN", "Tag": g_type,
                "Gold_Text": g_raw, "Model_Text": "MISSING", 
                "Description": "Present in Gold, missed by Model."
            })
            
    
    for m_type, m_norm in model_set:
        if (m_type, m_norm) not in gold_set:
            m_raw = model_raw_map[(m_type, m_norm)]
            results.append({
                "Model": model_name, "Error_Type": "FP", "Tag": m_type,
                "Gold_Text": "MISSING", "Model_Text": m_raw,
                "Description": "Detected by Model, not in Gold."
            })     
    return results


all_results = []
gold_data = extract_entities(GOLD_FILE)

for model_name, path in MODEL_FILES.items():
    if os.path.exists(path):
        model_data = extract_entities(path)
        all_results.extend(compare_models(gold_data, model_data, model_name))

df = pd.DataFrame(all_results)
df = df[["Model", "Error_Type", "Tag", "Gold_Text", "Model_Text", "Description"]]
df.to_csv("Detailed_Error_Analysis.csv", index=False, encoding='utf-8-sig')
print("File saved: Detailed_Error_Analysis.csv")


import os
import xml.etree.ElementTree as ET
import pandas as pd
from collections import Counter


GOLD_FILE = "gold.xml"
MODEL_FILES = {
    "ChatGPT": "chatgpt.xml",
    "Gemini": "gemini.xml",
    "Claude": "claude.xml",
    "Copilot": "copilot.xml",
}



def normalize_text(text):
    if not text:
        return ""
    return " ".join(text.split()).lower()

def extract_entities(xml_path):
    """
    Returns list of entities preserving order and duplicates (Instance-based).
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        entities = []
        for elem in root.iter():
            tag = elem.tag.split('}')[-1]
            entity_type = None
            
            if tag == "placeName":
                entity_type = "placeName"
            elif tag == "persName":
                entity_type = "persName"
            elif tag == "rs" and elem.get("type") == "ethnic":
                entity_type = "ethnic"
            
            if entity_type:
                raw_text = "".join(elem.itertext())
                norm_text = normalize_text(raw_text)
                if norm_text:
                    entities.append({
                        "type": entity_type,
                        "text_norm": norm_text,
                        "text_raw": raw_text
                    })
        return entities
    except Exception as e:
        print(f"Error parsing {xml_path}: {e}")
        return []

def compare_models_instance_based(gold_entities, model_entities, model_name):
    """
    Compares using Instance-Based (Bag of Words) matching.
    Counts occurrences (e.g. 5 'Galata' in Gold vs 4 'Galata' in Model -> 4 TP, 1 FN).
    """
    results = []
    
    
    gold_map = {}
    for item in gold_entities:
        key = (item['type'], item['text_norm'])
        if key not in gold_map: gold_map[key] = []
        gold_map[key].append(item['text_raw'])
        
    model_map = {}
    for item in model_entities:
        key = (item['type'], item['text_norm'])
        if key not in model_map: model_map[key] = []
        model_map[key].append(item['text_raw'])
    
    
    all_keys = set(gold_map.keys()) | set(model_map.keys())
    
    for key in all_keys:
        etype, enorm = key
        
        g_instances = gold_map.get(key, [])
        m_instances = model_map.get(key, [])
        
        count_g = len(g_instances)
        count_m = len(m_instances)
        
       
        tp_count = min(count_g, count_m)
        
        fn_count = count_g - tp_count
       
        fp_count = count_m - tp_count
        
      
        for i in range(tp_count):
            results.append({
                "Model": model_name, "Error_Type": "TP", "Tag": etype,
                "Gold_Text": g_instances[i], "Model_Text": m_instances[i],
                "Description": "Matched instance"
            })
            
       
        for i in range(fn_count):
            results.append({
                "Model": model_name, "Error_Type": "FN", "Tag": etype,
                "Gold_Text": g_instances[tp_count + i], "Model_Text": "MISSING",
                "Description": "Missed instance"
            })
            
        
        for i in range(fp_count):
            results.append({
                "Model": model_name, "Error_Type": "FP", "Tag": etype,
                "Gold_Text": "MISSING", "Model_Text": m_instances[tp_count + i],
                "Description": "Extra instance"
            })
            
    return results


all_results = []
gold_data = extract_entities(GOLD_FILE)

for model_name, path in MODEL_FILES.items():
    if os.path.exists(path):
        model_data = extract_entities(path)
        all_results.extend(compare_models_instance_based(gold_data, model_data, model_name))

df = pd.DataFrame(all_results)
df = df[["Model", "Error_Type", "Tag", "Gold_Text", "Model_Text", "Description"]]
df.to_csv("Detailed_Error_Analysis_Instances.csv", index=False, encoding='utf-8-sig')
print("File saved: Detailed_Error_Analysis_Instances.csv")




import os
import xml.etree.ElementTree as ET
import pandas as pd


GOLD_FILE = "gold.xml"
MODEL_FILES = {
    "ChatGPT": "chatgpt.xml",
    "Gemini": "gemini.xml",
    "Claude": "claude.xml",
    "Copilot": "copilot.xml",
}

def normalize_text(text):
    """Normalize text: lowercase and strip whitespace."""
    if not text: return ""
    return " ".join(text.split()).lower()

def extract_entities(xml_path):
    """Extract entities keeping all instances (no deduplication)."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        entities = []
        for elem in root.iter():
            tag = elem.tag.split('}')[-1]
            etype = None
            
            if tag == "placeName": etype = "placeName"
            elif tag == "persName": etype = "persName"
            elif tag == "rs" and elem.get("type") == "ethnic": etype = "ethnic"
            
            if etype:
                raw = "".join(elem.itertext())
                norm = normalize_text(raw)
                if norm:
                    entities.append({"type": etype, "norm": norm, "raw": raw})
        return entities
    except Exception as e:
        print(f"Error: {e}")
        return []

def match_entities(gold, model, model_name):
    """Match entities using instance counting (Bag of Entities)."""
    results = []
    
   
    gold_map = {}
    for x in gold:
        key = (x['type'], x['norm'])
        gold_map.setdefault(key, []).append(x['raw'])
        
    model_map = {}
    for x in model:
        key = (x['type'], x['norm'])
        model_map.setdefault(key, []).append(x['raw'])
        
    all_keys = set(gold_map.keys()) | set(model_map.keys())
    
    for key in all_keys:
        etype, enorm = key
        g_list = gold_map.get(key, [])
        m_list = model_map.get(key, [])
        
        tp = min(len(g_list), len(m_list))
        fn = len(g_list) - tp
        fp = len(m_list) - tp
        
    
        for i in range(tp):
            results.append({"Model": model_name, "Error_Type": "TP", "Entity_Type": etype, 
                            "Gold_Text": g_list[i], "Model_Text": m_list[i], "Description": "Correct Match"})
 
        for i in range(fn):
            results.append({"Model": model_name, "Error_Type": "FN", "Entity_Type": etype, 
                            "Gold_Text": g_list[tp+i], "Model_Text": "MISSING", "Description": "Missed (FN)"})
     
        for i in range(fp):
            results.append({"Model": model_name, "Error_Type": "FP", "Entity_Type": etype, 
                            "Gold_Text": "MISSING", "Model_Text": m_list[tp+i], "Description": "Spurious (FP)"})
    return results


all_res = []
gold_data = extract_entities(GOLD_FILE)
for m_name, m_path in MODEL_FILES.items():
    if os.path.exists(m_path):
        model_data = extract_entities(m_path)
        all_res.extend(match_entities(gold_data, model_data, m_name))

df = pd.DataFrame(all_res)
df.to_csv("Detailed_Error_Analysis_Final.csv", index=False, encoding='utf-8-sig')
print("Done.")

