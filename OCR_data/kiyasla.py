import os
import glob
import re
from collections import Counter


try:
    import pandas as pd
except ImportError:
    pd = None
    
def normalize_text(text):

    
    text = text.lower()
    
    
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()





def calculate_f1(gold_text, model_text):
    
    gold_tokens = gold_text.split()
    model_tokens = model_text.split()
    
    
    gold_counts = Counter(gold_tokens)
    model_counts = Counter(model_tokens)
    
    
    common = gold_counts & model_counts
    num_same = sum(common.values())
    
    
    if len(model_tokens) == 0:
        precision = 0
    else:
        precision = num_same / len(model_tokens)
        
    if len(gold_tokens) == 0:
        recall = 0
    else:
        recall = num_same / len(gold_tokens)
        
    
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
        
    return f1, precision, recall


def main():
   
    gold_file_path = "gold.txt"  
    models_folder = "models"     
    
    
    
    try:
        with open(gold_file_path, 'r', encoding='utf-8') as f:
            raw_gold = f.read()
            gold_text = normalize_text(raw_gold)
            print(f"Gold veri yüklendi. ({len(gold_text.split())} kelime)")
    except FileNotFoundError:
        print("HATA: Gold dosyası bulunamadı. Lütfen dosya yolunu kontrol et.")
        return

    
    model_files = glob.glob(os.path.join(models_folder, "*.txt"))
    
    results = []

    print("-" * 60)
    print(f"{'Model Adı':<30} | {'F1 Skoru':<10} | {'Precision':<10} | {'Recall':<10}")
    print("-" * 60)

    
    for m_file in model_files:
        model_name = os.path.basename(m_file)
        
        with open(m_file, 'r', encoding='utf-8') as f:
            raw_model = f.read()
            model_text = normalize_text(raw_model)
        
        f1, prec, rec = calculate_f1(gold_text, model_text)
        
        
        results.append({
            "model": model_name,
            "f1": f1,
            "precision": prec,
            "recall": rec
        })
        
        print(f"{model_name:<30} | {f1:.4f}     | {prec:.4f}     | {rec:.4f}")

    
    if results:
        best_model = max(results, key=lambda x: x['f1'])
        print("-" * 60)
        print(f"\n🏆 KAZANAN MODEL: {best_model['model']}")
        print(f"🥇 F1 Skoru: {best_model['f1']:.5f}")
    else:
        print("Karşılaştırılacak model dosyası bulunamadı.")


if __name__ == "__main__":
    main()


import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


GOLD_FILE_PATH = "gold.txt"
MODELS_FOLDER = "models"
OUTPUT_IMAGE_NAME = "ocr_model_comparison.png"


MODEL_MAPPING = {
    "chatgpt.txt": "ChatGPT 5.1 Pro",
    "claude.txt": "Claude Sonnet 4.5",
    "copilot.txt": "Copilot Smart (GPT-5)",
    "gemini.txt": "Gemini 3 Pro"
}

def normalize_text(text):
    """Normalize text: lowercase, remove extra whitespaces."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def calculate_f1(gold_text, model_text):
    """Calculates F1, Precision, and Recall based on Bag of Words."""
    gold_tokens = gold_text.split()
    model_tokens = model_text.split()
    
    gold_counts = Counter(gold_tokens)
    model_counts = Counter(model_tokens)
    
    # Intersection
    common = gold_counts & model_counts
    num_same = sum(common.values())
    
    # Precision
    if len(model_tokens) == 0: precision = 0
    else: precision = num_same / len(model_tokens)
        
    # Recall
    if len(gold_tokens) == 0: recall = 0
    else: recall = num_same / len(gold_tokens)
        
    # F1 Score
    if precision + recall == 0: f1 = 0
    else: f1 = 2 * (precision * recall) / (precision + recall)
        
    return f1, precision, recall

def main():
    # Load Gold Text
    try:
        with open(GOLD_FILE_PATH, 'r', encoding='utf-8') as f:
            gold_text = normalize_text(f.read())
            print(f"Dataset Loaded. Gold text length: {len(gold_text.split())} words.")
    except FileNotFoundError:
        print(f"Error: Gold file '{GOLD_FILE_PATH}' not found.")
        return

    # Find Model Files
    model_files = glob.glob(os.path.join(MODELS_FOLDER, "*.txt"))
    results = []

    for m_file in model_files:
        filename = os.path.basename(m_file)
        
        
        display_name = MODEL_MAPPING.get(filename, filename)
        
        with open(m_file, 'r', encoding='utf-8') as f:
            model_text = normalize_text(f.read())
        
        f1, prec, rec = calculate_f1(gold_text, model_text)
        
        results.append({
            "Model Name": display_name,
            "F1 Score": f1,
            "Precision": prec,
            "Recall": rec
        })

    # Create DataFrame and Sort
    df = pd.DataFrame(results)
    df = df.sort_values(by="F1 Score", ascending=False)

    # --- 2. PRINT ENGLISH TABLE ---
    print("\n" + "="*60)
    print("OCR BENCHMARK RESULTS (Comparison with Gold Standard)")
    print("="*60)
    print(df.to_string(index=False))
    print("="*60)
    
    best_model = df.iloc[0]
    print(f"\n🏆 WINNER: {best_model['Model Name']} (F1: {best_model['F1 Score']:.4f})")

    # --- 3. VISUALIZATION (GROUPED BAR CHART) ---
    plt.figure(figsize=(12, 7))
    
    # Plot
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # Blue, Orange, Green
    ax = df.plot(x="Model Name", y=["F1 Score", "Precision", "Recall"], 
                 kind="bar", figsize=(12, 7), width=0.8, rot=0, color=colors)

    plt.title("OCR Accuracy Comparison: Gold Standard vs. LLMs", fontsize=16, fontweight='bold')
    plt.ylabel("Score (0.0 - 1.0)", fontsize=12)
    plt.xlabel("Language Models", fontsize=12)
    plt.ylim(0.85, 1.05) # Y-axis zoomed in to show differences better (since scores are high)
    plt.legend(loc='lower right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Annotate bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.3f}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 9), textcoords='offset points', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE_NAME, dpi=300)
    print(f"\n📊 Graph saved successfully as: {OUTPUT_IMAGE_NAME}")

if __name__ == "__main__":
    main()




import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


GOLD_FILE_PATH = "gold.txt"
MODELS_FOLDER = "models"
OUTPUT_IMAGE_NAME = "ocr_comparison_zoomed.png"


MODEL_MAPPING = {
    "chatgpt.txt": "ChatGPT 5.1 Pro",
    "claude.txt": "Claude Sonnet 4.5",
    "copilot.txt": "Copilot Smart (GPT-5)",
    "gemini.txt": "Gemini 3 Pro"
}

def normalize_text(text):
    
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def calculate_f1(gold_text, model_text):
    
    gold_tokens = gold_text.split()
    model_tokens = model_text.split()
    
    gold_counts = Counter(gold_tokens)
    model_counts = Counter(model_tokens)
    
    
    common = gold_counts & model_counts
    num_same = sum(common.values())
    
    # Precision
    if len(model_tokens) == 0: precision = 0
    else: precision = num_same / len(model_tokens)
        
    # Recall
    if len(gold_tokens) == 0: recall = 0
    else: recall = num_same / len(gold_tokens)
        
    # F1 Skoru
    if precision + recall == 0: f1 = 0
    else: f1 = 2 * (precision * recall) / (precision + recall)
        
    return f1, precision, recall
















import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


GOLD_FILE_PATH = "gold.txt"
MODELS_FOLDER = "models"
OUTPUT_IMAGE_NAME = "ocr_comparison_acrobat.png"


MODEL_MAPPING = {
    "chatgpt.txt": "ChatGPT 5.1 Pro",
    "claude.txt": "Claude Sonnet 4.5",
    "copilot.txt": "Copilot Smart (GPT-5)",
    "gemini.txt": "Gemini 3 Pro",
    "acrobat.txt": "Adobe Acrobat Pro"
}

def normalize_text(text):
    text = text.lower()
    
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def calculate_f1(gold_text, model_text):
    gold_tokens = gold_text.split()
    model_tokens = model_text.split()
    
    gold_counts = Counter(gold_tokens)
    model_counts = Counter(model_tokens)
    
    common = gold_counts & model_counts
    num_same = sum(common.values())
    
    # Precision
    if len(model_tokens) == 0: precision = 0
    else: precision = num_same / len(model_tokens)
        
    # Recall
    if len(gold_tokens) == 0: recall = 0
    else: recall = num_same / len(gold_tokens)
        
    # F1 Skoru
    if precision + recall == 0: f1 = 0
    else: f1 = 2 * (precision * recall) / (precision + recall)




import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# --- 1. SETTINGS ---
GOLD_FILE_PATH = "gold.txt"
MODELS_FOLDER = "models"
OUTPUT_IMAGE_NAME = "ocr_benchmark_with_acrobat.png"

# Mapping filenames to Professional Names
MODEL_MAPPING = {
    "chatgpt.txt": "ChatGPT 5.1 Pro",
    "claude.txt": "Claude Sonnet 4.5",
    "copilot.txt": "Copilot Smart (GPT-5)",
    "gemini.txt": "Gemini 3 Pro",
    "acrobat.txt": "Adobe Acrobat Pro"
}

def normalize_text(text):
    """Normalize text: lowercase, remove extra whitespaces."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def calculate_f1(gold_text, model_text):
    """Calculates F1, Precision, and Recall."""
    gold_tokens = gold_text.split()
    model_tokens = model_text.split()
    
    gold_counts = Counter(gold_tokens)
    model_counts = Counter(model_tokens)
    
    # Intersection
    common = gold_counts & model_counts
    num_same = sum(common.values())
    
    # Precision
    if len(model_tokens) == 0: precision = 0
    else: precision = num_same / len(model_tokens)
        
    # Recall
    if len(gold_tokens) == 0: recall = 0
    else: recall = num_same / len(gold_tokens)
        
    # F1 Score
    if precision + recall == 0: f1 = 0
    else: f1 = 2 * (precision * recall) / (precision + recall)
        
    return f1, precision, recall

def main():
    # Load Gold Text
    try:
        with open(GOLD_FILE_PATH, 'r', encoding='utf-8') as f:
            gold_text = normalize_text(f.read())
            print(f"Dataset Loaded. Gold text length: {len(gold_text.split())} words.")
    except FileNotFoundError:
        print(f"Error: Gold file '{GOLD_FILE_PATH}' not found.")
        return

    # Find Model Files
    model_files = glob.glob(os.path.join(MODELS_FOLDER, "*.txt"))
    results = []

    for m_file in model_files:
        filename = os.path.basename(m_file)
        display_name = MODEL_MAPPING.get(filename, filename)
        
        try:
            with open(m_file, 'r', encoding='utf-8') as f:
                model_text = normalize_text(f.read())
            
            f1, prec, rec = calculate_f1(gold_text, model_text)
            
            results.append({
                "Model Name": display_name,
                "F1 Score": f1,
                "Precision": prec,
                "Recall": rec
            })
        except Exception as e:
            print(f"Error reading file {filename}: {e}")

    # Create DataFrame and Sort by F1
    df = pd.DataFrame(results)
    df = df.sort_values(by="F1 Score", ascending=False)

    # --- 2. PRINT TABLE (English) ---
    print("\n" + "="*65)
    print("OCR BENCHMARK RESULTS (LLMs vs. Traditional OCR)")
    print("="*65)
    print(df.to_string(index=False, formatters={
        'F1 Score': '{:.4f}'.format,
        'Precision': '{:.4f}'.format,
        'Recall': '{:.4f}'.format
    }))
    print("-" * 65)
    
    best_model = df.iloc[0]
    print(f"🏆 WINNER: {best_model['Model Name']} (F1: {best_model['F1 Score']:.4f})")

    # --- 3. VISUALIZATION ---
    plt.figure(figsize=(14, 8))
    
    # Define a color palette (Adding a distinct color for the 5th item)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    ax = df.plot(x="Model Name", y=["F1 Score", "Precision", "Recall"], 
                 kind="bar", figsize=(14, 8), width=0.85, rot=0, color=colors)

    plt.title("OCR Accuracy Comparison: Generative AI vs. Traditional OCR", fontsize=18, fontweight='bold', pad=20)
    plt.ylabel("Score (0.0 - 1.0)", fontsize=14)
    plt.xlabel("Models", fontsize=14)
    
    # *** ZOOM SETTING ***
    # Adjusted to 0.80 to comfortably show Acrobat's 0.88 score
    plt.ylim(0.80, 1.02) 
    
    plt.legend(loc='lower right', frameon=True, shadow=True, fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.6, which='major')
    
    # Annotate bars with values
    for p in ax.patches:
        val = p.get_height()
        if val > 0:
            ax.annotate(f'{val:.3f}', 
                        (p.get_x() + p.get_width() / 2., val), 
                        ha='center', va='bottom', 
                        xytext=(0, 5), textcoords='offset points', 
                        fontsize=10, fontweight='bold', color='#333333')

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE_NAME, dpi=300)
    print(f"\n📊 Chart saved successfully as: {OUTPUT_IMAGE_NAME}")

if __name__ == "__main__":
    main()








