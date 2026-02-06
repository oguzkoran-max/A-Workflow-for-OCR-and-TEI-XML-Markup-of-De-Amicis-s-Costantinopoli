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
        
    # F1 Skoru
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
    """Metin normalizasyonu: Küçük harf ve fazla boşluk temizliği."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def calculate_f1(gold_text, model_text):
    """Kelime bazlı F1, Precision ve Recall hesaplar."""
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

GOLD_FILE_PATH = "gold.txt"
MODELS_FOLDER = "models"
OUTPUT_IMAGE_NAME = "ocr_benchmark_with_acrobat.png"

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

    df = pd.DataFrame(results)
    df = df.sort_values(by="F1 Score", ascending=False)

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

    plt.figure(figsize=(14, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    ax = df.plot(x="Model Name", y=["F1 Score", "Precision", "Recall"], 
                 kind="bar", figsize=(14, 8), width=0.85, rot=0, color=colors)

    plt.title("OCR Accuracy Comparison: Generative AI vs. Traditional OCR", fontsize=18, fontweight='bold', pad=20)
    plt.ylabel("Score (0.0 - 1.0)", fontsize=14)
    plt.xlabel("Models", fontsize=14)
    
    plt.ylim(0.80, 1.02) 
    
    plt.legend(loc='lower right', frameon=True, shadow=True, fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.6, which='major')
    
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





import os
import glob
import re
import difflib
from collections import Counter
import pandas as pd

GOLD_FILE_PATH = "gold.txt"
MODELS_FOLDER = "models"

def normalize_text(text):
   
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def analyze_character_errors(gold, model):
   
    matcher = difflib.SequenceMatcher(None, gold, model)
    confusions = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            gold_segment = gold[i1:i2]
            model_segment = model[j1:j2]
            
            # Sadece tek karakterlik veya kısa değişimleri analiz edelim (Büyük bloklar kayma olabilir)
            if len(gold_segment) == len(model_segment) and len(gold_segment) < 5:
                for g_char, m_char in zip(gold_segment, model_segment):
                    if g_char != m_char:
                        confusions.append(f"'{g_char}' -> '{m_char}'")
    
    return Counter(confusions)

def analyze_word_diffs(gold, model):
    
    gold_words = gold.split()
    model_words = model.split()
    
    matcher = difflib.SequenceMatcher(None, gold_words, model_words)
    
    missed_words = []      # Gold'da var, Modelde yok (Recall hatası)
    hallucinated_words = [] # Modelde var, Gold'da yok (Precision hatası)
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'delete': # Gold'dan silinmiş
            missed_words.extend(gold_words[i1:i2])
        elif tag == 'insert': # Modele eklenmiş
            hallucinated_words.extend(model_words[j1:j2])
        elif tag == 'replace': # Kelime değişmiş
            missed_words.extend(gold_words[i1:i2])
            hallucinated_words.extend(model_words[j1:j2])
            
    return Counter(missed_words), Counter(hallucinated_words)

def main():
    # Gold Veriyi Oku
    try:
        with open(GOLD_FILE_PATH, 'r', encoding='utf-8') as f:
            gold_text = normalize_text(f.read())
    except FileNotFoundError:
        print("Gold dosya bulunamadı.")
        return

    model_files = glob.glob(os.path.join(MODELS_FOLDER, "*.txt"))
    
    # Raporlama Başlıyor
    print("="*60)
    print("NİTELİKSEL HATA ANALİZİ RAPORU")
    print("="*60)

    for m_file in model_files:
        model_name = os.path.basename(m_file)
        print(f"\n🔍 İNCELENEN MODEL: {model_name}")
        print("-" * 40)
        
        with open(m_file, 'r', encoding='utf-8') as f:
            model_text = normalize_text(f.read())

        # 1. Karakter Karışıklıkları (Confusion Matrix)
        char_confusions = analyze_character_errors(gold_text, model_text)
        print("🔤 En Sık Karıştırılan Karakterler (Top 5):")
        if char_confusions:
            for pair, count in char_confusions.most_common(5):
                print(f"   • {pair} : {count} kez")
        else:
            print("   (Belirgin karakter hatası yok)")

        # 2. Kelime Hataları
        missed, hallucinations = analyze_word_diffs(gold_text, model_text)
        
        print("\n❌ Gözden Kaçan Kelimeler (False Negative - Top 3):")
        for word, count in missed.most_common(3):
            print(f"   • '{word}' ({count} kez)")
            
        print("\n👻 Uydurulan Kelimeler/Halüsinasyon (False Positive - Top 3):")
        for word, count in hallucinations.most_common(3):
            print(f"   • '{word}' ({count} kez)")
            
        print("." * 40)

if __name__ == "__main__":
    main()







    import os
import glob
import re
import difflib
import pandas as pd
from collections import Counter


GOLD_FILE_PATH = "gold.txt"
MODELS_FOLDER = "models"


OUTPUT_SCORES_CSV = "OCR_General_Scores.csv"
OUTPUT_DETAILS_CSV = "OCR_Error_Details.csv"


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
    
    if len(model_tokens) == 0: precision = 0
    else: precision = num_same / len(model_tokens)
        
    if len(gold_tokens) == 0: recall = 0
    else: recall = num_same / len(gold_tokens)
        
    if precision + recall == 0: f1 = 0
    else: f1 = 2 * (precision * recall) / (precision + recall)
        
    return f1, precision, recall

def analyze_errors(gold, model):
   
    matcher = difflib.SequenceMatcher(None, gold, model)
    char_confusions = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            g_seg = gold[i1:i2]
            m_seg = model[j1:j2]
            if len(g_seg) == len(m_seg) and len(g_seg) < 5:
                for g, m in zip(g_seg, m_seg):
                    if g != m:
                        char_confusions.append(f"'{g}' -> '{m}'")

   
    gold_words = gold.split()
    model_words = model.split()
    word_matcher = difflib.SequenceMatcher(None, gold_words, model_words)
    missed = []
    hallucinated = []
    
    for tag, i1, i2, j1, j2 in word_matcher.get_opcodes():
        if tag == 'delete':
            missed.extend(gold_words[i1:i2])
        elif tag == 'insert':
            hallucinated.extend(model_words[j1:j2])
        elif tag == 'replace':
            missed.extend(gold_words[i1:i2])
            hallucinated.extend(model_words[j1:j2])
            
    return Counter(char_confusions), Counter(missed), Counter(hallucinated)

def main():
   
    try:
        with open(GOLD_FILE_PATH, 'r', encoding='utf-8') as f:
            gold_text = normalize_text(f.read())
            print(f"✅ Gold data loaded. ({len(gold_text)} characters)")
    except FileNotFoundError:
        print(f"❌ ERROR: File '{GOLD_FILE_PATH}' not found.")
        return

    model_files = glob.glob(os.path.join(MODELS_FOLDER, "*.txt"))
    
    scores_data = []
    details_data = []

    print("⏳ Analyzing models, please wait...")
    
    for m_file in model_files:
        filename = os.path.basename(m_file)
        display_name = MODEL_MAPPING.get(filename, filename)
        
        try:
            with open(m_file, 'r', encoding='utf-8') as f:
                model_text = normalize_text(f.read())
            
            
            f1, prec, rec = calculate_f1(gold_text, model_text)
            scores_data.append({
                "Model": display_name,
                "F1 Score": f1,
                "Precision": prec,
                "Recall": rec
            })
            
           
            char_err, missed, hallucinated = analyze_errors(gold_text, model_text)
            
            
            for item, count in char_err.most_common(20):
                details_data.append({
                    "Model": display_name, 
                    "Category": "Character Error", 
                    "Content": item, 
                    "Frequency": count
                })
            
            for item, count in missed.most_common(20):
                details_data.append({
                    "Model": display_name, 
                    "Category": "Missed Word", 
                    "Content": item, 
                    "Frequency": count
                })
                
            for item, count in hallucinated.most_common(20):
                details_data.append({
                    "Model": display_name, 
                    "Category": "Hallucination (Added)", 
                    "Content": item, 
                    "Frequency": count
                })

        except Exception as e:
            print(f"⚠️ File reading error ({filename}): {e}")


    if scores_data:
        df_scores = pd.DataFrame(scores_data).sort_values(by="F1 Score", ascending=False)

        df_scores.to_csv(OUTPUT_SCORES_CSV, index=False, sep=',', encoding='utf-8-sig')
        print(f"✅ File created: {OUTPUT_SCORES_CSV}")
    
    if details_data:
        df_details = pd.DataFrame(details_data)
        
        df_details.to_csv(OUTPUT_DETAILS_CSV, index=False, sep=',', encoding='utf-8-sig')
        print(f"✅ File created: {OUTPUT_DETAILS_CSV}")

    print("\n🎉 Process complete! You can open the .csv files with Excel.")

if __name__ == "__main__":
    main()
