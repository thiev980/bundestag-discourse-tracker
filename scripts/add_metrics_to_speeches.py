#!/usr/bin/env python3
"""
Bundestag Reden - Metriken berechnen (Vertex AI Edition)
========================================================
Dieses Script berechnet Embeddings und Sentiment mit Vertex AI
und speichert alles in BigQuery.

Features:
  - Embeddings: Vertex AI text-embedding-004
  - Sentiment: Gemini Flash (deutsche Parlamentssprache)
  - Emotionalität: Heuristische Analyse
  - Textmetriken: Wörter, Sätze, etc.

Voraussetzungen:
    pip install pandas google-cloud-bigquery google-cloud-aiplatform db-dtypes tqdm

Ausführen:
    python add_metrics_to_speeches.py

Geschätzte Kosten: ~$4 für 3.500 Reden

Autor: Bundestag Discourse Tracker Project
"""

import os
import re
import json
import time
import pandas as pd
from google.cloud import bigquery
from google.cloud import aiplatform
import vertexai
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel
from tqdm import tqdm
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# KONFIGURATION
# ============================================================

PROJECT_ID = "bt-discourse-tracker"
DATASET_ID = "bundestag_data"
SOURCE_TABLE = "speeches_raw"
TARGET_TABLE = "speeches_with_metrics"
LOCATION = "EU"
VERTEX_LOCATION = "europe-west1"

# Batch-Größen
EMBEDDING_BATCH_SIZE = 5  # Vertex AI Limit
SENTIMENT_BATCH_SIZE = 10  # Gemini kann mehr auf einmal

# Rate Limiting (konservativer wegen 503 Errors)
REQUESTS_PER_MINUTE = 30
SLEEP_BETWEEN_BATCHES = 2.5  # Sekunden (erhöht für Stabilität)

# Retry-Konfiguration
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 5  # Sekunden

# Checkpoint-Datei (um Fortschritt zu speichern)
CHECKPOINT_FILE = "embeddings_checkpoint.json"

# ============================================================
# VERTEX AI SETUP
# ============================================================

def init_vertex_ai():
    """Initialisiert Vertex AI."""
    print(f"  🔌 Initialisiere Vertex AI in {VERTEX_LOCATION}...")
    vertexai.init(project=PROJECT_ID, location=VERTEX_LOCATION)
    print("  ✓ Vertex AI initialisiert")


# ============================================================
# EMBEDDINGS MIT VERTEX AI
# ============================================================

def get_embeddings_batch(texts: List[str], model: TextEmbeddingModel) -> List[List[float]]:
    """
    Berechnet Embeddings für eine Liste von Texten.
    Mit Retry-Logik für 503 Errors.
    
    Args:
        texts: Liste von Texten (max 5 pro Batch wegen API-Limit)
        model: TextEmbeddingModel Instanz
    
    Returns:
        Liste von Embedding-Vektoren (768 Dimensionen)
    """
    
    # Texte vorbereiten (kürzen auf ~2000 Wörter für Kosten)
    prepared = []
    for text in texts:
        if not text:
            text = " "  # Leere Texte vermeiden
        # Kürzen auf ca. 10.000 Zeichen (ca. 2000 Wörter)
        if len(text) > 10000:
            text = text[:10000] + "..."
        prepared.append(text)
    
    # Retry-Logik mit exponential backoff
    for attempt in range(MAX_RETRIES):
        try:
            embeddings = model.get_embeddings(prepared)
            return [e.values for e in embeddings]
        except Exception as e:
            error_msg = str(e)
            if "503" in error_msg or "Broken pipe" in error_msg or "reset by peer" in error_msg:
                wait_time = INITIAL_RETRY_DELAY * (2 ** attempt)  # Exponential backoff
                print(f"\n    ⚠️ Server-Fehler (Versuch {attempt + 1}/{MAX_RETRIES}), warte {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"\n    ⚠️ Embedding-Fehler: {e}")
                break  # Anderer Fehler, nicht retrien
    
    # Fallback nach allen Retries: Null-Vektoren
    print(f"    ❌ Embedding fehlgeschlagen nach {MAX_RETRIES} Versuchen")
    return [[0.0] * 768 for _ in texts]


# ============================================================
# SENTIMENT MIT GEMINI
# ============================================================

def analyze_sentiment_gemini(texts: List[str], model: GenerativeModel) -> List[Dict]:
    """
    Analysiert Sentiment mit Gemini Flash.
    
    Gibt zurück: Liste von {label, score, reasoning}
    """
    
    # Prompt für Batch-Analyse
    prompt = """Du bist ein Experte für die Analyse von Bundestagsreden. 
Analysiere das Sentiment der folgenden Reden. 

Für jede Rede, gib zurück:
- sentiment: "positive", "negative", oder "neutral"
- score: Zahl von -1.0 (sehr negativ) bis +1.0 (sehr positiv)
- reasoning: Kurze Begründung (max 10 Wörter)

Beachte den Kontext von Parlamentsdebatten:
- Kritik an Regierung/Opposition ist normal, nicht automatisch "negativ"
- Sachliche Kritik = neutral
- Emotionale Angriffe, Empörung = negativ
- Lob, Zustimmung, Optimismus = positiv

Antworte NUR mit einem JSON-Array, keine Erklärungen davor oder danach.

Reden:
"""
    
    # Texte kürzen und nummerieren
    for i, text in enumerate(texts):
        short_text = text[:1500] if text else "(leer)"
        prompt += f"\n--- Rede {i+1} ---\n{short_text}\n"
    
    prompt += """

Antwort als JSON-Array (eine Zeile pro Rede):
[
  {"sentiment": "...", "score": 0.0, "reasoning": "..."},
  ...
]"""

    # Retry-Logik für Gemini
    for attempt in range(MAX_RETRIES):
        try:
            response = model.generate_content(prompt)
            response_text = response.text.strip()
            
            # JSON extrahieren (manchmal wrapped in ```json```)
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            
            results = json.loads(response_text)
            
            # Validieren und normalisieren
            validated = []
            for r in results:
                validated.append({
                    "sentiment_label": r.get("sentiment", "neutral"),
                    "sentiment_score": float(r.get("score", 0.0)),
                    "sentiment_reasoning": r.get("reasoning", "")[:100]
                })
            
            return validated
            
        except json.JSONDecodeError as e:
            print(f"    ⚠️ JSON-Parse-Fehler, Versuch {attempt + 1}/{MAX_RETRIES}")
            time.sleep(2)
        except Exception as e:
            error_msg = str(e)
            if "503" in error_msg or "500" in error_msg:
                wait_time = INITIAL_RETRY_DELAY * (2 ** attempt)
                print(f"\n    ⚠️ Server-Fehler (Versuch {attempt + 1}/{MAX_RETRIES}), warte {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"    ⚠️ Gemini-Fehler: {e}")
                break
    
    # Fallback
    return [{"sentiment_label": "neutral", "sentiment_score": 0.0, "sentiment_reasoning": "Fehler"} for _ in texts]


# ============================================================
# WEITERE METRIKEN
# ============================================================

def calculate_emotionality(text: str) -> Dict:
    """
    Berechnet Emotionalitäts-Indikatoren.
    
    Gibt zurück:
    - exclamation_count: Anzahl Ausrufezeichen
    - question_count: Anzahl Fragezeichen
    - caps_ratio: Anteil Großbuchstaben (ohne Satzanfänge)
    - emotionality_score: Kombinierter Score (0-1)
    """
    
    if not text:
        return {
            "exclamation_count": 0,
            "question_count": 0,
            "caps_ratio": 0.0,
            "emotionality_score": 0.0
        }
    
    # Zähle Satzzeichen
    exclamations = text.count('!')
    questions = text.count('?')
    
    # Großbuchstaben-Anteil (grobe Approximation für "Schreien")
    letters = [c for c in text if c.isalpha()]
    if letters:
        caps_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
    else:
        caps_ratio = 0.0
    
    # Normalisiere auf Textlänge (pro 1000 Zeichen)
    text_len = max(len(text), 1)
    excl_normalized = exclamations / (text_len / 1000)
    quest_normalized = questions / (text_len / 1000)
    
    # Kombinierter Score (gewichtet)
    # Mehr Ausrufezeichen = emotionaler
    emotionality_score = min(1.0, (excl_normalized * 0.5 + quest_normalized * 0.2 + caps_ratio * 0.3) / 2)
    
    return {
        "exclamation_count": exclamations,
        "question_count": questions,
        "caps_ratio": round(caps_ratio, 4),
        "emotionality_score": round(emotionality_score, 4)
    }


def calculate_speech_metrics(text: str) -> Dict:
    """
    Berechnet zusätzliche Text-Metriken.
    """
    
    if not text:
        return {
            "word_count": 0,
            "sentence_count": 0,
            "avg_word_length": 0.0,
            "avg_sentence_length": 0.0
        }
    
    # Wörter zählen
    words = text.split()
    word_count = len(words)
    
    # Sätze zählen (grob: . ! ?)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_count = len(sentences)
    
    # Durchschnittliche Wortlänge
    if words:
        avg_word_length = sum(len(w) for w in words) / len(words)
    else:
        avg_word_length = 0.0
    
    # Durchschnittliche Satzlänge (in Wörtern)
    if sentence_count > 0:
        avg_sentence_length = word_count / sentence_count
    else:
        avg_sentence_length = 0.0
    
    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_word_length": round(avg_word_length, 2),
        "avg_sentence_length": round(avg_sentence_length, 2)
    }


# ============================================================
# HAUPTPROGRAMM
# ============================================================

def main():
    print("""
╔════════════════════════════════════════════════════════════╗
║     Bundestag Reden - Metriken berechnen                  ║
║     🚀 Vertex AI Edition (Embeddings + Gemini)            ║
╚════════════════════════════════════════════════════════════╝
    """)
    
    # Vertex AI initialisieren
    print("=" * 60)
    print("SETUP")
    print("=" * 60)
    init_vertex_ai()
    
    # BigQuery Client
    print("  🔌 Verbinde mit BigQuery...")
    client = bigquery.Client(project=PROJECT_ID, location=LOCATION)
    print(f"  ✓ Verbunden mit {PROJECT_ID}")
    
    # Modelle laden
    print("  🤖 Lade Vertex AI Modelle...")
    embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
    gemini_model = GenerativeModel("gemini-2.0-flash-001")
    print("  ✓ Modelle geladen")
    
    # Daten laden
    print("\n" + "=" * 60)
    print("SCHRITT 1: Reden aus BigQuery laden")
    print("=" * 60)
    
    query = f"""
    SELECT *
    FROM `{PROJECT_ID}.{DATASET_ID}.{SOURCE_TABLE}`
    ORDER BY datum, sitzungsnr, rede_id
    """
    
    print("  📥 Lade Daten...")
    df = client.query(query).to_dataframe()
    print(f"  ✓ {len(df):,} Reden geladen")
    
    # Kostenschätzung
    total_chars = df['text'].fillna('').str.len().sum()
    est_embedding_cost = (total_chars / 1000) * 0.00025
    est_gemini_cost = (len(df) * 1500 / 1000000) * 0.15  # Input tokens
    print(f"\n  💰 Geschätzte Kosten:")
    print(f"     Embeddings: ~${est_embedding_cost:.2f}")
    print(f"     Gemini: ~${est_gemini_cost:.2f}")
    print(f"     Gesamt: ~${est_embedding_cost + est_gemini_cost:.2f}")
    
    # Metriken berechnen
    print("\n" + "=" * 60)
    print("SCHRITT 2: Lokale Metriken berechnen")
    print("=" * 60)
    
    # 2a: Textmetriken (schnell)
    print("\n  📊 Berechne Textmetriken...")
    text_metrics = []
    for text in tqdm(df['text'].fillna(''), desc="  Textmetriken"):
        text_metrics.append(calculate_speech_metrics(text))
    
    text_metrics_df = pd.DataFrame(text_metrics)
    for col in text_metrics_df.columns:
        df[col] = text_metrics_df[col]
    
    # 2b: Emotionalität (schnell)
    print("\n  😤 Berechne Emotionalität...")
    emotion_metrics = []
    for text in tqdm(df['text'].fillna(''), desc="  Emotionalität"):
        emotion_metrics.append(calculate_emotionality(text))
    
    emotion_metrics_df = pd.DataFrame(emotion_metrics)
    for col in emotion_metrics_df.columns:
        df[col] = emotion_metrics_df[col]
    
    # Vertex AI Metriken
    print("\n" + "=" * 60)
    print("SCHRITT 3: Vertex AI Metriken (Embeddings + Sentiment)")
    print("=" * 60)
    
    texts = df['text'].fillna('').tolist()
    
    # 3a: Embeddings mit Vertex AI (mit Checkpointing)
    print("\n  🧠 Berechne Embeddings mit Vertex AI...")
    
    # Checkpoint laden falls vorhanden
    checkpoint_path = os.path.join(os.path.dirname(__file__) or '.', CHECKPOINT_FILE)
    all_embeddings = []
    start_batch = 0
    
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint = json.load(f)
                all_embeddings = checkpoint.get('embeddings', [])
                start_batch = checkpoint.get('next_batch', 0)
                print(f"  📂 Checkpoint gefunden: {len(all_embeddings)} Embeddings, starte bei Batch {start_batch}")
        except:
            print("  ⚠️ Checkpoint korrupt, starte von vorn")
            all_embeddings = []
            start_batch = 0
    
    total_batches = (len(texts) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE
    remaining_batches = total_batches - start_batch
    print(f"     ({remaining_batches} Batches à {SLEEP_BETWEEN_BATCHES}s = ca. {remaining_batches * SLEEP_BETWEEN_BATCHES / 60:.1f} Minuten)")
    
    try:
        for i in tqdm(range(start_batch * EMBEDDING_BATCH_SIZE, len(texts), EMBEDDING_BATCH_SIZE), 
                      desc="  Embeddings", initial=start_batch, total=total_batches):
            batch = texts[i:i + EMBEDDING_BATCH_SIZE]
            embeddings = get_embeddings_batch(batch, embedding_model)
            all_embeddings.extend(embeddings)
            
            # Checkpoint speichern alle 20 Batches
            current_batch = i // EMBEDDING_BATCH_SIZE
            if current_batch % 20 == 0 and current_batch > start_batch:
                with open(checkpoint_path, 'w') as f:
                    json.dump({
                        'embeddings': all_embeddings,
                        'next_batch': current_batch + 1
                    }, f)
            
            time.sleep(SLEEP_BETWEEN_BATCHES)  # Rate limiting
            
    except KeyboardInterrupt:
        # Speichere Fortschritt bei Abbruch
        current_batch = len(all_embeddings) // EMBEDDING_BATCH_SIZE
        with open(checkpoint_path, 'w') as f:
            json.dump({
                'embeddings': all_embeddings,
                'next_batch': current_batch
            }, f)
        print(f"\n\n  💾 Fortschritt gespeichert ({len(all_embeddings)} Embeddings)")
        print(f"     Starte das Script erneut um fortzufahren.")
        return
    
    # Checkpoint löschen wenn fertig
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("  ✓ Checkpoint gelöscht (alle Embeddings fertig)")
    
    # Embeddings als JSON-String speichern (BigQuery ARRAY wäre besser, aber komplexer)
    df['embedding'] = [json.dumps(e) for e in all_embeddings]
    
    # 3b: Sentiment mit Gemini
    print("\n  🎭 Berechne Sentiment mit Gemini Flash...")
    print(f"     (Dies dauert ca. {len(texts) // SENTIMENT_BATCH_SIZE * SLEEP_BETWEEN_BATCHES / 60:.1f} Minuten)")
    
    all_sentiments = []
    for i in tqdm(range(0, len(texts), SENTIMENT_BATCH_SIZE), desc="  Sentiment"):
        batch = texts[i:i + SENTIMENT_BATCH_SIZE]
        sentiments = analyze_sentiment_gemini(batch, gemini_model)
        all_sentiments.extend(sentiments)
        time.sleep(SLEEP_BETWEEN_BATCHES)  # Rate limiting
    
    sentiment_df = pd.DataFrame(all_sentiments)
    df['sentiment_label'] = sentiment_df['sentiment_label']
    df['sentiment_score'] = sentiment_df['sentiment_score']
    df['sentiment_reasoning'] = sentiment_df['sentiment_reasoning']
    
    # Statistiken anzeigen
    print("\n  📊 Sentiment-Verteilung:")
    print(df['sentiment_label'].value_counts().to_string())
    
    print("\n  📊 Durchschnittswerte:")
    print(f"     Sentiment-Score: {df['sentiment_score'].mean():.3f}")
    print(f"     Emotionalität: {df['emotionality_score'].mean():.3f}")
    print(f"     Ø Wörter/Rede: {df['word_count'].mean():.0f}")
    
    # Nach BigQuery speichern
    print("\n" + "=" * 60)
    print("SCHRITT 4: Nach BigQuery speichern")
    print("=" * 60)
    
    table_id = f"{PROJECT_ID}.{DATASET_ID}.{TARGET_TABLE}"
    
    print(f"  ⬆️  Speichere nach {table_id}...")
    
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        autodetect=True,
    )
    
    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()
    
    table = client.get_table(table_id)
    print(f"  ✓ {table.num_rows:,} Reden mit Metriken gespeichert")
    
    # Zusammenfassung
    print("\n" + "=" * 60)
    print("✅ METRIKEN BERECHNET")
    print("=" * 60)
    print(f"""
Neue Tabelle: {table_id}

Neue Spalten:
  • embedding (768-dim Vektor als JSON)
  • sentiment_label (positive/negative/neutral)
  • sentiment_score (-1 bis +1)
  • sentiment_reasoning (Gemini's Begründung)
  • emotionality_score (0 bis 1)
  • exclamation_count, question_count
  • word_count, sentence_count
  • avg_word_length, avg_sentence_length

Teste mit dieser Abfrage - Sentiment pro Fraktion:

SELECT 
  fraktion,
  COUNT(*) as reden,
  ROUND(AVG(sentiment_score), 3) as avg_sentiment,
  ROUND(AVG(emotionality_score), 3) as avg_emotionality,
  ROUND(AVG(word_count)) as avg_words
FROM `{table_id}`
WHERE fraktion IS NOT NULL
GROUP BY fraktion
ORDER BY avg_sentiment DESC

Nächster Schritt:
  → Sitzungs-Barometer berechnen (Aggregation + Perzentile)
  → Clustering mit den Embeddings
""")


if __name__ == "__main__":
    main()