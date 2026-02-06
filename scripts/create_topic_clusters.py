#!/usr/bin/env python3
"""
Themen-Clustering für Bundestagsreden (Verbesserte Version)
===========================================================

Zwei Modi:
  1. ANALYSE: Elbow + Silhouette Plots → optimales k finden
  2. CLUSTERING: K-Means + Keywords + Gemini Labels

Methodik:
  - K-Means Clustering auf Vertex AI Embeddings
  - TF-IDF Keywords: Mittelwert pro Cluster
  - Discriminative Terms: Cluster-Mittel MINUS Korpus-Mittel
  - Gemini Flash für menschenlesbare Labels

Verwendung:
  python create_topic_clusters.py --analyze        # Erst: Plots anschauen
  python create_topic_clusters.py --cluster 12    # Dann: Mit k=12 clustern

Autor: Bundestag Discourse Tracker Project
"""

import os
import sys
import json
import time
import argparse
import pandas as pd
import numpy as np
from google.cloud import bigquery
import vertexai
from vertexai.generative_models import GenerativeModel
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# KONFIGURATION
# ============================================================

PROJECT_ID = "bt-discourse-tracker"
DATASET_ID = "bundestag_data"
SOURCE_TABLE = "speeches_with_metrics"
SPEECHES_TARGET = "speeches_with_clusters"
CLUSTERS_TARGET = "topic_clusters"
LOCATION = "EU"
VERTEX_LOCATION = "europe-west1"

# Analyse-Parameter
K_RANGE = range(5, 26)  # Teste k von 5 bis 25

# Clustering-Parameter
RANDOM_STATE = 42

# TF-IDF Parameter
TOP_KEYWORDS = 15  # Top TF-IDF Keywords
TOP_DISCRIMINATIVE = 15  # Top discriminative terms
MIN_DF = 5  # Mindestens in 5 Dokumenten
MAX_DF = 0.5  # Maximal in 50% der Dokumente

# Output
OUTPUT_DIR = "dashboard_data"
PLOTS_DIR = "plots"
OUTPUT_FILE = "topic_clusters.json"

# Deutsche Stopwords (erweitert für Parlamentssprache)
GERMAN_STOPWORDS = [
    'der', 'die', 'das', 'und', 'in', 'zu', 'den', 'ist', 'von', 'für',
    'mit', 'auf', 'des', 'eine', 'ein', 'dem', 'nicht', 'sich', 'auch',
    'es', 'an', 'als', 'werden', 'aus', 'dass', 'bei', 'sind', 'hat',
    'wir', 'haben', 'ich', 'wird', 'sie', 'er', 'aber', 'nach', 'noch',
    'wie', 'einem', 'einer', 'wenn', 'nur', 'kann', 'oder', 'so', 'zum',
    'diese', 'dieser', 'mehr', 'über', 'schon', 'vor', 'durch', 'muss',
    'sehr', 'hier', 'heute', 'gibt', 'müssen', 'immer', 'denn', 'damit',
    'dann', 'will', 'war', 'jetzt', 'zur', 'doch', 'etwa', 'beim', 'sollte',
    'worden', 'wurden', 'wurde', 'weil', 'können', 'sein', 'ihre', 'ihren',
    'eines', 'diesem', 'dieses', 'zwischen', 'ohne', 'unter', 'gegen',
    'allem', 'aller', 'alles', 'also', 'andere', 'anderen', 'anderer',
    'bereits', 'dabei', 'dadurch', 'dafür', 'dagegen', 'daher', 'damals',
    'darum', 'darunter', 'davon', 'dazu', 'deshalb', 'dessen', 'diejenigen',
    'dies', 'dieselbe', 'dieselben', 'diesen', 'diejenige', 'doch', 'dort',
    'dürfen', 'ebenfalls', 'ebenso', 'ehe', 'eigenen', 'eigentlich', 'einige',
    'einigen', 'einiger', 'einiges', 'einmal', 'ende', 'etwa', 'etwas',
    'falls', 'ganz', 'gar', 'genau', 'gerade', 'gern', 'gerne', 'gewesen',
    'hätte', 'hätten', 'habe', 'hin', 'hinter', 'höchstens', 'indem',
    'infolge', 'innen', 'innerhalb', 'insgesamt', 'irgend', 'irgendwie',
    'jedoch', 'jede', 'jedem', 'jeden', 'jeder', 'jedes', 'jemals', 'jemand',
    'jene', 'jenem', 'jenen', 'jener', 'jenes', 'kaum', 'kein', 'keine',
    'keinem', 'keinen', 'keiner', 'könnte', 'könnten', 'lassen', 'lässt',
    'machen', 'macht', 'manchmal', 'mehr', 'mehrere', 'meist', 'meisten',
    'möchte', 'mögen', 'nämlich', 'natürlich', 'neben', 'nichts', 'niemand',
    'nimmt', 'nun', 'obwohl', 'schließlich', 'seid', 'selbst', 'sogar',
    'solch', 'solche', 'solchem', 'solchen', 'solcher', 'sollen', 'sollten',
    'sondern', 'sonst', 'soweit', 'sowie', 'später', 'statt', 'trotz',
    'überhaupt', 'übrigens', 'unsere', 'unserem', 'unseren', 'unserer',
    'unten', 'während', 'wäre', 'wären', 'warum', 'weder', 'wegen', 'weiter',
    'weitere', 'weiteren', 'welche', 'welchem', 'welchen', 'welcher', 'wenig',
    'wenige', 'wenigstens', 'werde', 'wieder', 'wirklich', 'wissen', 'wohl',
    'wollen', 'wollte', 'würde', 'würden', 'ziemlich', 'zunächst', 'zwar',
    # Parlamentsspezifisch
    'herr', 'frau', 'kollege', 'kollegin', 'kollegen', 'kolleginnen',
    'präsident', 'präsidentin', 'abgeordnete', 'abgeordneten', 'fraktion',
    'antrag', 'anträge', 'gesetzentwurf', 'bundesregierung', 'regierung', 
    'bundestag', 'bundesrat', 'ausschuss', 'ausschüsse',
    'deutschland', 'deutschen', 'deutsche', 'deutscher', 'bitte', 'danke',
    'vielen', 'dank', 'rede', 'debatte', 'frage', 'fragen', 'antwort', 'thema',
    'liebe', 'lieber', 'verehrte', 'verehrten', 'meine', 'damen', 'herren',
    'punkt', 'punkte', 'seite', 'seiten', 'prozent', 'jahr', 'jahre', 'jahren',
    'million', 'millionen', 'milliarde', 'milliarden', 'euro', 'geld',
    'gesetz', 'gesetze', 'politik', 'politisch', 'politische', 'land', 'länder',
    'menschen', 'bürger', 'bürgerinnen', 'gesellschaft', 'staat', 'staaten',
]

# ============================================================
# DATEN LADEN
# ============================================================

def load_data(client: bigquery.Client) -> Tuple[pd.DataFrame, np.ndarray]:
    """Lädt Reden und Embeddings aus BigQuery."""
    
    query = f"""
    SELECT 
        rede_id,
        wahlperiode,
        sitzungsnr,
        datum,
        fraktion,
        vorname,
        nachname,
        text,
        embedding,
        sentiment_score,
        emotionality_score,
        word_count
    FROM `{PROJECT_ID}.{DATASET_ID}.{SOURCE_TABLE}`
    WHERE embedding IS NOT NULL 
      AND text IS NOT NULL
      AND LENGTH(text) > 100
    ORDER BY datum
    """
    
    print("  📥 Lade Daten aus BigQuery...")
    df = client.query(query).to_dataframe()
    print(f"  ✓ {len(df):,} Reden geladen")
    
    # Embeddings parsen
    print("  🔢 Parse Embeddings...")
    embeddings_list = []
    for e in tqdm(df['embedding'], desc="  Embeddings"):
        try:
            embeddings_list.append(np.array(json.loads(e)))
        except:
            embeddings_list.append(np.zeros(768))
    
    embeddings = np.array(embeddings_list)
    
    # Null-Embeddings filtern
    valid_mask = np.any(embeddings != 0, axis=1)
    print(f"  ✓ {valid_mask.sum():,} gültige Embeddings")
    
    df = df[valid_mask].reset_index(drop=True)
    embeddings = embeddings[valid_mask]
    
    return df, embeddings


# ============================================================
# ANALYSE: ELBOW + SILHOUETTE
# ============================================================

def run_analysis(embeddings: np.ndarray):
    """Berechnet Elbow und Silhouette Scores, erstellt Plots."""
    
    print("\n" + "=" * 60)
    print("ANALYSE: Optimale Cluster-Anzahl finden")
    print("=" * 60)
    
    inertias = []
    silhouette_scores = []
    
    for k in tqdm(K_RANGE, desc="  Teste k"):
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        
        inertias.append(kmeans.inertia_)
        sil = silhouette_score(embeddings, labels)
        silhouette_scores.append(sil)
    
    # Beste Silhouette
    best_k_idx = np.argmax(silhouette_scores)
    best_k = list(K_RANGE)[best_k_idx]
    best_sil = silhouette_scores[best_k_idx]
    
    print(f"\n  🏆 Bester Silhouette Score: {best_sil:.3f} bei k={best_k}")
    
    # Plots erstellen
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Elbow Plot
    axes[0].plot(list(K_RANGE), inertias, 'b-o', linewidth=2, markersize=6)
    axes[0].set_xlabel('Anzahl Cluster (k)', fontsize=12)
    axes[0].set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
    axes[0].set_title('Elbow-Methode', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(x=best_k, color='r', linestyle='--', alpha=0.7, label=f'Best Silhouette: k={best_k}')
    axes[0].legend()
    
    # Silhouette Plot
    axes[1].plot(list(K_RANGE), silhouette_scores, 'g-o', linewidth=2, markersize=6)
    axes[1].set_xlabel('Anzahl Cluster (k)', fontsize=12)
    axes[1].set_ylabel('Silhouette Score', fontsize=12)
    axes[1].set_title('Silhouette-Analyse', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(x=best_k, color='r', linestyle='--', alpha=0.7)
    axes[1].axhline(y=best_sil, color='r', linestyle='--', alpha=0.7, label=f'Max: {best_sil:.3f}')
    axes[1].legend()
    
    plt.tight_layout()
    
    plot_path = os.path.join(PLOTS_DIR, 'cluster_analysis.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n  📊 Plot gespeichert: {plot_path}")
    
    # Tabelle ausgeben
    print("\n  📋 Übersicht:")
    print("  " + "-" * 35)
    print(f"  {'k':>4} | {'Inertia':>12} | {'Silhouette':>10}")
    print("  " + "-" * 35)
    for i, k in enumerate(K_RANGE):
        marker = " ← best" if k == best_k else ""
        print(f"  {k:>4} | {inertias[i]:>12.0f} | {silhouette_scores[i]:>10.3f}{marker}")
    
    print("\n" + "=" * 60)
    print(f"EMPFEHLUNG: k = {best_k} (höchster Silhouette Score)")
    print("=" * 60)
    print(f"""
Nächster Schritt:
  python create_topic_clusters.py --cluster {best_k}

Oder wähle ein anderes k basierend auf dem Plot.
""")


# ============================================================
# TF-IDF + DISCRIMINATIVE TERMS
# ============================================================

def extract_keywords_and_discriminative(
    texts: List[str], 
    labels: np.ndarray, 
    n_clusters: int
) -> Tuple[Dict[int, List[str]], Dict[int, List[str]]]:
    """
    Extrahiert zwei Arten von Keywords pro Cluster:
    1. Top TF-IDF Terms: Mittelwert pro Cluster
    2. Discriminative Terms: Cluster-Mittel MINUS Korpus-Mittel
    """
    
    # TF-IDF auf allen Texten
    vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=MIN_DF,
        max_df=MAX_DF,
        stop_words=GERMAN_STOPWORDS,
        ngram_range=(1, 2)
    )
    
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    # Korpus-Mittelwert (über alle Dokumente)
    corpus_mean = tfidf_matrix.mean(axis=0).A1
    
    cluster_keywords = {}
    cluster_discriminative = {}
    
    for cluster_id in range(n_clusters):
        cluster_docs = np.where(labels == cluster_id)[0]
        
        if len(cluster_docs) == 0:
            cluster_keywords[cluster_id] = []
            cluster_discriminative[cluster_id] = []
            continue
        
        # Cluster-Mittelwert
        cluster_mean = tfidf_matrix[cluster_docs].mean(axis=0).A1
        
        # 1. Top TF-IDF Terms (klassisch)
        top_tfidf_idx = cluster_mean.argsort()[-TOP_KEYWORDS:][::-1]
        cluster_keywords[cluster_id] = [feature_names[i] for i in top_tfidf_idx]
        
        # 2. Discriminative Terms (Cluster - Korpus)
        discriminative_scores = cluster_mean - corpus_mean
        top_disc_idx = discriminative_scores.argsort()[-TOP_DISCRIMINATIVE:][::-1]
        # Nur Terms mit positivem Score (überrepräsentiert im Cluster)
        cluster_discriminative[cluster_id] = [
            feature_names[i] for i in top_disc_idx 
            if discriminative_scores[i] > 0
        ]
    
    return cluster_keywords, cluster_discriminative


# ============================================================
# GEMINI LABELS
# ============================================================

def generate_cluster_labels_gemini(
    cluster_keywords: Dict[int, List[str]], 
    cluster_discriminative: Dict[int, List[str]],
    cluster_samples: Dict[int, List[str]],
    model: GenerativeModel
) -> Dict[int, Dict]:
    """
    Generiert menschenlesbare Labels mit Gemini.
    Nutzt sowohl Top-TF-IDF als auch Discriminative Terms.
    """
    
    labels = {}
    
    for cluster_id in tqdm(cluster_keywords.keys(), desc="  Generiere Labels"):
        keywords = cluster_keywords.get(cluster_id, [])
        discriminative = cluster_discriminative.get(cluster_id, [])
        samples = cluster_samples.get(cluster_id, [])
        
        if not keywords and not discriminative:
            labels[cluster_id] = {
                "label": f"Cluster {cluster_id}",
                "description": "Keine Keywords verfügbar"
            }
            continue
        
        prompt = f"""Du bist ein Experte für deutsche Parlamentspolitik und Bundestagsdebatten.

Basierend auf den folgenden Informationen, gib diesem Themen-Cluster einen prägnanten Namen.

**Häufigste Begriffe (TF-IDF):**
{', '.join(keywords[:12])}

**Unterscheidende Begriffe (was diesen Cluster von anderen unterscheidet):**
{', '.join(discriminative[:12])}

**Beispiel-Ausschnitte aus Reden:**
{chr(10).join(['- "' + s[:250] + '..."' for s in samples[:3]])}

Antworte NUR mit JSON, keine Erklärung davor oder danach:
{{"label": "Kurzer Themenname (2-5 Wörter)", "description": "Ein prägnanter Satz, der das Thema beschreibt"}}"""

        try:
            response = model.generate_content(prompt)
            response_text = response.text.strip()
            
            # JSON extrahieren
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            
            result = json.loads(response_text)
            labels[cluster_id] = {
                "label": result.get("label", f"Cluster {cluster_id}"),
                "description": result.get("description", "")
            }
        except Exception as e:
            print(f"\n    ⚠️ Fehler Cluster {cluster_id}: {e}")
            # Fallback: Nutze discriminative term
            fallback = discriminative[0] if discriminative else (keywords[0] if keywords else f"Cluster {cluster_id}")
            labels[cluster_id] = {
                "label": fallback.title(),
                "description": f"Basierend auf: {', '.join((discriminative or keywords)[:5])}"
            }
        
        time.sleep(1)  # Rate limiting
    
    return labels


# ============================================================
# CLUSTERING
# ============================================================

def run_clustering(df: pd.DataFrame, embeddings: np.ndarray, n_clusters: int, client: bigquery.Client):
    """Führt Clustering durch und speichert Ergebnisse."""
    
    print(f"\n  🎯 Clustere in {n_clusters} Themen...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    df['cluster_id'] = kmeans.fit_predict(embeddings)
    
    sil_score = silhouette_score(embeddings, df['cluster_id'])
    print(f"  ✓ Clustering fertig (Silhouette Score: {sil_score:.3f})")
    
    # Cluster-Größen
    print("\n  📊 Cluster-Größen:")
    for cid, size in df['cluster_id'].value_counts().sort_index().items():
        print(f"     Cluster {cid:2d}: {size:4d} Reden ({size/len(df)*100:5.1f}%)")
    
    # Keywords extrahieren
    print("\n" + "=" * 60)
    print("SCHRITT 3: Keyword-Extraktion")
    print("=" * 60)
    
    texts = df['text'].tolist()
    cluster_keywords, cluster_discriminative = extract_keywords_and_discriminative(
        texts, df['cluster_id'].values, n_clusters
    )
    
    print("\n  📝 Keywords pro Cluster:")
    print("  " + "-" * 70)
    for cid in range(n_clusters):
        kw = ', '.join(cluster_keywords[cid][:5])
        disc = ', '.join(cluster_discriminative[cid][:5])
        print(f"  Cluster {cid:2d}:")
        print(f"    TF-IDF:         {kw}")
        print(f"    Discriminative: {disc}")
    
    # Beispiel-Texte
    cluster_samples = {
        cid: df[df['cluster_id'] == cid]['text'].head(5).tolist()
        for cid in range(n_clusters)
    }
    
    # Gemini Labels
    print("\n" + "=" * 60)
    print("SCHRITT 4: Labels mit Gemini generieren")
    print("=" * 60)
    
    vertexai.init(project=PROJECT_ID, location=VERTEX_LOCATION)
    gemini = GenerativeModel("gemini-2.0-flash-001")
    
    cluster_labels = generate_cluster_labels_gemini(
        cluster_keywords, cluster_discriminative, cluster_samples, gemini
    )
    
    print("\n  🏷️  Generierte Labels:")
    for cid, info in sorted(cluster_labels.items()):
        print(f"     {cid:2d}: {info['label']}")
    
    df['cluster_label'] = df['cluster_id'].map(lambda x: cluster_labels[x]['label'])
    
    # Cluster-Statistiken
    print("\n" + "=" * 60)
    print("SCHRITT 5: Statistiken berechnen")
    print("=" * 60)
    
    cluster_stats = []
    for cid in range(n_clusters):
        cluster_df = df[df['cluster_id'] == cid]
        faction_dist = cluster_df['fraktion'].value_counts().to_dict()
        top_factions = dict(sorted(faction_dist.items(), key=lambda x: x[1], reverse=True)[:5])
        
        stats = {
            'cluster_id': cid,
            'label': cluster_labels[cid]['label'],
            'description': cluster_labels[cid]['description'],
            'num_speeches': len(cluster_df),
            'keywords_tfidf': cluster_keywords[cid][:10],
            'keywords_discriminative': cluster_discriminative[cid][:10],
            'avg_sentiment': float(cluster_df['sentiment_score'].mean()),
            'avg_emotionality': float(cluster_df['emotionality_score'].mean()),
            'avg_word_count': float(cluster_df['word_count'].mean()),
            'top_factions': top_factions,
            'wahlperioden': sorted(cluster_df['wahlperiode'].unique().tolist())
        }
        cluster_stats.append(stats)
    
    # Speichern
    print("\n" + "=" * 60)
    print("SCHRITT 6: Speichern")
    print("=" * 60)
    
    # BigQuery: Reden
    speeches_for_bq = df.drop(columns=['embedding'])
    speeches_for_bq['datum'] = pd.to_datetime(speeches_for_bq['datum']).dt.strftime('%Y-%m-%d')
    
    speeches_table = f"{PROJECT_ID}.{DATASET_ID}.{SPEECHES_TARGET}"
    print(f"  ⬆️  {speeches_table}...")
    
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        autodetect=True,
    )
    job = client.load_table_from_dataframe(speeches_for_bq, speeches_table, job_config=job_config)
    job.result()
    print(f"  ✓ {len(speeches_for_bq)} Reden gespeichert")
    
    # BigQuery: Cluster
    cluster_stats_df = pd.DataFrame(cluster_stats)
    cluster_stats_df['keywords_tfidf'] = cluster_stats_df['keywords_tfidf'].apply(json.dumps)
    cluster_stats_df['keywords_discriminative'] = cluster_stats_df['keywords_discriminative'].apply(json.dumps)
    cluster_stats_df['top_factions'] = cluster_stats_df['top_factions'].apply(json.dumps)
    cluster_stats_df['wahlperioden'] = cluster_stats_df['wahlperioden'].apply(json.dumps)
    
    clusters_table = f"{PROJECT_ID}.{DATASET_ID}.{CLUSTERS_TARGET}"
    print(f"  ⬆️  {clusters_table}...")
    
    job = client.load_table_from_dataframe(cluster_stats_df, clusters_table, job_config=job_config)
    job.result()
    print(f"  ✓ {n_clusters} Cluster gespeichert")
    
    # JSON
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    
    dashboard_data = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "n_clusters": n_clusters,
        "total_speeches": len(df),
        "silhouette_score": float(sil_score),
        "clusters": []
    }
    
    for stats in sorted(cluster_stats, key=lambda x: x['num_speeches'], reverse=True):
        dashboard_data["clusters"].append({
            "id": stats['cluster_id'],
            "label": stats['label'],
            "description": stats['description'],
            "num_speeches": stats['num_speeches'],
            "percentage": round(stats['num_speeches'] / len(df) * 100, 1),
            "keywords": {
                "tfidf": stats['keywords_tfidf'],
                "discriminative": stats['keywords_discriminative']
            },
            "metrics": {
                "avg_sentiment": round(stats['avg_sentiment'], 3),
                "avg_emotionality": round(stats['avg_emotionality'], 4),
                "avg_word_count": round(stats['avg_word_count'], 0)
            },
            "top_factions": stats['top_factions']
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dashboard_data, f, ensure_ascii=False, indent=2)
    
    print(f"  ✓ {output_path}")
    
    # Zusammenfassung
    print("\n" + "=" * 60)
    print("✅ CLUSTERING ABGESCHLOSSEN")
    print("=" * 60)
    print(f"""
Ergebnisse:
  • {speeches_table}
  • {clusters_table}
  • {output_path}

Top 5 Cluster:
""")
    for stats in sorted(cluster_stats, key=lambda x: x['num_speeches'], reverse=True)[:5]:
        print(f"  {stats['label']} ({stats['num_speeches']} Reden)")
        print(f"    Discriminative: {', '.join(stats['keywords_discriminative'][:5])}")
        print()


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Themen-Clustering für Bundestagsreden')
    parser.add_argument('--analyze', action='store_true', help='Nur Analyse (Elbow + Silhouette Plots)')
    parser.add_argument('--cluster', type=int, metavar='K', help='Clustering mit k Clustern durchführen')
    
    args = parser.parse_args()
    
    if not args.analyze and not args.cluster:
        parser.print_help()
        print("\n💡 Empfohlener Workflow:")
        print("   1. python create_topic_clusters.py --analyze")
        print("   2. Plot anschauen, k wählen")
        print("   3. python create_topic_clusters.py --cluster K")
        sys.exit(0)
    
    print("""
╔════════════════════════════════════════════════════════════╗
║     Themen-Clustering für Bundestagsreden                 ║
║     K-Means + TF-IDF + Discriminative Terms + Gemini      ║
╚════════════════════════════════════════════════════════════╝
    """)
    
    # Setup
    print("🔌 Setup...")
    client = bigquery.Client(project=PROJECT_ID, location=LOCATION)
    
    # Daten laden
    print("\n" + "=" * 60)
    print("SCHRITT 1: Daten laden")
    print("=" * 60)
    df, embeddings = load_data(client)
    
    if args.analyze:
        run_analysis(embeddings)
    elif args.cluster:
        print("\n" + "=" * 60)
        print(f"SCHRITT 2: K-Means Clustering (k={args.cluster})")
        print("=" * 60)
        run_clustering(df, embeddings, args.cluster, client)


if __name__ == "__main__":
    main()