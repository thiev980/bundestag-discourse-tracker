#!/usr/bin/env python3
"""
Sitzungs-Barometer erstellen
============================
Aggregiert Reden zu Sitzungen und berechnet Perzentile für jede Metrik.

Output:
  - BigQuery Tabelle: sessions_barometer
  - JSON für Dashboard: sessions_barometer.json

Metriken pro Sitzung:
  - Anzahl Reden
  - Durchschnittliches Sentiment
  - Durchschnittliche Emotionalität
  - Gesamtdauer (Wörter)
  - Sentiment-Spread (Standardabweichung)
  - Fraktions-Aktivität

Für jede Metrik wird das Perzentil berechnet:
  "Diese Sitzung war emotionaler als X% aller Sitzungen"

Autor: Bundestag Discourse Tracker Project
"""

import os
import json
import pandas as pd
import numpy as np
from google.cloud import bigquery
from datetime import datetime
from typing import Dict, List

# ============================================================
# KONFIGURATION
# ============================================================

PROJECT_ID = "bt-discourse-tracker"
DATASET_ID = "bundestag_data"
SOURCE_TABLE = "speeches_with_metrics"  # Haupt-Tabelle (alle Reden)
CLUSTERS_TABLE = "speeches_with_clusters"  # Für Topic-Aggregation (optional)
TARGET_TABLE = "sessions_barometer"
LOCATION = "EU"

# Output für Dashboard
OUTPUT_DIR = "dashboard_data"
OUTPUT_FILE = "sessions_barometer.json"

# ============================================================
# HILFSFUNKTIONEN
# ============================================================

def calculate_percentile(value: float, all_values: pd.Series) -> float:
    """Berechnet das Perzentil eines Werts in einer Serie."""
    if pd.isna(value) or len(all_values) == 0:
        return 50.0
    return float((all_values < value).sum() / len(all_values) * 100)


def get_percentile_label(percentile: float) -> str:
    """Gibt ein menschenlesbares Label für ein Perzentil."""
    if percentile >= 95:
        return "außergewöhnlich hoch"
    elif percentile >= 80:
        return "überdurchschnittlich"
    elif percentile >= 60:
        return "leicht überdurchschnittlich"
    elif percentile >= 40:
        return "durchschnittlich"
    elif percentile >= 20:
        return "leicht unterdurchschnittlich"
    elif percentile >= 5:
        return "unterdurchschnittlich"
    else:
        return "außergewöhnlich niedrig"


def generate_session_summary(row: pd.Series) -> str:
    """Generiert eine textuelle Zusammenfassung der Sitzung."""
    
    highlights = []
    
    # Sentiment
    if row['sentiment_percentile'] >= 90:
        highlights.append(f"sehr positives Sentiment (Top {100-row['sentiment_percentile']:.0f}%)")
    elif row['sentiment_percentile'] <= 10:
        highlights.append(f"sehr negatives Sentiment (untere {row['sentiment_percentile']:.0f}%)")
    
    # Emotionalität
    if row['emotionality_percentile'] >= 90:
        highlights.append(f"hohe Emotionalität (Top {100-row['emotionality_percentile']:.0f}%)")
    elif row['emotionality_percentile'] <= 10:
        highlights.append(f"sehr sachlich (untere {row['emotionality_percentile']:.0f}%)")
    
    # Länge
    if row['total_words_percentile'] >= 90:
        highlights.append(f"überdurchschnittlich lang (Top {100-row['total_words_percentile']:.0f}%)")
    elif row['total_words_percentile'] <= 10:
        highlights.append(f"kurze Sitzung (untere {row['total_words_percentile']:.0f}%)")
    
    # Kontroverse (hoher Sentiment-Spread)
    if row['sentiment_spread_percentile'] >= 90:
        highlights.append("kontroverse Debatte")
    
    if not highlights:
        return "Durchschnittliche Sitzung"
    
    return ", ".join(highlights).capitalize()


# ============================================================
# HAUPTPROGRAMM
# ============================================================

def main():
    print("""
╔════════════════════════════════════════════════════════════╗
║     Sitzungs-Barometer erstellen                          ║
║     Aggregation + Perzentile + Dashboard-Export           ║
╚════════════════════════════════════════════════════════════╝
    """)
    
    # BigQuery Client
    print("🔌 Verbinde mit BigQuery...")
    client = bigquery.Client(project=PROJECT_ID, location=LOCATION)
    
    # Daten laden
    print("\n" + "=" * 60)
    print("SCHRITT 1: Reden mit Metriken laden")
    print("=" * 60)
    
    query = f"""
    SELECT 
        wahlperiode,
        sitzungsnr,
        datum,
        rede_id,
        fraktion,
        sentiment_score,
        emotionality_score,
        word_count
    FROM `{PROJECT_ID}.{DATASET_ID}.{SOURCE_TABLE}`
    WHERE datum IS NOT NULL
    ORDER BY datum, sitzungsnr
    """
    
    print("  📥 Lade Daten...")
    df = client.query(query).to_dataframe()
    print(f"  ✓ {len(df):,} Reden geladen")
    print(f"  ✓ {df['sitzungsnr'].nunique()} Sitzungen")
    
    # Aggregation pro Sitzung
    print("\n" + "=" * 60)
    print("SCHRITT 2: Aggregation pro Sitzung")
    print("=" * 60)
    
    # Gruppieren nach Sitzung
    sessions = df.groupby(['wahlperiode', 'sitzungsnr', 'datum']).agg({
        'rede_id': 'count',
        'sentiment_score': ['mean', 'std', 'min', 'max'],
        'emotionality_score': ['mean', 'std', 'max'],
        'word_count': ['sum', 'mean'],
    }).reset_index()
    
    # Flatten MultiIndex columns
    sessions.columns = [
        'wahlperiode', 'sitzungsnr', 'datum',
        'num_speeches',
        'avg_sentiment', 'sentiment_spread', 'min_sentiment', 'max_sentiment',
        'avg_emotionality', 'emotionality_spread', 'max_emotionality',
        'total_words', 'avg_words_per_speech',
    ]
    
    # NaN in Spread-Spalten durch 0 ersetzen (wenn nur 1 Rede)
    sessions['sentiment_spread'] = sessions['sentiment_spread'].fillna(0)
    sessions['emotionality_spread'] = sessions['emotionality_spread'].fillna(0)
    
    # Zusätzliche Metriken
    sessions['sentiment_range'] = sessions['max_sentiment'] - sessions['min_sentiment']
    
    print(f"  ✓ {len(sessions)} Sitzungen aggregiert")
    
    # Themen pro Sitzung aggregieren (falls Cluster-Daten vorhanden)
    print("  📊 Berechne Top-Themen pro Sitzung...")
    
    # Versuche Cluster-Daten zu laden
    try:
        cluster_query = f"""
        SELECT wahlperiode, sitzungsnr, cluster_label
        FROM `{PROJECT_ID}.{DATASET_ID}.{CLUSTERS_TABLE}`
        WHERE cluster_label IS NOT NULL
        """
        cluster_df = client.query(cluster_query).to_dataframe()
        has_cluster_data = len(cluster_df) > 0
        print(f"  ✓ {len(cluster_df)} Reden mit Cluster-Zuordnung gefunden")
    except Exception as e:
        print(f"  ⚠️ Cluster-Daten nicht verfügbar: {e}")
        cluster_df = pd.DataFrame()
        has_cluster_data = False
    
    if has_cluster_data:
        topic_counts = cluster_df.groupby(['wahlperiode', 'sitzungsnr', 'cluster_label']).size().reset_index(name='count')
        
        def get_top_topics(wp, snr, n=3):
            """Holt die Top-n Themen für eine Sitzung."""
            session_topics = topic_counts[
                (topic_counts['wahlperiode'] == wp) & 
                (topic_counts['sitzungsnr'] == snr)
            ].nlargest(n, 'count')
            
            return [
                {"label": row['cluster_label'], "count": int(row['count'])}
                for _, row in session_topics.iterrows()
            ]
        
        sessions['top_topics'] = sessions.apply(
            lambda row: get_top_topics(row['wahlperiode'], row['sitzungsnr']), 
            axis=1
        )
        print(f"  ✓ Themen pro Sitzung berechnet")
    else:
        sessions['top_topics'] = [[] for _ in range(len(sessions))]
        print(f"  ⚠️ Keine Themen (Clustering nicht ausgeführt)")
    
    # Fraktions-Aktivität pro Sitzung
    print("  📊 Berechne Fraktions-Aktivität...")
    
    faction_activity = df.groupby(['wahlperiode', 'sitzungsnr', 'fraktion']).agg({
        'rede_id': 'count',
        'sentiment_score': 'mean',
        'word_count': 'sum'
    }).reset_index()
    faction_activity.columns = ['wahlperiode', 'sitzungsnr', 'fraktion', 'speeches', 'avg_sentiment', 'total_words']
    
    # Pivot für Top-Fraktionen pro Sitzung
    def get_top_factions(group):
        top = group.nlargest(3, 'speeches')
        return {
            'top_factions': top['fraktion'].tolist(),
            'top_faction_speeches': top['speeches'].tolist()
        }
    
    # Perzentile berechnen
    print("\n" + "=" * 60)
    print("SCHRITT 3: Perzentile berechnen")
    print("=" * 60)
    
    # Metriken für Perzentil-Berechnung
    metrics_for_percentiles = [
        ('avg_sentiment', 'sentiment_percentile'),
        ('avg_emotionality', 'emotionality_percentile'),
        ('total_words', 'total_words_percentile'),
        ('num_speeches', 'num_speeches_percentile'),
        ('sentiment_spread', 'sentiment_spread_percentile'),
        ('avg_words_per_speech', 'avg_words_percentile'),
    ]
    
    for metric, percentile_col in metrics_for_percentiles:
        sessions[percentile_col] = sessions[metric].apply(
            lambda x: calculate_percentile(x, sessions[metric])
        ).round(1)
        print(f"  ✓ {percentile_col}")
    
    # Labels generieren
    print("\n  📝 Generiere Zusammenfassungen...")
    sessions['summary'] = sessions.apply(generate_session_summary, axis=1)
    
    # Datum formatieren
    sessions['datum_str'] = pd.to_datetime(sessions['datum']).dt.strftime('%d.%m.%Y')
    sessions['jahr'] = pd.to_datetime(sessions['datum']).dt.year
    sessions['monat'] = pd.to_datetime(sessions['datum']).dt.month
    
    # Session-ID für einfacheren Zugriff
    sessions['session_id'] = sessions['wahlperiode'].astype(str) + '_' + sessions['sitzungsnr'].astype(str)
    
    print(f"  ✓ {len(sessions)} Sitzungen mit Perzentilen")
    
    # Statistiken anzeigen
    print("\n" + "=" * 60)
    print("STATISTIKEN")
    print("=" * 60)
    
    print("\n  📊 Sentiment-Verteilung:")
    print(f"     Min: {sessions['avg_sentiment'].min():.3f}")
    print(f"     Max: {sessions['avg_sentiment'].max():.3f}")
    print(f"     Median: {sessions['avg_sentiment'].median():.3f}")
    
    print("\n  📊 Emotionalität-Verteilung:")
    print(f"     Min: {sessions['avg_emotionality'].min():.3f}")
    print(f"     Max: {sessions['avg_emotionality'].max():.3f}")
    print(f"     Median: {sessions['avg_emotionality'].median():.3f}")
    
    print("\n  🔥 Extremste Sitzungen:")
    
    # Negativste Sitzung
    most_negative = sessions.loc[sessions['avg_sentiment'].idxmin()]
    print(f"\n     Negativstes Sentiment:")
    print(f"     → WP {most_negative['wahlperiode']}, Sitzung {most_negative['sitzungsnr']} ({most_negative['datum_str']})")
    print(f"       Score: {most_negative['avg_sentiment']:.3f}")
    
    # Positivste Sitzung
    most_positive = sessions.loc[sessions['avg_sentiment'].idxmax()]
    print(f"\n     Positivstes Sentiment:")
    print(f"     → WP {most_positive['wahlperiode']}, Sitzung {most_positive['sitzungsnr']} ({most_positive['datum_str']})")
    print(f"       Score: {most_positive['avg_sentiment']:.3f}")
    
    # Emotionalste Sitzung
    most_emotional = sessions.loc[sessions['avg_emotionality'].idxmax()]
    print(f"\n     Höchste Emotionalität:")
    print(f"     → WP {most_emotional['wahlperiode']}, Sitzung {most_emotional['sitzungsnr']} ({most_emotional['datum_str']})")
    print(f"       Score: {most_emotional['avg_emotionality']:.3f}")
    
    # Nach BigQuery speichern
    print("\n" + "=" * 60)
    print("SCHRITT 4: Nach BigQuery speichern")
    print("=" * 60)
    
    # Datum-Spalte als String für BigQuery (vermeidet Probleme)
    sessions_for_bq = sessions.copy()
    sessions_for_bq['datum'] = pd.to_datetime(sessions_for_bq['datum']).dt.strftime('%Y-%m-%d')
    sessions_for_bq['top_topics'] = sessions_for_bq['top_topics'].apply(json.dumps)
    
    table_id = f"{PROJECT_ID}.{DATASET_ID}.{TARGET_TABLE}"
    
    print(f"  ⬆️  Speichere nach {table_id}...")
    
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        autodetect=True,
    )
    
    job = client.load_table_from_dataframe(sessions_for_bq, table_id, job_config=job_config)
    job.result()
    
    table = client.get_table(table_id)
    print(f"  ✓ {table.num_rows} Sitzungen gespeichert")
    
    # JSON für Dashboard exportieren
    print("\n" + "=" * 60)
    print("SCHRITT 5: JSON für Dashboard exportieren")
    print("=" * 60)
    
    # Output-Verzeichnis erstellen
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    
    # Daten für Dashboard vorbereiten
    dashboard_data = {
        "generated_at": datetime.now().isoformat(),
        "total_sessions": len(sessions),
        "wahlperioden": sorted(sessions['wahlperiode'].unique().tolist()),
        "date_range": {
            "from": sessions['datum_str'].min(),
            "to": sessions['datum_str'].max()
        },
        "statistics": {
            "avg_sentiment": {
                "min": float(sessions['avg_sentiment'].min()),
                "max": float(sessions['avg_sentiment'].max()),
                "median": float(sessions['avg_sentiment'].median()),
                "mean": float(sessions['avg_sentiment'].mean())
            },
            "avg_emotionality": {
                "min": float(sessions['avg_emotionality'].min()),
                "max": float(sessions['avg_emotionality'].max()),
                "median": float(sessions['avg_emotionality'].median()),
                "mean": float(sessions['avg_emotionality'].mean())
            },
            "num_speeches": {
                "min": int(sessions['num_speeches'].min()),
                "max": int(sessions['num_speeches'].max()),
                "median": float(sessions['num_speeches'].median()),
                "mean": float(sessions['num_speeches'].mean())
            }
        },
        "sessions": []
    }
    
    # Sessions als Liste (sortiert nach Datum, neueste zuerst)
    sessions_sorted = sessions.sort_values('datum', ascending=False)
    
    for _, row in sessions_sorted.iterrows():
        session_data = {
            "session_id": row['session_id'],
            "wahlperiode": int(row['wahlperiode']),
            "sitzungsnr": int(row['sitzungsnr']),
            "datum": row['datum_str'],
            "jahr": int(row['jahr']),
            "num_speeches": int(row['num_speeches']),
            "top_topics": row['top_topics'],
            "metrics": {
                "sentiment": {
                    "value": round(float(row['avg_sentiment']), 3),
                    "percentile": float(row['sentiment_percentile']),
                    "spread": round(float(row['sentiment_spread']), 3)
                },
                "emotionality": {
                    "value": round(float(row['avg_emotionality']), 4),
                    "percentile": float(row['emotionality_percentile'])
                },
                "length": {
                    "total_words": int(row['total_words']),
                    "avg_per_speech": round(float(row['avg_words_per_speech']), 0),
                    "percentile": float(row['total_words_percentile'])
                }
            },
            "summary": row['summary']
        }
        dashboard_data["sessions"].append(session_data)
    
    # JSON speichern
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dashboard_data, f, ensure_ascii=False, indent=2)
    
    print(f"  ✓ {output_path} erstellt ({os.path.getsize(output_path) / 1024:.1f} KB)")
    
    # Zusammenfassung
    print("\n" + "=" * 60)
    print("✅ SITZUNGS-BAROMETER ERSTELLT")
    print("=" * 60)
    print(f"""
BigQuery Tabelle: {table_id}
Dashboard JSON: {output_path}

Metriken pro Sitzung:
  • avg_sentiment + sentiment_percentile
  • avg_emotionality + emotionality_percentile  
  • total_words + total_words_percentile
  • num_speeches + num_speeches_percentile
  • sentiment_spread (Kontroverse)
  • summary (Textuelle Zusammenfassung)

Teste mit dieser Abfrage - Top 5 kontroverseste Sitzungen:

SELECT 
  session_id,
  datum,
  num_speeches,
  ROUND(avg_sentiment, 3) as sentiment,
  ROUND(sentiment_spread, 3) as spread,
  sentiment_spread_percentile as controversy_pct,
  summary
FROM `{table_id}`
ORDER BY sentiment_spread DESC
LIMIT 5

Nächster Schritt:
  → Dashboard mit GitHub Pages erstellen
  → Clustering für Themen-Analyse
""")


if __name__ == "__main__":
    main()