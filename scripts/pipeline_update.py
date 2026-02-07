#!/usr/bin/env python3
"""
Bundestag Pipeline - Neue Sitzungen verarbeiten
================================================
Automatisierte Pipeline zum Abrufen und Verarbeiten neuer Plenarprotokolle.

Workflow:
  1. Prüfe welche Sitzungen schon in BigQuery sind
  2. Scrape Open Data Seite für neue Protokolle
  3. Lade neue XMLs herunter
  4. Parse und speichere in BigQuery
  5. Berechne Metriken (Embeddings, Sentiment)
  6. Aktualisiere Clustering
  7. Exportiere Dashboard JSONs

Verwendung:
  python pipeline_update.py              # Vollständiger Update
  python pipeline_update.py --check      # Nur prüfen, nicht verarbeiten
  python pipeline_update.py --dry-run    # Zeigt was passieren würde

Autor: Bundestag Discourse Tracker Project
"""

import os
import sys
import re
import json
import time
import argparse
import tempfile
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from google.cloud import bigquery
import vertexai
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel
from lxml import etree
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# KONFIGURATION
# ============================================================

PROJECT_ID = "bt-discourse-tracker"
DATASET_ID = "bundestag_data"
LOCATION = "EU"
VERTEX_LOCATION = "europe-west1"

# URLs
OPEN_DATA_URL = "https://www.bundestag.de/services/opendata"
BASE_BLOB_URL = "https://www.bundestag.de"

# Tabellen
SPEECHES_RAW_TABLE = "speeches_raw"
SPEECHES_METRICS_TABLE = "speeches_with_metrics"
SPEECHES_CLUSTERS_TABLE = "speeches_with_clusters"
SESSIONS_TABLE = "sessions_barometer"
CLUSTERS_TABLE = "topic_clusters"

# Lokale Pfade
XML_DIR = "bundestag_xml_data"
OUTPUT_DIR = "dashboard_data"

# Rate Limiting
EMBEDDING_BATCH_SIZE = 5
SENTIMENT_BATCH_SIZE = 10
SLEEP_BETWEEN_REQUESTS = 2.0
MAX_RETRIES = 3

# ============================================================
# SCRAPING
# ============================================================

def get_existing_sessions(client: bigquery.Client) -> set:
    """Holt alle bereits verarbeiteten Sitzungen aus BigQuery."""
    
    query = f"""
    SELECT DISTINCT CONCAT(CAST(wahlperiode AS STRING), '_', CAST(sitzungsnr AS STRING)) as session_id
    FROM `{PROJECT_ID}.{DATASET_ID}.{SPEECHES_RAW_TABLE}`
    """
    
    try:
        df = client.query(query).to_dataframe()
        return set(df['session_id'].tolist())
    except Exception as e:
        print(f"  ⚠️ Konnte existierende Sitzungen nicht laden: {e}")
        return set()


def scrape_open_data_page() -> List[Dict]:
    """
    Scraped die Bundestag Open Data Seite für Plenarprotokolle.
    Gibt Liste von {wahlperiode, sitzungsnr, url, title} zurück.
    """
    
    print("  🌐 Scrape Open Data Seite...")
    
    try:
        response = requests.get(OPEN_DATA_URL, timeout=30)
        response.raise_for_status()
    except Exception as e:
        print(f"  ❌ Fehler beim Abrufen: {e}")
        return []
    
    soup = BeautifulSoup(response.text, 'html.parser')
    protocols = []
    
    # Suche nach Protokoll-Links
    # Pattern: /resource/blob/{id}/{wp}{sitzung}.xml
    for link in soup.find_all('a', href=True):
        href = link['href']
        
        if '/resource/blob/' in href and '.xml' in href:
            # Extrahiere WP und Sitzungsnummer aus Dateiname
            match = re.search(r'/(\d{2})(\d{3})\.xml', href)
            if match:
                wp = int(match.group(1))
                sitzung = int(match.group(2))
                
                # Vollständige URL
                full_url = href if href.startswith('http') else BASE_BLOB_URL + href
                
                protocols.append({
                    'wahlperiode': wp,
                    'sitzungsnr': sitzung,
                    'url': full_url,
                    'session_id': f"{wp}_{sitzung}"
                })
    
    print(f"  ✓ {len(protocols)} Protokolle gefunden")
    return protocols


def download_xml(url: str, save_path: Path) -> bool:
    """Lädt eine XML-Datei herunter."""
    
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_bytes(response.content)
        
        return True
    except Exception as e:
        print(f"    ❌ Download fehlgeschlagen: {e}")
        return False


# ============================================================
# XML PARSING (aus load_bundestag_xml_to_bigquery.py)
# ============================================================

def parse_xml_protocol(xml_path: Path) -> List[Dict]:
    """Parst ein Plenarprotokoll-XML und extrahiert Reden."""
    
    try:
        tree = etree.parse(str(xml_path))
        root = tree.getroot()
    except Exception as e:
        print(f"    ⚠️ XML-Parse-Fehler: {e}")
        return []
    
    # Metadaten
    kopfdaten = root.find('.//kopfdaten')
    if kopfdaten is None:
        return []
    
    wahlperiode_elem = kopfdaten.find('.//wahlperiode')
    sitzungsnr_elem = kopfdaten.find('.//sitzungsnr')
    datum_elem = kopfdaten.find('.//datum')
    
    if wahlperiode_elem is None or sitzungsnr_elem is None:
        return []
    
    wahlperiode = int(wahlperiode_elem.text)
    sitzungsnr_text = sitzungsnr_elem.text.strip()
    sitzungsnr = int(re.sub(r'\D', '', sitzungsnr_text))
    
    datum = None
    if datum_elem is not None and datum_elem.get('date'):
        datum = datum_elem.get('date')
    
    # Reden extrahieren
    speeches = []
    for rede in root.findall('.//rede'):
        rede_id = rede.get('id', '')
        
        # Redner
        redner = rede.find('.//redner')
        redner_id = redner.get('id', '') if redner is not None else ''
        
        vorname = ''
        nachname = ''
        fraktion = None
        rolle = None
        
        if redner is not None:
            name_elem = redner.find('.//name')
            if name_elem is not None:
                vorname_elem = name_elem.find('vorname')
                nachname_elem = name_elem.find('nachname')
                fraktion_elem = name_elem.find('fraktion')
                rolle_elem = name_elem.find('rolle_lang')
                
                vorname = vorname_elem.text if vorname_elem is not None and vorname_elem.text else ''
                nachname = nachname_elem.text if nachname_elem is not None and nachname_elem.text else ''
                fraktion = fraktion_elem.text if fraktion_elem is not None and fraktion_elem.text else None
                rolle = rolle_elem.text if rolle_elem is not None and rolle_elem.text else None
        
        # Text zusammenbauen
        text_parts = []
        kommentare = []
        
        for elem in rede.iter():
            if elem.tag == 'p' and elem.text:
                text_parts.append(elem.text.strip())
            elif elem.tag == 'kommentar' and elem.text:
                kommentare.append(elem.text.strip())
        
        text = ' '.join(text_parts)
        
        if text.strip():
            speeches.append({
                'wahlperiode': wahlperiode,
                'sitzungsnr': sitzungsnr,
                'datum': datum,
                'rede_id': rede_id,
                'redner_id': redner_id,
                'vorname': vorname,
                'nachname': nachname,
                'fraktion': fraktion,
                'rolle': rolle,
                'text': text,
                'text_length': len(text),
                'kommentare_count': len(kommentare)
            })
    
    return speeches


# ============================================================
# METRIKEN (Embeddings + Sentiment)
# ============================================================

def calculate_embeddings(texts: List[str], model: TextEmbeddingModel) -> List[List[float]]:
    """Berechnet Embeddings für Texte mit Retry-Logik."""
    
    all_embeddings = []
    
    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[i:i + EMBEDDING_BATCH_SIZE]
        
        # Texte vorbereiten
        prepared = []
        for text in batch:
            if not text:
                text = " "
            if len(text) > 10000:
                text = text[:10000] + "..."
            prepared.append(text)
        
        # Mit Retry
        for attempt in range(MAX_RETRIES):
            try:
                embeddings = model.get_embeddings(prepared)
                all_embeddings.extend([e.values for e in embeddings])
                break
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    wait_time = 5 * (2 ** attempt)
                    print(f"    ⚠️ Retry in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"    ❌ Embedding fehlgeschlagen")
                    all_embeddings.extend([[0.0] * 768 for _ in batch])
        
        time.sleep(SLEEP_BETWEEN_REQUESTS)
    
    return all_embeddings


def calculate_sentiment(texts: List[str], model: GenerativeModel) -> List[Dict]:
    """Berechnet Sentiment für Texte mit Gemini."""
    
    all_sentiments = []
    
    for i in range(0, len(texts), SENTIMENT_BATCH_SIZE):
        batch = texts[i:i + SENTIMENT_BATCH_SIZE]
        
        prompt = """Analysiere das Sentiment der folgenden Bundestagsreden.
Für jede Rede, gib zurück:
- sentiment: "positive", "negative", oder "neutral"
- score: Zahl von -1.0 bis +1.0

Antworte NUR mit einem JSON-Array:
"""
        
        for j, text in enumerate(batch):
            short_text = text[:1500] if text else "(leer)"
            prompt += f"\n--- Rede {j+1} ---\n{short_text}\n"
        
        prompt += '\n[{"sentiment": "...", "score": 0.0}, ...]'
        
        for attempt in range(MAX_RETRIES):
            try:
                response = model.generate_content(prompt)
                response_text = response.text.strip()
                
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0]
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].split("```")[0]
                
                results = json.loads(response_text)
                
                for r in results:
                    all_sentiments.append({
                        "sentiment_label": r.get("sentiment", "neutral"),
                        "sentiment_score": float(r.get("score", 0.0))
                    })
                break
                
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(5)
                else:
                    all_sentiments.extend([
                        {"sentiment_label": "neutral", "sentiment_score": 0.0}
                        for _ in batch
                    ])
        
        time.sleep(SLEEP_BETWEEN_REQUESTS)
    
    return all_sentiments


# ============================================================
# HAUPTPROGRAMM
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Bundestag Pipeline - Neue Sitzungen verarbeiten')
    parser.add_argument('--check', action='store_true', help='Nur prüfen, nicht verarbeiten')
    parser.add_argument('--dry-run', action='store_true', help='Zeigt was passieren würde')
    parser.add_argument('--skip-metrics', action='store_true', help='Überspringe Embeddings/Sentiment')
    parser.add_argument('--url', type=str, help='Manuelle URL eines Protokolls (z.B. https://www.bundestag.de/resource/blob/1140642/21057.xml)')
    parser.add_argument('--delete-session', type=str, help='Lösche eine Sitzung zum Testen (Format: WP_SITZUNG, z.B. 21_57)')
    args = parser.parse_args()
    
    print("""
╔════════════════════════════════════════════════════════════╗
║     Bundestag Pipeline - Neue Sitzungen verarbeiten       ║
╚════════════════════════════════════════════════════════════╝
    """)
    
    # Setup
    print("🔌 Setup...")
    client = bigquery.Client(project=PROJECT_ID, location=LOCATION)
    print("  ✓ BigQuery verbunden")
    
    # Optional: Sitzung löschen zum Testen
    if args.delete_session:
        print("\n" + "=" * 60)
        print("TEST-MODUS: Sitzung löschen")
        print("=" * 60)
        
        wp, sitzung = args.delete_session.split('_')
        
        for table in [SPEECHES_RAW_TABLE, SPEECHES_METRICS_TABLE, SPEECHES_CLUSTERS_TABLE]:
            delete_query = f"""
            DELETE FROM `{PROJECT_ID}.{DATASET_ID}.{table}`
            WHERE wahlperiode = {wp} AND sitzungsnr = {sitzung}
            """
            try:
                client.query(delete_query).result()
                print(f"  ✓ Gelöscht aus {table}")
            except Exception as e:
                print(f"  ⚠️ {table}: {e}")
        
        print(f"\n  Sitzung WP {wp}/{sitzung} gelöscht. Starte Pipeline erneut ohne --delete-session.")
        return
    
    # Schritt 1: Existierende Sitzungen prüfen
    print("\n" + "=" * 60)
    print("SCHRITT 1: Existierende Sitzungen prüfen")
    print("=" * 60)
    
    existing = get_existing_sessions(client)
    print(f"  ✓ {len(existing)} Sitzungen bereits in BigQuery")
    
    # Schritt 2: Neue Protokolle finden
    print("\n" + "=" * 60)
    print("SCHRITT 2: Neue Protokolle suchen")
    print("=" * 60)
    
    # Manueller Modus: URL direkt angegeben
    if args.url:
        print(f"  📎 Manueller Modus: {args.url}")
        
        # Extrahiere WP und Sitzung aus URL
        match = re.search(r'/(\d{2})(\d{3})\.xml', args.url)
        if not match:
            print("  ❌ Konnte WP/Sitzung nicht aus URL extrahieren!")
            print("     Erwartet: .../WPSSS.xml (z.B. 21057.xml)")
            return
        
        wp = int(match.group(1))
        sitzung = int(match.group(2))
        session_id = f"{wp}_{sitzung}"
        
        if session_id in existing:
            print(f"  ⚠️ Sitzung WP {wp}/{sitzung} existiert bereits!")
            print(f"     Nutze --delete-session {session_id} zum Löschen und erneut testen.")
            return
        
        new_protocols = [{
            'wahlperiode': wp,
            'sitzungsnr': sitzung,
            'url': args.url,
            'session_id': session_id
        }]
        
        print(f"  ✓ Verarbeite WP {wp}, Sitzung {sitzung}")
    else:
        # Automatischer Modus: Scrape Open Data
        available = scrape_open_data_page()
        new_protocols = [p for p in available if p['session_id'] not in existing]
    
    if not new_protocols:
        print("\n  ✅ Keine neuen Sitzungen gefunden!")
        return
    
    print(f"\n  🆕 {len(new_protocols)} neue Sitzungen gefunden:")
    for p in new_protocols:
        print(f"     WP {p['wahlperiode']}, Sitzung {p['sitzungsnr']}")
    
    if args.check or args.dry_run:
        print("\n  [Dry-Run/Check Modus - keine Verarbeitung]")
        return
    
    # Schritt 3: XMLs herunterladen
    print("\n" + "=" * 60)
    print("SCHRITT 3: XMLs herunterladen")
    print("=" * 60)
    
    all_speeches = []
    
    for protocol in tqdm(new_protocols, desc="  Download"):
        wp = protocol['wahlperiode']
        sitzung = protocol['sitzungsnr']
        
        xml_path = Path(XML_DIR) / f"wp{wp}" / f"{wp}{sitzung:03d}.xml"
        
        if download_xml(protocol['url'], xml_path):
            speeches = parse_xml_protocol(xml_path)
            all_speeches.extend(speeches)
            print(f"    ✓ WP {wp}/{sitzung}: {len(speeches)} Reden")
        
        time.sleep(1)  # Rate limiting
    
    if not all_speeches:
        print("\n  ⚠️ Keine Reden extrahiert!")
        return
    
    print(f"\n  ✓ {len(all_speeches)} Reden aus {len(new_protocols)} Sitzungen")
    
    # Schritt 4: Nach BigQuery (speeches_raw)
    print("\n" + "=" * 60)
    print("SCHRITT 4: Nach BigQuery speichern")
    print("=" * 60)
    
    df = pd.DataFrame(all_speeches)
    
    # Datum konvertieren: "30.01.2026" → "2026-01-30" (ISO String für JSON)
    def parse_datum(d):
        if not d or d == 'None':
            return None
        try:
            return pd.to_datetime(d, format='%d.%m.%Y').strftime('%Y-%m-%d')
        except:
            try:
                return pd.to_datetime(d).strftime('%Y-%m-%d')
            except:
                return None
    
    df['datum'] = df['datum'].apply(parse_datum)
    
    # None-Werte bereinigen
    df['fraktion'] = df['fraktion'].fillna('')
    df['rolle'] = df['rolle'].fillna('')
    
    print(f"  📋 DataFrame: {len(df)} Zeilen, {len(df.columns)} Spalten")
    print(f"  📋 Datum: {df['datum'].iloc[0]}")
    
    # Via JSON/Newline-Delimited laden (umgeht Parquet-Probleme)
    table_id = f"{PROJECT_ID}.{DATASET_ID}.{SPEECHES_RAW_TABLE}"
    
    # Temporäre JSON-Datei
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        df.to_json(f.name, orient='records', lines=True, date_format='iso')
        temp_path = f.name
    
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
    )
    
    with open(temp_path, 'rb') as f:
        job = client.load_table_from_file(f, table_id, job_config=job_config)
    job.result()
    
    # Cleanup
    os.remove(temp_path)
    
    print(f"  ✓ {len(df)} Reden zu {table_id} hinzugefügt")
    
    if args.skip_metrics:
        print("\n  [Metriken übersprungen]")
        return
    
    # Schritt 5: Metriken berechnen
    print("\n" + "=" * 60)
    print("SCHRITT 5: Metriken berechnen (Embeddings + Sentiment)")
    print("=" * 60)
    
    vertexai.init(project=PROJECT_ID, location=VERTEX_LOCATION)
    embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
    gemini_model = GenerativeModel("gemini-2.0-flash-001")
    
    texts = df['text'].fillna('').tolist()
    
    print(f"\n  🧠 Berechne Embeddings für {len(texts)} Reden...")
    embeddings = calculate_embeddings(texts, embedding_model)
    df['embedding'] = [json.dumps(e) for e in embeddings]
    
    print(f"\n  🎭 Berechne Sentiment für {len(texts)} Reden...")
    sentiments = calculate_sentiment(texts, gemini_model)
    df['sentiment_label'] = [s['sentiment_label'] for s in sentiments]
    df['sentiment_score'] = [s['sentiment_score'] for s in sentiments]
    
    # Emotionalität (einfache Heuristik)
    def calc_emotionality(text):
        if not text:
            return 0.0
        excl = text.count('!')
        text_len = max(len(text), 1)
        return min(1.0, (excl / (text_len / 1000)) / 10)
    
    df['emotionality_score'] = df['text'].apply(calc_emotionality)
    df['word_count'] = df['text'].apply(lambda x: len(x.split()) if x else 0)
    
    # Speichern in speeches_with_metrics (via JSON)
    metrics_table_id = f"{PROJECT_ID}.{DATASET_ID}.{SPEECHES_METRICS_TABLE}"
    
    # Temporäre JSON-Datei
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        df.to_json(f.name, orient='records', lines=True, date_format='iso')
        temp_path = f.name
    
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
    )
    
    with open(temp_path, 'rb') as f:
        job = client.load_table_from_file(f, metrics_table_id, job_config=job_config)
    job.result()
    
    # Cleanup
    os.remove(temp_path)
    
    print(f"  ✓ Metriken zu {metrics_table_id} hinzugefügt")
    
    # Schritt 6: Hinweis für manuelle Schritte
    print("\n" + "=" * 60)
    print("✅ PIPELINE ABGESCHLOSSEN")
    print("=" * 60)
    print(f"""
Verarbeitet: {len(new_protocols)} Sitzungen, {len(all_speeches)} Reden

Für vollständiges Update, führe noch aus:

1. Clustering aktualisieren (optional):
   python scripts/create_topic_clusters.py --cluster 22

2. Barometer aktualisieren:
   python scripts/create_session_barometer.py

3. JSONs ins Dashboard kopieren:
   cp dashboard_data/*.json dashboard/data/

4. Auf GitHub pushen:
   git add . && git commit -m "Update {datetime.now().strftime('%Y-%m-%d')}" && git push
""")


if __name__ == "__main__":
    main()