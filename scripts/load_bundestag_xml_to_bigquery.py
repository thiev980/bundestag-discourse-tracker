#!/usr/bin/env python3
"""
Bundestag Open Data XML → BigQuery Loader
==========================================
Dieses Script lädt Plenarprotokolle direkt von der Bundestag Open Data Seite,
extrahiert die einzelnen Reden und lädt sie nach BigQuery.

Voraussetzungen:
    pip install pandas requests lxml tqdm google-cloud-bigquery db-dtypes

Ausführen:
    python load_bundestag_xml_to_bigquery.py

Autor: Bundestag Discourse Tracker Project
"""

import os
import re
import sys
import json
import requests
import pandas as pd
from lxml import etree
from datetime import datetime
from tqdm import tqdm
from typing import List, Dict, Optional
from google.cloud import bigquery

# ============================================================
# KONFIGURATION
# ============================================================

PROJECT_ID = "bt-discourse-tracker"  # ← Deine Projekt-ID
DATASET_ID = "bundestag_data"
LOCATION = "EU"

# Wahlperioden zum Laden (19, 20, 21 haben strukturierte XMLs)
WAHLPERIODEN = [19, 20, 21]

# Lokaler Download-Ordner
DATA_DIR = "./bundestag_xml_data"

# DIP API für Protokoll-Metadaten
DIP_API_KEY = "OSOegLs.PR2lwJ1dwCeje9vTj7FPOt3hvpYKtwKkhw"
DIP_BASE_URL = "https://search.dip.bundestag.de/api/v1"

# ============================================================
# TEIL 1: PROTOKOLL-URLS SAMMELN
# ============================================================

def get_protocol_metadata(wahlperiode: int) -> List[Dict]:
    """Holt Metadaten aller Protokolle einer Wahlperiode via DIP API."""
    
    print(f"  📋 Lade Metadaten für WP {wahlperiode}...")
    
    all_protocols = []
    cursor = None
    
    headers = {"Authorization": f"ApiKey {DIP_API_KEY}"}
    
    while True:
        params = {
            "f.wahlperiode": wahlperiode,
            "f.zuordnung": "BT",  # Nur Bundestag, nicht Bundesrat
            "rows": 100,
        }
        if cursor:
            params["cursor"] = cursor
        
        response = requests.get(
            f"{DIP_BASE_URL}/plenarprotokoll",
            headers=headers,
            params=params
        )
        
        if response.status_code != 200:
            print(f"    ⚠️ API-Fehler: {response.status_code}")
            break
        
        data = response.json()
        documents = data.get("documents", [])
        
        if not documents:
            break
        
        all_protocols.extend(documents)
        cursor = data.get("cursor")
        
        if not cursor:
            break
    
    print(f"    ✓ {len(all_protocols)} Protokolle gefunden")
    return all_protocols


def get_xml_url_for_protocol(doc: Dict) -> Optional[str]:
    """Extrahiert die XML-URL aus den Protokoll-Metadaten."""
    
    fundstelle = doc.get("fundstelle", {})
    
    # Versuche XML-URL zu finden
    xml_url = fundstelle.get("xml_url")
    if xml_url:
        return xml_url
    
    # Alternativ: Konstruiere URL aus Dokumentnummer
    # Format: https://www.bundestag.de/resource/blob/XXXXX/YYYYY.xml
    # Das ist leider nicht vorhersagbar...
    
    # Fallback: PDF-URL zu XML umwandeln (funktioniert manchmal)
    pdf_url = fundstelle.get("pdf_url", "")
    if "dserver.bundestag.de/btp" in pdf_url:
        # z.B. https://dserver.bundestag.de/btp/20/20214.pdf
        # → XML ist auf bundestag.de/services/opendata
        pass
    
    return None


# ============================================================
# TEIL 2: XML HERUNTERLADEN
# ============================================================

def download_xml(url: str, filepath: str) -> bool:
    """Lädt eine XML-Datei herunter."""
    
    if os.path.exists(filepath):
        return True
    
    try:
        response = requests.get(url, timeout=60)
        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                f.write(response.content)
            return True
        else:
            return False
    except Exception as e:
        print(f"    ⚠️ Download-Fehler: {e}")
        return False


def download_from_open_data_page(wahlperiode: int) -> List[str]:
    """
    Lädt XMLs von der Bundestag Open Data Seite.
    Da die URLs nicht programmatisch zugänglich sind, 
    müssen wir sie manuell oder via Scraping holen.
    """
    
    # Die Open Data Seite hat keine API - wir müssen die Seite parsen
    # oder die XMLs manuell herunterladen
    
    print(f"\n  🌐 Prüfe Open Data Seite für WP {wahlperiode}...")
    
    # Versuch 1: DIP API mit xml_url Feld
    protocols = get_protocol_metadata(wahlperiode)
    
    xml_files = []
    wp_dir = os.path.join(DATA_DIR, f"wp{wahlperiode}")
    os.makedirs(wp_dir, exist_ok=True)
    
    for doc in tqdm(protocols, desc=f"WP {wahlperiode}"):
        doc_nr = doc.get("dokumentnummer", "unknown")
        filename = f"{doc_nr.replace('/', '_')}.xml"
        filepath = os.path.join(wp_dir, filename)
        
        # Prüfe ob schon vorhanden
        if os.path.exists(filepath):
            xml_files.append(filepath)
            continue
        
        # Versuche XML-URL aus API
        xml_url = get_xml_url_for_protocol(doc)
        if xml_url and download_xml(xml_url, filepath):
            xml_files.append(filepath)
    
    return xml_files


# ============================================================
# TEIL 3: XML PARSEN → REDEN EXTRAHIEREN
# ============================================================

def parse_protocol_xml(filepath: str) -> List[Dict]:
    """
    Parst ein Plenarprotokoll-XML und extrahiert alle Reden.
    
    Struktur (ab WP 19):
    <dbtplenarprotokoll>
      <vorspann>...</vorspann>
      <sitzungsverlauf>
        <sitzungsbeginn>...</sitzungsbeginn>
        <tagesordnungspunkt>
          <rede id="...">
            <p klasse="redner">
              <redner id="...">
                <name>
                  <vorname>...</vorname>
                  <nachname>...</nachname>
                  <fraktion>...</fraktion>
                </name>
              </redner>
            </p>
            <p klasse="J">Text der Rede...</p>
            <kommentar>Beifall bei der SPD</kommentar>
          </rede>
        </tagesordnungspunkt>
      </sitzungsverlauf>
    </dbtplenarprotokoll>
    """
    
    speeches = []
    
    try:
        tree = etree.parse(filepath)
        root = tree.getroot()
    except Exception as e:
        print(f"    ⚠️ Parse-Fehler in {filepath}: {e}")
        return speeches
    
    # Metadaten aus Kopfdaten
    wahlperiode = None
    sitzungsnr = None
    datum = None
    
    kopfdaten = root.find(".//kopfdaten")
    if kopfdaten is not None:
        wp_elem = kopfdaten.find(".//wahlperiode")
        if wp_elem is not None and wp_elem.text:
            # Extrahiere nur die Zahl
            wp_text = wp_elem.text.strip()
            wp_match = re.match(r'(\d+)', wp_text)
            if wp_match:
                wahlperiode = int(wp_match.group(1))
        
        sitz_elem = kopfdaten.find(".//sitzungsnr")
        if sitz_elem is not None and sitz_elem.text:
            # Extrahiere nur die Zahl (z.B. "48 (neu)" → 48)
            sitz_text = sitz_elem.text.strip()
            sitz_match = re.match(r'(\d+)', sitz_text)
            if sitz_match:
                sitzungsnr = int(sitz_match.group(1))
        
        datum_elem = kopfdaten.find(".//datum")
        if datum_elem is not None:
            datum_str = datum_elem.get("date", datum_elem.text)
            if datum_str:
                # Format: "30.01.2026" oder "2026-01-30"
                try:
                    if "." in datum_str:
                        datum = datetime.strptime(datum_str, "%d.%m.%Y").date()
                    else:
                        datum = datetime.strptime(datum_str, "%Y-%m-%d").date()
                except:
                    pass
    
    # Alle Reden finden
    for rede in root.findall(".//rede"):
        speech = {
            "wahlperiode": wahlperiode,
            "sitzungsnr": sitzungsnr,
            "datum": datum,
            "rede_id": rede.get("id"),
            "redner_id": None,
            "vorname": None,
            "nachname": None,
            "fraktion": None,
            "rolle": None,
            "text": "",
            "text_length": 0,
            "kommentare_count": 0,
        }
        
        # Redner-Informationen
        redner = rede.find(".//redner")
        if redner is not None:
            speech["redner_id"] = redner.get("id")
            
            name = redner.find(".//name")
            if name is not None:
                vn = name.find("vorname")
                nn = name.find("nachname")
                fr = name.find("fraktion")
                rolle = name.find("rolle_lang")
                
                if vn is not None and vn.text:
                    speech["vorname"] = vn.text.strip()
                if nn is not None and nn.text:
                    speech["nachname"] = nn.text.strip()
                if fr is not None and fr.text:
                    speech["fraktion"] = fr.text.strip()
                if rolle is not None and rolle.text:
                    speech["rolle"] = rolle.text.strip()
        
        # Redetext sammeln (alle <p> Elemente, aber nicht Kommentare)
        text_parts = []
        for p in rede.findall(".//p"):
            # Überspringe Redner-Absätze
            klasse = p.get("klasse", "")
            if klasse == "redner":
                continue
            
            # Text extrahieren (inkl. verschachtelter Elemente)
            text = etree.tostring(p, method="text", encoding="unicode")
            if text:
                text_parts.append(text.strip())
        
        speech["text"] = " ".join(text_parts)
        speech["text_length"] = len(speech["text"])
        
        # Kommentare zählen (Beifall, Zwischenrufe, etc.)
        kommentare = rede.findall(".//kommentar")
        speech["kommentare_count"] = len(kommentare)
        
        # Nur Reden mit Text hinzufügen
        if speech["text_length"] > 50:  # Mindestlänge
            speeches.append(speech)
    
    return speeches


# ============================================================
# TEIL 4: NACH BIGQUERY LADEN
# ============================================================

def upload_speeches_to_bigquery(speeches: List[Dict], client: bigquery.Client):
    """Lädt die extrahierten Reden nach BigQuery."""
    
    table_id = f"{PROJECT_ID}.{DATASET_ID}.speeches_raw"
    
    print(f"\n  ⬆️  Lade {len(speeches):,} Reden nach BigQuery...")
    
    # DataFrame erstellen
    df = pd.DataFrame(speeches)
    
    # Datum-Spalte konvertieren
    if "datum" in df.columns:
        df["datum"] = pd.to_datetime(df["datum"])
    
    # Schema definieren
    schema = [
        bigquery.SchemaField("wahlperiode", "INTEGER"),
        bigquery.SchemaField("sitzungsnr", "INTEGER"),
        bigquery.SchemaField("datum", "DATE"),
        bigquery.SchemaField("rede_id", "STRING"),
        bigquery.SchemaField("redner_id", "STRING"),
        bigquery.SchemaField("vorname", "STRING"),
        bigquery.SchemaField("nachname", "STRING"),
        bigquery.SchemaField("fraktion", "STRING"),
        bigquery.SchemaField("rolle", "STRING"),
        bigquery.SchemaField("text", "STRING"),
        bigquery.SchemaField("text_length", "INTEGER"),
        bigquery.SchemaField("kommentare_count", "INTEGER"),
    ]
    
    job_config = bigquery.LoadJobConfig(
        schema=schema,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )
    
    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()
    
    table = client.get_table(table_id)
    print(f"  ✓ {table.num_rows:,} Reden in {table_id}")


# ============================================================
# HAUPTPROGRAMM
# ============================================================

def main():
    print("""
╔════════════════════════════════════════════════════════════╗
║     Bundestag Open Data XML → BigQuery Loader             ║
║     Bundestag Discourse Tracker Project                    ║
╚════════════════════════════════════════════════════════════╝
    """)
    
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Schritt 1: Prüfe ob XMLs lokal vorhanden
    print("=" * 60)
    print("SCHRITT 1: Lokale XML-Dateien prüfen")
    print("=" * 60)
    
    local_xml_files = []
    for wp in WAHLPERIODEN:
        wp_dir = os.path.join(DATA_DIR, f"wp{wp}")
        if os.path.exists(wp_dir):
            files = [os.path.join(wp_dir, f) for f in os.listdir(wp_dir) if f.endswith('.xml')]
            print(f"  WP {wp}: {len(files)} XML-Dateien gefunden")
            local_xml_files.extend(files)
        else:
            print(f"  WP {wp}: Keine Dateien (Ordner {wp_dir} existiert nicht)")
    
    # Wenn keine lokalen Dateien, Anleitung zum manuellen Download
    if not local_xml_files:
        print(f"""
╔════════════════════════════════════════════════════════════╗
║  📥 MANUELLER DOWNLOAD ERFORDERLICH                        ║
╠════════════════════════════════════════════════════════════╣
║                                                            ║
║  Die Bundestag XMLs müssen manuell heruntergeladen werden. ║
║                                                            ║
║  1. Öffne: https://www.bundestag.de/services/opendata      ║
║                                                            ║
║  2. Scrolle zu "Plenarprotokolle der 19/20/21. Wahlperiode"║
║                                                            ║
║  3. Lade alle XML-Dateien herunter                         ║
║     (Klicke auf jede XML-Datei zum Download)               ║
║                                                            ║
║  4. Speichere sie in diese Ordner:                         ║
║     {DATA_DIR}/wp19/                                        ║
║     {DATA_DIR}/wp20/                                        ║
║     {DATA_DIR}/wp21/                                        ║
║                                                            ║
║  5. Führe dieses Script erneut aus                         ║
║                                                            ║
║  TIPP: Du kannst auch nur WP 20 + 21 herunterladen         ║
║        für einen schnelleren Start (~270 Dateien)          ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
""")
        
        # Trotzdem BigQuery-Verbindung testen
        print("\n🔌 Teste BigQuery-Verbindung...")
        try:
            client = bigquery.Client(project=PROJECT_ID, location=LOCATION)
            print(f"   ✓ Verbindung OK: {PROJECT_ID}")
        except Exception as e:
            print(f"   ⚠️ Verbindungsfehler: {e}")
        
        print("\n→ Bitte lade die XMLs herunter und führe das Script erneut aus.")
        sys.exit(0)
    
    # Schritt 2: BigQuery verbinden
    print("\n" + "=" * 60)
    print("SCHRITT 2: BigQuery Verbindung")
    print("=" * 60)
    
    try:
        client = bigquery.Client(project=PROJECT_ID, location=LOCATION)
        print(f"  ✓ Verbunden mit {PROJECT_ID}")
    except Exception as e:
        print(f"  ❌ Fehler: {e}")
        print("  → gcloud auth application-default login")
        sys.exit(1)
    
    # Schritt 3: XMLs parsen
    print("\n" + "=" * 60)
    print("SCHRITT 3: XML-Dateien parsen")
    print("=" * 60)
    
    all_speeches = []
    
    for filepath in tqdm(local_xml_files, desc="Parsing"):
        speeches = parse_protocol_xml(filepath)
        all_speeches.extend(speeches)
    
    print(f"\n  ✓ {len(all_speeches):,} Reden extrahiert")
    
    # Statistiken
    df_temp = pd.DataFrame(all_speeches)
    if len(df_temp) > 0:
        print(f"\n  📊 Statistiken:")
        print(f"     Wahlperioden: {sorted(df_temp['wahlperiode'].dropna().unique().tolist())}")
        print(f"     Fraktionen: {df_temp['fraktion'].dropna().nunique()} verschiedene")
        print(f"     Redner: {df_temp['redner_id'].dropna().nunique()} verschiedene")
        print(f"     Durchschn. Textlänge: {df_temp['text_length'].mean():,.0f} Zeichen")
    
    # Schritt 4: Nach BigQuery laden
    print("\n" + "=" * 60)
    print("SCHRITT 4: Nach BigQuery laden")
    print("=" * 60)
    
    upload_speeches_to_bigquery(all_speeches, client)
    
    # Zusammenfassung
    print("\n" + "=" * 60)
    print("✅ IMPORT ABGESCHLOSSEN")
    print("=" * 60)
    print(f"""
Tabelle: {PROJECT_ID}.{DATASET_ID}.speeches_raw
Reden: {len(all_speeches):,}

Teste mit dieser Abfrage:

SELECT 
  fraktion,
  COUNT(*) as anzahl_reden,
  AVG(text_length) as avg_laenge,
  AVG(kommentare_count) as avg_kommentare
FROM `{PROJECT_ID}.{DATASET_ID}.speeches_raw`
WHERE wahlperiode = 21
GROUP BY fraktion
ORDER BY anzahl_reden DESC

Nächste Schritte:
1. Weitere Metriken hinzufügen (Sentiment, Embeddings)
2. Aggregations-Views erstellen
3. Pipeline für neue Protokolle aufsetzen
""")


if __name__ == "__main__":
    main()