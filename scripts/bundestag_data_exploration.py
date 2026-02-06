#!/usr/bin/env python3
"""
Bundestag Plenarprotokolle - Data Exploration
==============================================
Dieses Script erkundet die Bundestag Open Data Quellen:
1. DIP API für Metadaten
2. XML-Dateien für strukturierte Reden

Ausführen in Cloud Shell oder lokal:
    pip install requests pandas lxml --break-system-packages
    python bundestag_data_exploration.py
"""

import requests
import json
from datetime import datetime
import xml.etree.ElementTree as ET
from collections import Counter
import os

# ============================================================
# KONFIGURATION
# ============================================================

# Öffentlicher API-Key (gültig bis Mai 2026)
API_KEY = "OSOegLs.PR2lwJ1dwCeje9vTj7FPOt3hvpYKtwKkhw"
BASE_URL = "https://search.dip.bundestag.de/api/v1"

# Header für API-Requests
HEADERS = {
    "Authorization": f"ApiKey {API_KEY}",
    "Accept": "application/json"
}

# ============================================================
# TEIL 1: DIP API - Metadaten erkunden
# ============================================================

def explore_dip_api():
    """Erkundet die DIP API und zeigt verfügbare Protokolle."""
    
    print("=" * 60)
    print("TEIL 1: DIP API - Plenarprotokolle Metadaten")
    print("=" * 60)
    
    # Neueste Plenarprotokolle abrufen (nur Bundestag, nicht Bundesrat)
    params = {
        "f.zuordnung": "BT",  # Nur Bundestag
        "f.wahlperiode": 20,  # 20. Wahlperiode (aktuell)
        "rows": 10            # Nur 10 für Exploration
    }
    
    response = requests.get(
        f"{BASE_URL}/plenarprotokoll",
        headers=HEADERS,
        params=params
    )
    
    if response.status_code != 200:
        print(f"❌ API-Fehler: {response.status_code}")
        print(response.text)
        return None
    
    data = response.json()
    
    print(f"\n📊 Gefundene Protokolle insgesamt: {data.get('numFound', 'N/A')}")
    print(f"📥 Abgerufene Protokolle: {len(data.get('documents', []))}")
    
    print("\n📝 Beispiel-Protokolle:")
    print("-" * 40)
    
    for doc in data.get("documents", [])[:5]:
        print(f"  • {doc.get('dokumentnummer')}: {doc.get('titel', 'N/A')[:60]}...")
        print(f"    Datum: {doc.get('datum')}")
        if doc.get('fundstelle', {}).get('pdf_url'):
            print(f"    PDF: {doc['fundstelle']['pdf_url']}")
        print()
    
    return data


def get_protocol_fulltext(protocol_id: int):
    """Holt den Volltext eines Protokolls via DIP API."""
    
    print(f"\n📄 Lade Volltext für Protokoll-ID {protocol_id}...")
    
    response = requests.get(
        f"{BASE_URL}/plenarprotokoll-text/{protocol_id}",
        headers=HEADERS
    )
    
    if response.status_code != 200:
        print(f"❌ Fehler: {response.status_code}")
        return None
    
    data = response.json()
    text = data.get("text", "")
    
    print(f"✅ Textlänge: {len(text):,} Zeichen")
    print(f"\n📖 Erste 500 Zeichen:\n{'-' * 40}")
    print(text[:500] if text else "Kein Text verfügbar")
    
    return data


# ============================================================
# TEIL 2: XML Open Data - Strukturierte Reden
# ============================================================

def download_sample_xml():
    """Lädt ein Beispiel-XML Protokoll herunter."""
    
    print("\n" + "=" * 60)
    print("TEIL 2: XML Open Data - Strukturierte Reden")
    print("=" * 60)
    
    # Beispiel: Protokoll 20/150 (Sitzung 150 der 20. Wahlperiode)
    # URL-Pattern für XML: https://www.bundestag.de/resource/blob/{blob_id}/...
    # Alternativ direkt von der Open Data Seite
    
    # Wir nutzen einen bekannten Download-Link für ein neueres Protokoll
    # Die XML-Dateien sind auf der Open Data Seite verlinkt
    
    # Für dieses Beispiel: DIP API gibt uns die Dokumentnummer,
    # wir konstruieren die XML-URL
    xml_url = "https://www.bundestag.de/resource/blob/992142/20200.xml"
    
    print(f"\n📥 Lade XML von: {xml_url}")
    
    try:
        response = requests.get(xml_url, timeout=30)
        if response.status_code == 200:
            # Speichern für lokale Analyse
            with open("sample_protocol.xml", "wb") as f:
                f.write(response.content)
            print(f"✅ Gespeichert als 'sample_protocol.xml' ({len(response.content):,} bytes)")
            return response.content
        else:
            print(f"⚠️ Download fehlgeschlagen: {response.status_code}")
            print("   Versuche alternatives Protokoll...")
            return None
    except Exception as e:
        print(f"❌ Fehler beim Download: {e}")
        return None


def parse_xml_protocol(xml_content):
    """Parst ein XML-Protokoll und extrahiert Reden."""
    
    if xml_content is None:
        print("\n⚠️ Kein XML-Content zum Parsen")
        return
    
    print("\n📊 Parsing XML-Struktur...")
    print("-" * 40)
    
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        print(f"❌ XML Parse Error: {e}")
        return
    
    # Namespace handling (Bundestag XMLs haben manchmal Namespaces)
    # Die DTD zeigt: dbtplenarprotokoll ist das Root-Element
    
    # Kopfdaten extrahieren
    kopfdaten = root.find(".//kopfdaten")
    if kopfdaten is not None:
        wahlperiode = kopfdaten.findtext(".//wahlperiode", "N/A")
        sitzungsnr = kopfdaten.findtext(".//sitzungsnr", "N/A")
        datum = kopfdaten.findtext(".//datum", "N/A")
        print(f"📅 Wahlperiode: {wahlperiode}, Sitzung: {sitzungsnr}, Datum: {datum}")
    
    # Reden finden
    # Die XML-Struktur hat <rede> Elemente mit Attributen
    reden = root.findall(".//rede")
    print(f"\n🎤 Anzahl Reden im Protokoll: {len(reden)}")
    
    if not reden:
        # Alternative: Versuche andere Struktur
        print("   Versuche alternative Struktur...")
        reden = root.findall(".//redner")
        print(f"   Gefundene Redner-Elemente: {len(reden)}")
    
    # Erste Reden analysieren
    print("\n📝 Beispiel-Reden:")
    print("-" * 40)
    
    speeches = []
    
    for i, rede in enumerate(reden[:5]):
        # Redner-Info extrahieren
        redner = rede.find(".//redner")
        if redner is not None:
            redner_id = redner.get("id", "N/A")
            vorname = redner.findtext(".//vorname", "")
            nachname = redner.findtext(".//nachname", "")
            fraktion = redner.findtext(".//fraktion", "N/A")
            rolle = redner.findtext(".//rolle_lang", "")
            
            # Text der Rede (alle <p> Elemente)
            paragraphs = rede.findall(".//p")
            text = " ".join([p.text or "" for p in paragraphs if p.text])
            
            speech_info = {
                "redner_id": redner_id,
                "name": f"{vorname} {nachname}".strip(),
                "fraktion": fraktion,
                "rolle": rolle,
                "text_length": len(text),
                "text_preview": text[:200] if text else "N/A"
            }
            speeches.append(speech_info)
            
            print(f"\n  Rede {i+1}:")
            print(f"    Redner: {speech_info['name']}")
            print(f"    Fraktion: {fraktion}")
            print(f"    Rolle: {rolle}")
            print(f"    Textlänge: {speech_info['text_length']:,} Zeichen")
            print(f"    Vorschau: {speech_info['text_preview'][:100]}...")
    
    # Fraktionsverteilung
    if speeches:
        print("\n📊 Fraktionsverteilung (erste 5 Reden):")
        fraktionen = Counter([s["fraktion"] for s in speeches])
        for fraktion, count in fraktionen.most_common():
            print(f"    {fraktion}: {count}")
    
    return speeches


def explore_xml_structure(xml_content):
    """Zeigt die XML-Struktur für besseres Verständnis."""
    
    if xml_content is None:
        return
    
    print("\n" + "=" * 60)
    print("XML-STRUKTUR ÜBERSICHT")
    print("=" * 60)
    
    root = ET.fromstring(xml_content)
    
    def show_structure(element, indent=0):
        """Rekursive Funktion zur Strukturanzeige."""
        prefix = "  " * indent
        attribs = " ".join([f'{k}="{v}"' for k, v in element.attrib.items()][:2])
        if attribs:
            attribs = f" ({attribs})"
        print(f"{prefix}<{element.tag}>{attribs}")
        
        # Nur erste 3 Kinder pro Level zeigen
        children = list(element)[:3]
        for child in children:
            if indent < 4:  # Max depth
                show_structure(child, indent + 1)
        
        if len(list(element)) > 3:
            print(f"{prefix}  ... ({len(list(element)) - 3} weitere Elemente)")
    
    show_structure(root)


# ============================================================
# TEIL 3: Statistiken und Erkenntnisse
# ============================================================

def analyze_api_coverage():
    """Analysiert wie viele Protokolle pro Wahlperiode verfügbar sind."""
    
    print("\n" + "=" * 60)
    print("TEIL 3: Datenabdeckung pro Wahlperiode")
    print("=" * 60)
    
    wahlperioden = [19, 20, 21]  # Neuere Perioden mit XML-Support
    
    for wp in wahlperioden:
        params = {
            "f.zuordnung": "BT",
            "f.wahlperiode": wp,
            "rows": 1  # Nur Anzahl interessiert uns
        }
        
        response = requests.get(
            f"{BASE_URL}/plenarprotokoll",
            headers=HEADERS,
            params=params
        )
        
        if response.status_code == 200:
            data = response.json()
            count = data.get("numFound", 0)
            print(f"  WP {wp}: {count:>4} Plenarprotokolle")
        else:
            print(f"  WP {wp}: ❌ Fehler")


# ============================================================
# HAUPTPROGRAMM
# ============================================================

def main():
    print("""
╔════════════════════════════════════════════════════════════╗
║     Bundestag Plenarprotokolle - Data Exploration         ║
║     Für: Bundestag Discourse Tracker Projekt              ║
╚════════════════════════════════════════════════════════════╝
    """)
    
    # Teil 1: API erkunden
    api_data = explore_dip_api()
    
    # Optional: Volltext eines Protokolls laden
    # Auskommentiert um API-Requests zu sparen
    # if api_data and api_data.get("documents"):
    #     first_id = api_data["documents"][0].get("id")
    #     get_protocol_fulltext(first_id)
    
    # Teil 2: XML-Daten erkunden
    xml_content = download_sample_xml()
    if xml_content:
        explore_xml_structure(xml_content)
        parse_xml_protocol(xml_content)
    
    # Teil 3: Abdeckung analysieren
    analyze_api_coverage()
    
    # Zusammenfassung
    print("\n" + "=" * 60)
    print("📋 ZUSAMMENFASSUNG FÜR PIPELINE-DESIGN")
    print("=" * 60)
    print("""
    Datenquellen:
    ├── DIP API: Metadaten, Volltexte (unstrukturiert)
    └── XML Open Data: Strukturierte Reden mit Redner-Info
    
    Empfohlener Ansatz:
    1. DIP API nutzen um neue Protokolle zu finden (täglich)
    2. XML-Version herunterladen für strukturierte Extraktion
    3. Reden einzeln in BigQuery speichern mit:
       - redner_id, name, fraktion
       - datum, wahlperiode, sitzungsnr
       - volltext, text_length
    
    Nächste Schritte:
    1. Cloud Function für täglichen Scraper bauen
    2. BigQuery Schema definieren
    3. Embedding-Pipeline aufsetzen
    """)


if __name__ == "__main__":
    main()