#!/bin/bash
# ============================================================
# Dashboard Setup Script
# Kopiert JSON-Daten und bereitet für GitHub Pages vor
# ============================================================

echo "🏛️  Bundestag Discourse Tracker - Dashboard Setup"
echo "=================================================="

# Verzeichnisse erstellen
mkdir -p dashboard/data

# JSON-Dateien kopieren
echo "📦 Kopiere Daten..."

if [ -f "dashboard_data/sessions_barometer.json" ]; then
    cp dashboard_data/sessions_barometer.json dashboard/data/
    echo "  ✓ sessions_barometer.json"
else
    echo "  ⚠️  sessions_barometer.json nicht gefunden!"
    echo "     Führe erst create_session_barometer.py aus."
fi

if [ -f "dashboard_data/topic_clusters.json" ]; then
    cp dashboard_data/topic_clusters.json dashboard/data/
    echo "  ✓ topic_clusters.json"
else
    echo "  ⚠️  topic_clusters.json nicht gefunden!"
    echo "     Führe erst create_topic_clusters.py aus."
fi

echo ""
echo "✅ Setup abgeschlossen!"
echo ""
echo "Nächste Schritte:"
echo ""
echo "1. Lokal testen:"
echo "   cd dashboard"
echo "   python -m http.server 8000"
echo "   → http://localhost:8000"
echo ""
echo "2. Auf GitHub deployen:"
echo "   git init"
echo "   git add ."
echo "   git commit -m 'Initial dashboard'"
echo "   git remote add origin https://github.com/DEIN-USER/bundestag-discourse-tracker.git"
echo "   git push -u origin main"
echo ""
echo "3. GitHub Pages aktivieren:"
echo "   → Repository Settings → Pages → Source: main branch"
echo ""