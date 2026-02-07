# 🏛️ Bundestag Discourse Tracker

**Automatisierte Analyse von Bundestagsreden mit KI**

[![Live Demo](https://img.shields.io/badge/Demo-Live-green)](https://DEIN-USER.github.io/bundestag-discourse-tracker)

---

## 📊 Features

### Sitzungs-Barometer
- **Sentiment-Analyse** jeder Sitzung (positiv/negativ/neutral)
- **Perzentil-Vergleich**: "Diese Sitzung war emotionaler als 85% aller Sitzungen"
- **Zeitlicher Verlauf** über Wahlperioden hinweg

### Themen-Clustering
- **Automatische Themenerkennung** aus 3.800+ Reden
- **22 Themen-Cluster** (Klimapolitik, Sozialpolitik, Außenpolitik, ...)
- **Discriminative Keywords**: Was unterscheidet jedes Thema?

---

## 🛠️ Tech Stack

| Komponente | Technologie |
|------------|-------------|
| **Datenquelle** | [Bundestag Open Data](https://www.bundestag.de/services/opendata) (XML) |
| **Datenbank** | Google BigQuery |
| **Embeddings** | Vertex AI `text-embedding-004` |
| **Sentiment** | Gemini Flash |
| **Clustering** | K-Means + TF-IDF |
| **Frontend** | HTML + Tailwind CSS + Chart.js |
| **Hosting** | GitHub Pages |

---

## 📁 Projektstruktur

```
bundestag-discourse-tracker/
├── scripts/
│   ├── load_bundestag_xml_to_bigquery.py   # XML → BigQuery
│   ├── add_metrics_to_speeches.py          # Embeddings + Sentiment
│   ├── create_session_barometer.py         # Sitzungs-Aggregation
│   └── create_topic_clusters.py            # Themen-Clustering
├── dashboard/
│   ├── index.html                          # Dashboard UI
│   └── data/
│       ├── sessions_barometer.json
│       └── topic_clusters.json
└── README.md
```

---

## 🚀 Setup

### 1. GCP Projekt erstellen

```bash
gcloud projects create bt-discourse-tracker
gcloud config set project bt-discourse-tracker
gcloud services enable aiplatform.googleapis.com bigquery.googleapis.com
```

### 2. Daten laden

```bash
# Bundestag XMLs herunterladen (von Open Data Portal)
# In bundestag_xml_data/wp19/, wp20/, wp21/ speichern

python scripts/load_bundestag_xml_to_bigquery.py
```

### 3. Metriken berechnen

```bash
python scripts/add_metrics_to_speeches.py    # ~30 Min, ~$4 API Kosten
```

### 4. Barometer + Clustering

```bash
python scripts/create_session_barometer.py

python scripts/create_topic_clusters.py --analyze   # k wählen
python scripts/create_topic_clusters.py --cluster 22
```

### 5. Dashboard deployen

```bash
chmod +x dashboard/setup.sh
./dashboard/setup.sh

cd dashboard
python -m http.server 8000   # Lokal testen
```

---

## 💰 Kosten

| Service | Kosten |
|---------|--------|
| Vertex AI Embeddings | ~$3 |
| Gemini Flash | ~$1 |
| BigQuery | < $1 |
| GitHub Pages | Kostenlos |
| **Gesamt** | **~$5** |

---

## 📈 Beispiel-Insights

> "Sitzung 21/15 vom 12.01.2026 hatte das negativste Sentiment der letzten 6 Monate (Perzentil 3%)"

> "Thema 'Klimapolitik' hat 12% aller Reden und das höchste durchschnittliche Sentiment (+0.23)"

> "Die AfD-Fraktion hält die emotional intensivsten Reden (Emotionalitäts-Score 0.042)"

---

## 📝 Lizenz

MIT License - Daten: [Bundestag Open Data](https://www.bundestag.de/services/opendata) (CC0)

---

## 🙋 Autor

**[Dein Name]** – Data Scientist / ML Engineer

- Portfolio: [dein-portfolio.com](https://dein-portfolio.com)
- LinkedIn: [linkedin.com/in/dein-profil](https://linkedin.com/in/dein-profil)
- GitHub: [@dein-user](https://github.com/dein-user)