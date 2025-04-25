# Burushaski Words Network: Centrality Analysis

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![NetworkX](https://img.shields.io/badge/NetworkX-2.6+-green)
![Pandas](https://img.shields.io/badge/Pandas-1.3+-yellow)

A **computational linguistics and network science project** focused on exploring **lexical relationships between Burushaski**—a unique language isolate of Northern Pakistan—and over 50 major world languages. Using graph theory and centrality analysis, this research uncovers the structural and lexical connections embedded within multilingual networks.

---

## 📚 Table of Contents

- [🌍 Project Overview](#project-overview)  
- [✨ Key Features](#key-features)  
- [⚙️ Installation](#installation)  
- [🚀 Usage](#usage)  
- [🧠 Methodology](#methodology)  
- [📊 Data Sources](#data-sources)  
- [📈 Visualization Examples](#visualization-examples)  
- [🤝 Contributing](#contributing)  
- [📬 Contact](#contact)  
- [📝 License](#license)  

---

## 🌍 Project Overview

The **Burushaski Words Network** is a bipartite graph model that:

- Connects **Burushaski words** with **50 global languages** based on shared characters, roots, or phonetic elements.
- Uses advanced **centrality metrics** to evaluate:
  - Key **lexical bridges** between Burushaski and other languages.
  - Languages with the **strongest linguistic ties** to Burushaski.
  - Underlying **topological properties** of the multilingual network.

This project blends **linguistic insight** with **network analysis**, creating a valuable resource for researchers in **natural language processing (NLP)**, **comparative linguistics**, and **language preservation**.

---

## ✨ Key Features

### 🔧 Network Construction
- Automated extraction and merging of lexical data.
- Bipartite and projected graph generation using **NetworkX**.
- Fully compatible with **Jupyter/Colab** and **local environments**.

### 📊 Centrality Analysis
- Implements five core centrality measures:
  - **Degree Centrality**
  - **Betweenness Centrality**
  - **Closeness Centrality**
  - **Eigenvector Centrality**
  - **PageRank Centrality**
- Centrality scores are aggregated for global and language-specific insights.
- Enables identification of high-impact words and languages.

### 🖼️ Visualization
- Interactive **network graphs** for structural exploration.
- **Centrality distribution plots** to compare word influence.
- **Heatmaps** to visualize cross-language similarities.
- Custom graph layouts: *spring*, *circular*, *Kamada-Kawai*, etc.

---

## ⚙️ Installation

### ✅ Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### 🛠️ Setup

```bash
# Clone the repository
git clone https://github.com/sardaralikhamosh/Burushaski_Words_Network.git
cd Burushaski_Words_Network

# Create a virtual environment (recommended)
python -m venv venv

# Activate the environment
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

# Install required packages
pip install -r requirements.txt
```

---

## 🚀 Usage

Once installed, simply run the analysis or visualization scripts as per your requirement. Example notebooks and script files are included for easy execution in both Jupyter and Google Colab environments.

---

## 🧠 Methodology

The project uses **weighted bipartite graphs** where:

- **Nodes** represent either Burushaski words or target languages.
- **Edges** indicate shared alphabets or phonetic features.
- Edge **weights** are determined by common characters or linguistic overlap.

Centrality measures are computed on both the full and projected networks to assess word/language importance and connectedness.

---

## 📊 Data Sources

Lexical data is aggregated from:

- Publicly available **language alphabet sets**
- Custom Burushaski word lists
- Ethnologue, Wiktionary, and linguistic corpora (where permitted)

---

## 📈 Visualization Examples

![Network Graph](https://github.com/sardaralikhamosh/pkr-currency-converter/blob/main/burushaski-words-network.png)

Explore word-languages relationships through aesthetically rendered network visualizations and insightful centrality charts.

---

## 🤝 Contributing

Contributions are welcome! If you wish to improve the project, submit pull requests or report issues. Please follow the contribution guidelines in `CONTRIBUTING.md`.

---

## 📬 Contact

For queries, feedback, or collaborations:

**Sardar Ali Khamosh**  
📧 [Email](mailto:sardaralikhamosh@gmail.com)  
🌐 [LinkedIn](https://linkedin.com/in/sardaralikhamosh)  
🐙 [GitHub Profile](https://github.com/sardaralikhamosh)
🌐 [Website](https://digicellinternational.codehuntspk.com/)

---

## 📝 License

This project is open-source and available under the [MIT License](LICENSE).
