# 🔬 Projet Clustering — M1 Informatique

> **Séance 7 évaluée — Analyse de clustering sur un jeu de données de grande dimension**  
> Méthodologie : CRISP-DM · Algorithme : K-Means++ · Langage : Python

---

## 🖥️ Web App — Démonstration

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

Lancer l'application web :
```bash
streamlit run streamlit_app.py
```

---

## 📁 Structure du projet

```
projet_clustering/
│
├── 📓 clustering_projet.ipynb        ← Notebook principal (toute l'analyse CRISP-DM)
├── 🌐 streamlit_app.py               ← Web App de déploiement (Streamlit)
├── 📊 dataset_clustering_td7.xlsx    ← Jeu de données (à placer ici — non versionné)
├── 📊 resultats_clustering.xlsx      ← Export des résultats (généré par le notebook)
├── 📄 requirements.txt               ← Dépendances Python
├── 🚫 .gitignore                     ← Fichiers exclus de Git
│
├── 📁 modele/                        ← Artefacts du modèle (générés par le notebook)
│   ├── scaler.pkl                    → StandardScaler ajusté
│   ├── pca.pkl                       → Modèle ACP ajusté
│   ├── kmeans_model.pkl              → Modèle K-Means final
│   ├── variables_pertinentes.pkl     → Variables sélectionnées
│   ├── colonnes.pkl                  → Ordre des colonnes d'entraînement
│   ├── profil_zscore.pkl             → Profil d'interprétation des clusters
│   ├── k_optimal.pkl                 → Nombre de clusters retenu
│   └── seuil_zscore.pkl              → Seuil de caractérisation
│
└── 📁 figures/                       ← Graphiques exportés (générés par le notebook)
    ├── distributions.png
    ├── acp_variance.png
    ├── criteres_k.png
    ├── silhouette_plot.png
    ├── clusters_pca.png
    ├── correlation_heatmap.png
    └── profil_clusters_heatmap.png
```

---

## ⚙️ Installation

### Prérequis
- Python **3.9+**

### Installation des dépendances

```bash
pip install -r requirements.txt
```

---

## 🚀 Guide d'utilisation

### Étape 1 — Placer le dataset

```
projet_clustering/
├── dataset_clustering_td7.xlsx   ← ici
└── ...
```

### Étape 2 — Exécuter le notebook

```bash
jupyter notebook clustering_projet.ipynb
```

Exécutez **toutes les cellules dans l'ordre** (*Kernel → Restart & Run All*).  
Le notebook génère automatiquement le dossier `modele/`.

### Étape 3a — Lancer la Web App (Streamlit)

```bash
streamlit run streamlit_app.py
```

L'application s'ouvre dans votre navigateur à `http://localhost:8501`.

### Étape 3b — Lancer l'application console (alternative)

```bash
python app_deploiement.py
```

---

## 🌐 Fonctionnalités de la Web App

| Onglet | Contenu |
|--------|---------|
| 📊 **Résultat** | Cluster prédit, distances aux centroïdes, variables caractéristiques |
| 📈 **Profil des clusters** | Heatmap des z-scores, tableau interactif |
| ℹ️ **À propos** | Pipeline technique, artefacts, méthodes |

**Fonctionnalités principales :**
- Saisie interactive de toutes les variables pertinentes (sidebar)
- Regroupement des variables par blocs de 10 pour la lisibilité
- Graphique des distances aux centroïdes
- Badges colorés pour les variables élevées / faibles
- Heatmap du profil des clusters avec sélection du nombre de variables
- Reset des valeurs en un clic
- Modèle chargé **une seule fois** via `@st.cache_resource` (aucun réentraînement)

---

## 📓 Description du notebook (CRISP-DM)

| Cellule | Titre | Description |
|---------|-------|-------------|
| 1 | Importation des bibliothèques | Toutes les dépendances |
| 2 | Chargement des données | Lecture Excel, dimensions |
| 3 | Exploration (EDA) | Types, valeurs manquantes, distributions |
| 4 | Prétraitement | Imputation, variance nulle, standardisation |
| 5 | Réduction ACP | Sélection composantes (85% variance) |
| 6 | Nombre optimal de clusters | Elbow, Silhouette, Davies-Bouldin, Calinski-Harabasz |
| 7 | Clustering final (K-Means++) | Modèle final, silhouette plot, projection ACP |
| 8 | Sélection des variables | ANOVA + corrélation |
| 9 | Profiling des clusters | Z-scores, heatmap d'interprétation |
| 10 | Export Excel | 6 feuilles pour indicateurs tabulaires |
| 11 | Sauvegarde du modèle | Récapitulatif artefacts |

---

## 📊 Fichier Excel — `resultats_clustering.xlsx`

| Feuille | Contenu |
|---------|---------|
| `Données_Clustérisées` | Données brutes + colonne Cluster |
| `Moyennes_Clusters` | Moyenne de chaque variable par cluster |
| `ZScores_Clusters` | Z-score des centroïdes (profil relatif) |
| `Répartition` | Effectif et proportion de chaque cluster |
| `Métriques` | Silhouette, Davies-Bouldin, Calinski-Harabasz |
| `Distances_Centroïdes` | Matrice de distances euclidiennes entre centroïdes |

---

## 🔑 Architecture technique

### Pipeline de transformation
```
Nouvelles valeurs brutes
        ↓
StandardScaler.transform()    (même scaler que l'entraînement)
        ↓
PCA.transform()               (même ACP que l'entraînement)
        ↓
KMeans.predict()              (distance aux centroïdes → cluster le plus proche)
```

### Stratégie de sélection des variables
1. **ANOVA** (F-test, p < 0.05) → élimine les variables sans lien avec les clusters  
2. **Corrélation** (|r| > 0.85) → élimine les variables redondantes

### Pas de réentraînement
Le modèle est entraîné **une seule fois** (notebook). L'application charge les `.pkl` pré-calculés. `@st.cache_resource` garantit qu'il n'est chargé qu'une fois en mémoire.

---

## 👤 Auteur

| | |
|--|--|
| **Nom** | NGOM Khadim |
| **Formation** | M1 Informatique |
| **Cours** | Clustering – Séance 7 évaluée |

---

## 📝 Licence

Projet académique — Usage éducatif uniquement.
