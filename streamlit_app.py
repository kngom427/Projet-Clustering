"""
╔══════════════════════════════════════════════════════════════════╗
║       WEB APP STREAMLIT — DÉPLOIEMENT MODÈLE DE CLUSTERING       ║
║       M1 Informatique – Séance 7 évaluée                         ║
╚══════════════════════════════════════════════════════════════════╝

Usage :
    streamlit run streamlit_app.py

Auteur : NGOM Khadim
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.metrics.pairwise import euclidean_distances

# ══════════════════════════════════════════════════════════════
# CONFIGURATION DE LA PAGE
# ══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Clustering — Affectation",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
# CSS PERSONNALISÉ
# ══════════════════════════════════════════════════════════════

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
  }

  /* ── Fond général ── */
  .stApp {
    background: #0d1117;
    color: #e6edf3;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #30363d;
  }
  [data-testid="stSidebar"] .stMarkdown p,
  [data-testid="stSidebar"] label {
    color: #8b949e !important;
    font-size: 0.82rem;
    font-family: 'IBM Plex Mono', monospace;
  }

  /* ── Titres ── */
  h1 { 
    font-family: 'IBM Plex Mono', monospace !important; 
    color: #58a6ff !important;
    letter-spacing: -0.5px;
    font-size: 1.8rem !important;
  }
  h2, h3 { 
    font-family: 'IBM Plex Mono', monospace !important; 
    color: #c9d1d9 !important;
  }

  /* ── Cartes métriques ── */
  .metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 20px 24px;
    text-align: center;
    transition: border-color 0.2s;
  }
  .metric-card:hover { border-color: #58a6ff; }
  .metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #58a6ff;
    line-height: 1;
  }
  .metric-label {
    font-size: 0.75rem;
    color: #8b949e;
    margin-top: 6px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
  }

  /* ── Résultat cluster ── */
  .result-banner {
    background: linear-gradient(135deg, #0d419d22, #1f6feb33);
    border: 2px solid #1f6feb;
    border-radius: 12px;
    padding: 28px 32px;
    text-align: center;
    margin: 16px 0;
  }
  .result-cluster-num {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 4rem;
    font-weight: 700;
    color: #58a6ff;
    line-height: 1;
  }
  .result-cluster-label {
    font-size: 0.9rem;
    color: #8b949e;
    margin-top: 8px;
    letter-spacing: 1px;
    text-transform: uppercase;
  }

  /* ── Badges variables ── */
  .badge-high {
    display: inline-block;
    background: #1a3a2a;
    border: 1px solid #2ea043;
    color: #3fb950;
    border-radius: 4px;
    padding: 3px 10px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    margin: 3px 2px;
  }
  .badge-low {
    display: inline-block;
    background: #3a1a1a;
    border: 1px solid #da3633;
    color: #f85149;
    border-radius: 4px;
    padding: 3px 10px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    margin: 3px 2px;
  }

  /* ── Barre de distance ── */
  .dist-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 6px 0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
  }
  .dist-label { color: #8b949e; width: 80px; flex-shrink: 0; }
  .dist-bar-bg {
    flex: 1;
    background: #21262d;
    border-radius: 4px;
    height: 10px;
    overflow: hidden;
  }
  .dist-bar-fill {
    height: 100%;
    border-radius: 4px;
    background: #30363d;
    transition: width 0.5s ease;
  }
  .dist-bar-fill.active { background: #58a6ff; }
  .dist-val { color: #c9d1d9; width: 70px; text-align: right; }

  /* ── Info box ── */
  .info-box {
    background: #161b22;
    border-left: 4px solid #58a6ff;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin: 12px 0;
    font-size: 0.85rem;
    color: #8b949e;
    font-family: 'IBM Plex Mono', monospace;
  }

  /* ── Inputs ── */
  .stNumberInput input, .stTextInput input {
    background: #21262d !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.85rem !important;
  }
  .stNumberInput input:focus, .stTextInput input:focus {
    border-color: #58a6ff !important;
    box-shadow: 0 0 0 3px #58a6ff22 !important;
  }

  /* ── Bouton principal ── */
  .stButton > button {
    background: #1f6feb !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    padding: 12px 28px !important;
    width: 100%;
    letter-spacing: 0.3px;
    transition: background 0.2s, transform 0.1s !important;
  }
  .stButton > button:hover {
    background: #388bfd !important;
    transform: translateY(-1px) !important;
  }
  .stButton > button:active { transform: translateY(0) !important; }

  /* ── Slider ── */
  .stSlider > div > div > div { background: #1f6feb !important; }

  /* ── Divider ── */
  hr { border-color: #21262d !important; margin: 24px 0 !important; }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: #0d1117; }
  ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: #58a6ff; }

  /* ── Expander ── */
  .streamlit-expanderHeader {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    color: #8b949e !important;
    font-size: 0.85rem !important;
  }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace !important;
    color: #8b949e !important;
    font-size: 0.85rem !important;
  }
  .stTabs [aria-selected="true"] {
    color: #58a6ff !important;
    border-bottom-color: #58a6ff !important;
  }

  /* ── Alert / success ── */
  .stSuccess { background: #1a3a2a !important; border-color: #2ea043 !important; }
  .stError   { background: #3a1a1a !important; border-color: #da3633 !important; }
  .stWarning { background: #2a2010 !important; border-color: #d29922 !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# CHARGEMENT DU MODÈLE (mis en cache)
# ══════════════════════════════════════════════════════════════

DOSSIER_MODELE = os.path.join(os.path.dirname(__file__), 'modele')

ARTEFACTS = {
    'scaler'      : 'scaler.pkl',
    'pca'         : 'pca.pkl',
    'kmeans'      : 'kmeans_model.pkl',
    'variables'   : 'variables_pertinentes.pkl',
    'colonnes'    : 'colonnes.pkl',
    'profil'      : 'profil_zscore.pkl',
    'k_optimal'   : 'k_optimal.pkl',
    'seuil_zscore': 'seuil_zscore.pkl',
}

@st.cache_resource(show_spinner=False)
def charger_modele():
    """Charge tous les artefacts une seule fois (mis en cache par Streamlit)."""
    modele = {}
    for cle, fichier in ARTEFACTS.items():
        chemin = os.path.join(DOSSIER_MODELE, fichier)
        if not os.path.exists(chemin):
            return None, fichier
        modele[cle] = joblib.load(chemin)
    return modele, None


def predire_cluster(valeurs_individu, modele):
    """Prédit le cluster d'un individu et retourne les distances."""
    colonnes_completes    = modele['colonnes']
    variables_pertinentes = modele['variables']

    vecteur    = np.zeros(len(colonnes_completes))
    col_to_idx = {col: i for i, col in enumerate(colonnes_completes)}

    for var, val in valeurs_individu.items():
        if var in col_to_idx:
            vecteur[col_to_idx[var]] = val

    X        = vecteur.reshape(1, -1)
    X_scaled = modele['scaler'].transform(X)
    X_pca    = modele['pca'].transform(X_scaled)
    distances = modele['kmeans'].transform(X_pca)[0]
    cluster   = modele['kmeans'].predict(X_pca)[0]

    return cluster, distances


def interpreter_cluster(cluster_id, modele):
    """Retourne les variables hautes / basses du cluster."""
    profil = modele['profil']
    seuil  = modele['seuil_zscore']

    if cluster_id not in profil.index:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    z = profil.loc[cluster_id]
    return z[z > seuil].sort_values(ascending=False), z[z < -seuil].sort_values()


def figure_distances(distances, cluster_predit, k):
    """Génère un graphique en barres horizontales des distances aux centroïdes."""
    fig, ax = plt.subplots(figsize=(7, max(2.5, k * 0.55)))
    fig.patch.set_facecolor('#161b22')
    ax.set_facecolor('#161b22')

    couleurs = ['#58a6ff' if i == cluster_predit else '#30363d' for i in range(k)]
    bars = ax.barh(
        [f'Cluster {i}' for i in range(k)],
        distances,
        color=couleurs,
        edgecolor='#0d1117',
        height=0.55
    )

    # Valeurs
    for bar, val in zip(bars, distances):
        ax.text(bar.get_width() + max(distances) * 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', color='#8b949e',
                fontsize=9, fontfamily='monospace')

    ax.set_xlabel('Distance euclidienne au centroïde', color='#8b949e', fontsize=9)
    ax.set_title('Distances aux centroïdes', color='#c9d1d9', fontsize=11, pad=12,
                 fontfamily='monospace')
    ax.tick_params(colors='#8b949e', labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')
    ax.xaxis.label.set_color('#8b949e')

    plt.tight_layout()
    return fig


def figure_profil_zscore(profil_zscore, cluster_predit, n_vars=20):
    """Heatmap du profil z-score pour les variables les plus discriminantes."""
    vars_disc = profil_zscore.abs().max(axis=0).nlargest(n_vars).index.tolist()
    data      = profil_zscore[vars_disc]

    fig, ax = plt.subplots(figsize=(10, max(4, len(vars_disc) * 0.38)))
    fig.patch.set_facecolor('#161b22')
    ax.set_facecolor('#161b22')

    sns.heatmap(
        data.T, cmap='RdBu_r', center=0, vmin=-2, vmax=2,
        annot=True, fmt='.2f', linewidths=0.4,
        cbar_kws={'label': 'Z-score', 'shrink': 0.7},
        ax=ax, annot_kws={'size': 8, 'color': '#e6edf3'}
    )

    # Met en évidence la colonne du cluster prédit
    ax.add_patch(plt.Rectangle(
        (cluster_predit, 0), 1, len(vars_disc),
        fill=False, edgecolor='#58a6ff', lw=2.5, zorder=5
    ))

    ax.set_xticklabels([f'C{i}' for i in range(data.shape[0])],
                       color='#c9d1d9', fontsize=10, fontfamily='monospace')
    ax.set_yticklabels(ax.get_yticklabels(), color='#8b949e', fontsize=8, rotation=0)
    ax.set_xlabel('Cluster', color='#8b949e', fontsize=9)
    ax.set_ylabel('')
    ax.set_title(f'Profil des clusters — {n_vars} variables les plus discriminantes',
                 color='#c9d1d9', fontsize=11, pad=12, fontfamily='monospace')

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(colors='#8b949e', labelsize=8)
    cbar.set_label('Z-score', color='#8b949e', fontsize=8)

    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════
# INTERFACE PRINCIPALE
# ══════════════════════════════════════════════════════════════

def main():

    # ── Chargement du modèle ──────────────────────────────────
    with st.spinner("Chargement du modèle..."):
        modele, fichier_manquant = charger_modele()

    if modele is None:
        st.error(f"❌ Fichier manquant : `modele/{fichier_manquant}`")
        st.info("Exécutez d'abord le notebook `clustering_projet.ipynb` pour générer le dossier `modele/`.")
        st.stop()

    k          = modele['k_optimal']
    vars_pert  = modele['variables']
    profil     = modele['profil']
    seuil_z    = modele['seuil_zscore']

    # ── En-tête ───────────────────────────────────────────────
    col_title, col_badge = st.columns([4, 1])
    with col_title:
        st.markdown("# 🔬 Clustering — Système d'affectation")
        st.markdown(
            "<p style='color:#8b949e; font-size:0.88rem; font-family:IBM Plex Mono,monospace;'>"
            "M1 Informatique · Séance 7 · Modèle K-Means pré-entraîné</p>",
            unsafe_allow_html=True
        )
    with col_badge:
        st.markdown(
            f"<div style='text-align:right; padding-top:12px;'>"
            f"<span style='background:#1a3a2a; border:1px solid #2ea043; color:#3fb950; "
            f"border-radius:20px; padding:6px 14px; font-family:IBM Plex Mono,monospace; font-size:0.8rem;'>"
            f"✅ Modèle chargé</span></div>",
            unsafe_allow_html=True
        )

    st.markdown("---")

    # ── Métriques rapides ─────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    metriques = [
        ("k_optimal.pkl", "k_optimal", "Clusters", k),
        (None, None, "Variables pertinentes", len(vars_pert)),
        (None, None, "Composantes ACP", modele['pca'].n_components_),
        (None, None, "Variables d'entraînement", len(modele['colonnes'])),
    ]
    for col, (_, _, label, val) in zip([c1, c2, c3, c4], metriques):
        with col:
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-value'>{val}</div>"
                f"<div class='metric-label'>{label}</div>"
                f"</div>",
                unsafe_allow_html=True
            )

    st.markdown("---")

    # ══════════════════════════════════════════════════════════
    # SIDEBAR — Saisie des valeurs
    # ══════════════════════════════════════════════════════════
    with st.sidebar:
        st.markdown(
            "<div style='font-family:IBM Plex Mono,monospace; font-size:1rem; "
            "font-weight:600; color:#c9d1d9; padding-bottom:8px;'>"
            "⌨️ Saisie de l'individu</div>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<div class='info-box'>"
            f"Renseignez les <strong style='color:#c9d1d9'>{len(vars_pert)} variables pertinentes</strong> "
            "du modèle pour obtenir l'affectation au cluster."
            "</div>",
            unsafe_allow_html=True
        )

        # Bouton de reset
        if st.button("🔄 Réinitialiser les valeurs", key="reset"):
            for var in vars_pert:
                key = f"val_{var}"
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

        st.markdown("---")

        valeurs_saisies = {}

        # Regroupement par chunks de 10 pour la lisibilité
        CHUNK = 10
        for chunk_i in range(0, len(vars_pert), CHUNK):
            chunk_vars = vars_pert[chunk_i: chunk_i + CHUNK]
            label_exp  = f"Variables {chunk_i + 1}–{min(chunk_i + CHUNK, len(vars_pert))}"

            with st.expander(label_exp, expanded=(chunk_i == 0)):
                for var in chunk_vars:
                    val = st.number_input(
                        label=var,
                        value=0.0,
                        format="%.4f",
                        step=0.01,
                        key=f"val_{var}",
                    )
                    valeurs_saisies[var] = val

        st.markdown("---")
        lancer = st.button("🚀 Prédire le cluster", key="predict")

    # ══════════════════════════════════════════════════════════
    # ZONE PRINCIPALE — Résultats
    # ══════════════════════════════════════════════════════════
    tabs = st.tabs(["📊 Résultat", "📈 Profil des clusters", "ℹ️ À propos"])

    # ── Onglet 1 : Résultat ───────────────────────────────────
    with tabs[0]:

        if not lancer:
            st.markdown(
                "<div class='info-box' style='text-align:center; padding:32px;'>"
                "👈 Renseignez les variables dans la barre latérale, "
                "puis cliquez sur <strong style='color:#58a6ff'>Prédire le cluster</strong>."
                "</div>",
                unsafe_allow_html=True
            )
        else:
            # ── Prédiction ──
            with st.spinner("⏳ Calcul en cours..."):
                cluster_predit, distances = predire_cluster(valeurs_saisies, modele)
                vars_hautes, vars_basses  = interpreter_cluster(cluster_predit, modele)

            # ── Bannière résultat ──
            st.markdown(
                f"<div class='result-banner'>"
                f"<div class='result-cluster-label'>Cluster d'affectation</div>"
                f"<div class='result-cluster-num'>{cluster_predit}</div>"
                f"<div class='result-cluster-label' style='margin-top:6px;'>"
                f"Distance minimale : {distances[cluster_predit]:.4f}</div>"
                f"</div>",
                unsafe_allow_html=True
            )

            col_dist, col_vars = st.columns([1, 1.4])

            # ── Graphique distances ──
            with col_dist:
                st.markdown(
                    "<h3 style='font-size:0.95rem;'>📏 Distances aux centroïdes</h3>",
                    unsafe_allow_html=True
                )
                fig_dist = figure_distances(distances, cluster_predit, k)
                st.pyplot(fig_dist, use_container_width=True)
                plt.close(fig_dist)

            # ── Variables caractéristiques ──
            with col_vars:
                st.markdown(
                    f"<h3 style='font-size:0.95rem;'>🎯 Caractéristiques du Cluster {cluster_predit}</h3>",
                    unsafe_allow_html=True
                )

                if len(vars_hautes) > 0:
                    st.markdown(
                        f"<p style='color:#3fb950; font-family:IBM Plex Mono,monospace; "
                        f"font-size:0.82rem; margin-bottom:6px;'>▲ NOTABLEMENT ÉLEVÉES (z > {seuil_z})</p>",
                        unsafe_allow_html=True
                    )
                    badges_h = "".join(
                        f"<span class='badge-high'>{v} <strong>{z:+.2f}</strong></span>"
                        for v, z in vars_hautes.head(12).items()
                    )
                    st.markdown(badges_h, unsafe_allow_html=True)
                else:
                    st.markdown(
                        "<p style='color:#8b949e; font-size:0.82rem;'>"
                        "Aucune variable notablement élevée.</p>",
                        unsafe_allow_html=True
                    )

                st.markdown("<br>", unsafe_allow_html=True)

                if len(vars_basses) > 0:
                    st.markdown(
                        f"<p style='color:#f85149; font-family:IBM Plex Mono,monospace; "
                        f"font-size:0.82rem; margin-bottom:6px;'>▼ NOTABLEMENT FAIBLES (z < -{seuil_z})</p>",
                        unsafe_allow_html=True
                    )
                    badges_l = "".join(
                        f"<span class='badge-low'>{v} <strong>{z:+.2f}</strong></span>"
                        for v, z in vars_basses.head(12).items()
                    )
                    st.markdown(badges_l, unsafe_allow_html=True)
                else:
                    st.markdown(
                        "<p style='color:#8b949e; font-size:0.82rem;'>"
                        "Aucune variable notablement faible.</p>",
                        unsafe_allow_html=True
                    )

            # ── Tableau synthèse distances ──
            st.markdown("---")
            st.markdown(
                "<h3 style='font-size:0.95rem;'>📋 Tableau récapitulatif des distances</h3>",
                unsafe_allow_html=True
            )
            df_dist = pd.DataFrame({
                'Cluster'             : [f'Cluster {i}' for i in range(k)],
                'Distance euclidienne': [f'{d:.4f}' for d in distances],
                'Affecté'             : ['✅' if i == cluster_predit else '' for i in range(k)],
            })
            st.dataframe(
                df_dist, use_container_width=True, hide_index=True,
                column_config={
                    'Cluster'             : st.column_config.TextColumn('Cluster'),
                    'Distance euclidienne': st.column_config.TextColumn('Distance'),
                    'Affecté'             : st.column_config.TextColumn('Affecté'),
                }
            )

    # ── Onglet 2 : Profil des clusters ────────────────────────
    with tabs[1]:
        st.markdown(
            "<p style='color:#8b949e; font-family:IBM Plex Mono,monospace; font-size:0.85rem;'>"
            "Heatmap des z-scores des centroïdes pour les variables les plus discriminantes. "
            "Valeur positive = au-dessus de la moyenne globale, négative = en-dessous.</p>",
            unsafe_allow_html=True
        )

        n_affich = st.slider(
            "Nombre de variables à afficher", min_value=5, max_value=min(40, len(vars_pert)),
            value=min(20, len(vars_pert)), step=5
        )

        cluster_ref = st.selectbox(
            "Cluster à mettre en évidence",
            options=list(range(k)),
            format_func=lambda x: f"Cluster {x}",
            index=0
        )

        fig_profil = figure_profil_zscore(profil, cluster_ref, n_vars=n_affich)
        st.pyplot(fig_profil, use_container_width=True)
        plt.close(fig_profil)

        # Tableau z-scores
        st.markdown("---")
        st.markdown(
            "<h3 style='font-size:0.95rem;'>📊 Z-scores du cluster sélectionné</h3>",
            unsafe_allow_html=True
        )
        z_cluster = profil.loc[cluster_ref].sort_values(key=abs, ascending=False)
        df_z = pd.DataFrame({
            'Variable'  : z_cluster.index,
            'Z-score'   : z_cluster.values.round(4),
            'Caractère' : ['▲ Élevée' if v > seuil_z else ('▼ Faible' if v < -seuil_z else '≈ Neutre')
                           for v in z_cluster.values]
        })
        st.dataframe(df_z, use_container_width=True, hide_index=True)

    # ── Onglet 3 : À propos ───────────────────────────────────
    with tabs[2]:
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("""
### 🔧 Pipeline du modèle

```
Données brutes (200 variables)
        ↓
StandardScaler      [scaler.pkl]
        ↓
PCA (85% variance)  [pca.pkl]
        ↓
K-Means++           [kmeans_model.pkl]
        ↓
Cluster d'affectation
```

### 📦 Artefacts sauvegardés
| Fichier | Contenu |
|---------|---------|
| `scaler.pkl` | StandardScaler ajusté |
| `pca.pkl` | Modèle ACP |
| `kmeans_model.pkl` | K-Means final |
| `variables_pertinentes.pkl` | Variables sélectionnées |
| `profil_zscore.pkl` | Profil d'interprétation |
""")

        with col_b:
            st.markdown("""
### 🎯 Stratégie de sélection des variables

**1. Test ANOVA** (F-test, p < 0.05)
- Élimine les variables sans lien avec les clusters
- Conserve uniquement les variables discriminantes

**2. Corrélation** (|r| > 0.85)
- Supprime les variables redondantes
- Réduit la multicolinéarité

### ℹ️ Note importante
Le modèle est chargé **une seule fois** au démarrage  
(mis en cache par `@st.cache_resource`).  
Il n'est **jamais réentraîné** lors des prédictions.
""")

        st.markdown("---")
        st.markdown(
            "<div class='info-box'>"
            "👤 <strong style='color:#c9d1d9'>Auteur :</strong> NGOM Khadim · "
            "M1 Informatique · Clustering – Séance 7 évaluée"
            "</div>",
            unsafe_allow_html=True
        )


if __name__ == '__main__':
    main()
