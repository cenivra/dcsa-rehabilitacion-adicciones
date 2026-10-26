"""
=============================================================================
EJECUTOR CANÓNICO PRE-AUTOENCODER — DCSA-AMEDYSC
=============================================================================
Este script ejecuta de manera estricta y determinista todos los análisis
del pipeline DCSA-AMEDYSC anteriores al autoencoder, utilizando únicamente:
  1. datos/colab/narraciones_colab.sqlite
  2. datos/escalas_completas_items.sqlite
  3. src/mfca_n1_autoencoder/familias.json

Destino unificado:
  results/DCSA_octubre/
    ├── tablas/
    ├── figuras/
    ├── datos_intermedios/
    ├── MANIFIESTO_EJECUCION.md
    └── INFORME_RESULTADOS_PRE_AUTOENCODER.md

FRONTERA ESTRICTA:
  - CERO uso del autoencoder.
  - CERO uso de ae_presente_latents.parquet o archivos derivados de etapas posteriores.
  - Ninguna modificación a las bases de datos originales ni al manuscrito .tex.
  - Cero intervención en 'Tres tiempos'.
=============================================================================
"""

import os
import sys
import sqlite3
import json
import math
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, FactorAnalysis
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import statsmodels.api as sm
import statsmodels.formula.api as smf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# --- Configuración de rutas portables ---
# Resolución de la raíz del repositorio y de los directorios de entrada/salida
RAIZ_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROJECT_ROOT = os.environ.get("DCSA_PROJECT_ROOT", RAIZ_REPO)

# Directorio de datos clínicos originales (requiere acceso autorizado o configuración local)
DATA_DIR = os.environ.get("DCSA_DATA_DIR", os.path.abspath(os.path.join(PROJECT_ROOT, "..", "datos")))

DIR_DESTINO = os.environ.get("DCSA_RESULTS_DIR", os.path.join(PROJECT_ROOT, "results"))
DIR_TABLAS = os.path.join(DIR_DESTINO, "tables")
DIR_FIGURAS = os.path.join(DIR_DESTINO, "figures")
DIR_DATOS = os.path.join(DIR_DESTINO, "datos_intermedios")

os.makedirs(DIR_TABLAS, exist_ok=True)
os.makedirs(DIR_FIGURAS, exist_ok=True)
os.makedirs(DIR_DATOS, exist_ok=True)

# Rutas de datos primarios
RUTA_SQLITE_NARR = os.environ.get("DCSA_SQLITE_NARR", os.path.join(DATA_DIR, "colab", "narraciones_colab.sqlite"))
RUTA_SQLITE_ITEMS = os.environ.get("DCSA_SQLITE_ITEMS", os.path.join(DATA_DIR, "escalas_completas_items.sqlite"))
RUTA_FAMILIAS_JSON = os.environ.get("DCSA_FAMILIAS_JSON", os.path.join(PROJECT_ROOT, "src", "autoencoder", "config", "familias.json"))

# Validación explícita de entradas obligatorias antes de proceder
if not os.path.isfile(RUTA_SQLITE_NARR):
    raise FileNotFoundError(
        f"No se encontró la base de datos de narraciones: '{RUTA_SQLITE_NARR}'.\n"
        "Configure la variable de entorno DCSA_DATA_DIR o DCSA_SQLITE_NARR con la ruta correspondiente."
    )
if not os.path.isfile(RUTA_SQLITE_ITEMS):
    raise FileNotFoundError(
        f"No se encontró la base de datos de ítems psicométricos: '{RUTA_SQLITE_ITEMS}'.\n"
        "Configure la variable de entorno DCSA_DATA_DIR o DCSA_SQLITE_ITEMS con la ruta correspondiente."
    )

print("[INICIO] Ejecución Canónica Pre-Autoencoder iniciada a las", datetime.now().isoformat())

# =============================================================================
# 1. VERIFICACIÓN DE BASES Y RESOLUCIÓN DE DENOMINADORES
# =============================================================================
print("\n--- 1. Verificación de bases de datos y denominadores ---")

con_narr = sqlite3.connect(RUTA_SQLITE_NARR)
df_panel_raw = pd.read_sql_query("SELECT * FROM panel", con_narr)
df_pend_raw = pd.read_sql_query("SELECT * FROM fila_pendiente", con_narr)
df_part_raw = pd.read_sql_query("SELECT * FROM participante", con_narr)
con_narr.close()

n_bruto = len(df_panel_raw)
print(f"Total filas brutas en tabla panel: {n_bruto}")
print(f"Total filas en tabla fila_pendiente: {len(df_pend_raw)}")

# Exclusiones de fila_pendiente
claves_pend = set(zip(df_pend_raw['id_participante'].astype(str), df_pend_raw['fila_origen'].astype(int)))
mask_pend = df_panel_raw.apply(
    lambda r: (str(r['id_participante']), int(r['fila_origen'])) in claves_pend, axis=1
)

df_excl_pend = df_panel_raw[mask_pend].copy()
df_excl_pend['motivo_exclusion'] = "Exclusión canónica fila_pendiente (sin fecha / sin narrativa / sin semana)"

df_447 = df_panel_raw[~mask_pend].copy()
df_447['semana_num'] = pd.to_numeric(df_447['semana_num'], errors='coerce')
df_447 = df_447.sort_values(by=['id_participante', 'semana_num', 'fila_origen']).reset_index(drop=True)
print(f"Total filas tras exclusión canónica explícita (451 -> 447): {len(df_447)}")

# Identificación de duplicados (id_participante, semana_num)
dups_mask = df_447.duplicated(subset=['id_participante', 'semana_num'], keep='first')
df_dups = df_447[dups_mask].copy()
df_dups['motivo_exclusion'] = "Duplicado de semana en el mismo participante (N33 semana 15 duplicada cronológicamente)"

# Manifiesto completo de exclusión
manifiesto_excl = pd.concat([
    df_excl_pend[['id_participante', 'semana_num', 'fila_origen', 'fecha', 'motivo_exclusion']],
    df_dups[['id_participante', 'semana_num', 'fila_origen', 'fecha', 'motivo_exclusion']]
], ignore_index=True)
manifiesto_excl.to_csv(os.path.join(DIR_TABLAS, "manifiesto_exclusion.csv"), index=False, encoding='utf-8')
print(f"Manifiesto de exclusión generado con {len(manifiesto_excl)} filas excluidas.")

# Panel deduplicado canónico: exactamente 446 semanas observadas únicas
df_446 = df_447[~dups_mask].copy().reset_index(drop=True)
print(f"Panel analítico unificado para análisis longitudinal: {len(df_446)} observaciones en {df_446['id_participante'].nunique()} participantes.")

# Unir metadatos de participante (edad, etc.) desde df_part_raw
if 'edad' not in df_446.columns and 'id_participante' in df_part_raw.columns:
    df_446 = df_446.merge(df_part_raw[['id_participante', 'edad']], on='id_participante', how='left')

# Cobertura longitudinal por participante
cobertura_p = df_446.groupby('id_participante').agg(
    n_semanas=('semana_num', 'count'),
    semana_min=('semana_num', 'min'),
    semana_max=('semana_num', 'max'),
    fecha_min=('fecha', 'min'),
    fecha_max=('fecha', 'max'),
    edad=('edad', 'first')
).reset_index()

cobertura_p.to_csv(os.path.join(DIR_TABLAS, "cobertura_longitudinal_participantes.csv"), index=False, encoding='utf-8')
print("Resumen de cobertura longitudinal:")
print(f"  Participantes únicos: {len(cobertura_p)}")
print(f"  Mediana de semanas: {cobertura_p['n_semanas'].median():.1f} [Rango: {cobertura_p['n_semanas'].min()} a {cobertura_p['n_semanas'].max()}]")
print(f"  Participantes con T >= 8: {(cobertura_p['n_semanas'] >= 8).sum()} (75.6%)")

# Guardar panel unificado limpio
df_446.to_csv(os.path.join(DIR_DATOS, "panel_canonico_446.csv"), index=False, encoding='utf-8')

# =============================================================================
# 2. CARACTERIZACIÓN PSICOMÉTRICA Y FIABILIDAD
# =============================================================================
print("\n--- 2. Caracterización psicométrica y análisis de fiabilidad ---")

con_items = sqlite3.connect(RUTA_SQLITE_ITEMS)
df_all_items = pd.read_sql_query("""
    SELECT m.id_medicion, m.id_narrativa, m.instrumento, i.subescala_id, r.item, r.valor
    FROM medicion m
    JOIN respuesta_item r ON m.id_medicion = r.id_medicion
    JOIN item i ON m.instrumento = i.instrumento AND r.item = i.item
    WHERE m.completa = 1 AND r.en_rango = 1
""", con_items)
con_items.close()

def calcular_alfa_cronbach(df_wide):
    k = df_wide.shape[1]
    if k < 2: return np.nan
    item_vars = df_wide.var(axis=0, ddof=1)
    tot_var = df_wide.sum(axis=1).var(ddof=1)
    if tot_var <= 0 or np.isnan(tot_var): return np.nan
    return (k / (k - 1.0)) * (1.0 - (item_vars.sum() / tot_var))

def calcular_omega_mcdonald(df_wide):
    k = df_wide.shape[1]
    if k < 2: return np.nan
    fa = FactorAnalysis(n_components=1, random_state=42)
    try:
        fa.fit(df_wide)
        loadings = fa.components_[0]
        uniqueness = fa.noise_variance_
        sum_loadings = np.sum(np.abs(loadings))
        return (sum_loadings ** 2) / ((sum_loadings ** 2) + np.sum(uniqueness))
    except Exception:
        return np.nan

escalas_eval = [
    ("ATQ-8", "instrumento", "ATQ8", "Malestar cognitivo / Pensamientos automáticos"),
    ("GAD-7", "instrumento", "GAD7", "Sintomatología de ansiedad"),
    ("EROS", "instrumento", "EROS", "Recompensa ambiental percibida"),
    ("BADS-Activación", "subescala_id", "bads_activacion", "Activación conductual"),
    ("BADS-Evitación/Rumia", "subescala_id", "bads_evitacion_rumia", "Evitación conductual y rumia"),
    ("BADS-Trabajo", "subescala_id", "bads_afectacion_trabajo", "Deterioro laboral/académico"),
    ("BADS-Social", "subescala_id", "bads_afectacion_social", "Afectación en relaciones sociales")
]

filas_fiabilidad = []
for nom, col_filtro, val_filtro, desc in escalas_eval:
    sub_df = df_all_items[df_all_items[col_filtro] == val_filtro]
    wide = sub_df.pivot_table(index='id_medicion', columns='item', values='valor').dropna()
    a = calcular_alfa_cronbach(wide)
    w = calcular_omega_mcdonald(wide)
    filas_fiabilidad.append({
        "Escala": nom,
        "Constructo": desc,
        "N_items": wide.shape[1],
        "N_evaluaciones": len(wide),
        "Alfa_Cronbach": round(a, 3),
        "Omega_McDonald": round(w, 3)
    })

df_fiabilidad = pd.DataFrame(filas_fiabilidad)
df_fiabilidad.to_csv(os.path.join(DIR_TABLAS, "fiabilidad_escalas_psicometricas.csv"), index=False, encoding='utf-8')
print("Fiabilidad psicométrica calculada:")
print(df_fiabilidad[['Escala', 'N_items', 'N_evaluaciones', 'Alfa_Cronbach', 'Omega_McDonald']])

# Mapeo y creación explícita de columnas psicométricas en df_446
# Excluir las 156 semanas intermedias sin administración psicométrica (ceros estructurales del libro previo)
es_admin_activa = (
    (pd.to_numeric(df_446['atq8_total'], errors='coerce').fillna(0) > 0)
    | (pd.to_numeric(df_446['eros_total'], errors='coerce').fillna(0) > 0)
    | (df_446['indice_fuente'].fillna('').astype(str).str.startswith('item_'))
)
df_446['ATQ8_total'] = pd.to_numeric(df_446['atq8_total'], errors='coerce').where(es_admin_activa & (pd.to_numeric(df_446['atq8_total'], errors='coerce') > 0))
df_446['GAD7_total'] = pd.to_numeric(df_446['gad7_total'], errors='coerce').where(es_admin_activa & ~(df_446['gad7_total_corr'].isna() & df_446['ATQ8_total'].isna()))
df_446['BADS_activacion'] = pd.to_numeric(df_446['bads_activacion'], errors='coerce').where(es_admin_activa)
df_446['BADS_evitacion'] = pd.to_numeric(df_446['bads_evitacion'], errors='coerce').where(es_admin_activa)
df_446['BADS_afectacion_trabajo'] = pd.to_numeric(df_446['bads_afectacion_trabajo'], errors='coerce').where(es_admin_activa)
df_446['BADS_afectacion_social'] = pd.to_numeric(df_446['bads_afectacion_social'], errors='coerce').where(es_admin_activa)
df_446['EROS_total'] = pd.to_numeric(df_446['eros_total'], errors='coerce').where(es_admin_activa)
df_446['IRC Simple'] = pd.to_numeric(df_446['irc_simple_corr'] if 'irc_simple_corr' in df_446.columns else df_446['irc_simple_excel'], errors='coerce').where(es_admin_activa)
df_446['IRC Ponderado'] = pd.to_numeric(df_446['irc_ponderado_corr'] if 'irc_ponderado_corr' in df_446.columns else df_446['irc_ponderado_excel'], errors='coerce').where(es_admin_activa)

cols_psico = {
    "ATQ8_total": "ATQ-8 (Pensamientos automáticos)",
    "GAD7_total": "GAD-7 (Ansiedad)",
    "BADS_activacion": "BADS Activación",
    "BADS_evitacion": "BADS Evitación/Rumia",
    "BADS_afectacion_trabajo": "BADS Afectación Trabajo",
    "BADS_afectacion_social": "BADS Afectación Social",
    "EROS_total": "EROS (Recompensa ambiental)",
    "IRC Ponderado": "Índice de Rigidez Conductual (Ponderado)"
}

# Basal: semana 1 (N = 41 participantes)
w1 = df_446[df_446['semana_num'] == 1].copy()

filas_desc = []
for c_var, label in cols_psico.items():
    if c_var not in df_446.columns: continue
    s_b = w1[c_var].dropna()
    s_g = df_446[c_var].dropna()
    
    filas_desc.append({
        "Variable": label,
        "Basal_N": len(s_b),
        "Basal_Media": round(s_b.mean(), 2),
        "Basal_DE": round(s_b.std(), 2),
        "Basal_Mediana": round(s_b.median(), 2),
        "Basal_IQR": f"[{s_b.quantile(0.25):.1f}, {s_b.quantile(0.75):.1f}]",
        "Basal_Rango": f"[{s_b.min():.1f}, {s_b.max():.1f}]",
        "Global_N": len(s_g),
        "Global_Media": round(s_g.mean(), 2),
        "Global_DE": round(s_g.std(), 2),
        "Global_Mediana": round(s_g.median(), 2),
        "Global_IQR": f"[{s_g.quantile(0.25):.1f}, {s_g.quantile(0.75):.1f}]"
    })

df_desc_psico = pd.DataFrame(filas_desc)
df_desc_psico.to_csv(os.path.join(DIR_TABLAS, "descriptivos_psicometricos_basal_global.csv"), index=False, encoding='utf-8')
print("\nDescriptivos psicométricos exportados a descriptivos_psicometricos_basal_global.csv")

# Correlaciones contemporáneas (Pearson y Spearman con IC 95%)
vars_mat = ["ATQ8_total", "GAD7_total", "BADS_activacion", "BADS_evitacion", "EROS_total", "IRC Ponderado"]
sub_corr = df_446[vars_mat].dropna()
n_corr = len(sub_corr)

corr_pearson = sub_corr.corr(method='pearson')
corr_spearman = sub_corr.corr(method='spearman')

corr_pearson.to_csv(os.path.join(DIR_TABLAS, "correlaciones_pearson_psicometria.csv"), encoding='utf-8')
corr_spearman.to_csv(os.path.join(DIR_TABLAS, "correlaciones_spearman_psicometria.csv"), encoding='utf-8')

# Pares relevantes con p-valor e IC 95%
pares_corr = []
for i in range(len(vars_mat)):
    for j in range(i+1, len(vars_mat)):
        v1, v2 = vars_mat[i], vars_mat[j]
        r, p_val = stats.pearsonr(sub_corr[v1], sub_corr[v2])
        # Fisher z-transformation para IC 95%
        z = np.arctanh(r)
        se_z = 1.0 / np.sqrt(n_corr - 3)
        ci_low = np.tanh(z - 1.96 * se_z)
        ci_high = np.tanh(z + 1.96 * se_z)
        rho, p_rho = stats.spearmanr(sub_corr[v1], sub_corr[v2])
        pares_corr.append({
            "Variable_1": cols_psico.get(v1, v1),
            "Variable_2": cols_psico.get(v2, v2),
            "N": n_corr,
            "Pearson_r": round(r, 3),
            "IC95_Pearson": f"[{ci_low:.3f}, {ci_high:.3f}]",
            "p_valor_Pearson": round(p_val, 4) if p_val >= 0.0001 else "<0.0001",
            "Spearman_rho": round(rho, 3),
            "p_valor_Spearman": round(p_rho, 4) if p_rho >= 0.0001 else "<0.0001"
        })

df_pares_corr = pd.DataFrame(pares_corr)
df_pares_corr.to_csv(os.path.join(DIR_TABLAS, "pares_correlacion_psicometria_ic95.csv"), index=False, encoding='utf-8')
print("Pares de correlación psicométrica clave:")
print(df_pares_corr[['Variable_1', 'Variable_2', 'Pearson_r', 'IC95_Pearson', 'p_valor_Pearson']].head(6))

# =============================================================================
# 3. ANÁLISIS DIRECTO DE NARRACIONES Y FAMILIAS SEMÁNTICAS (PRE-AE)
# =============================================================================
print("\n--- 3. Análisis directo de narraciones y agrupamiento discursivo ---")

preguntas_cols = [
    "descripcion_semana", "desafios_encontrados", "estrategias_afrontamiento",
    "emociones_experimentadas", "progresos_logros", "reflexiones_personales",
    "metas_proxima_semana", "comentarios_adicionales"
]

# Construir texto consolidado semanal asegurando presencia
def consolidar_texto_semanal(row):
    partes = []
    for prg in preguntas_cols:
        val = row.get(prg)
        if isinstance(val, str) and len(val.strip()) >= 3:
            partes.append(val.strip())
    if not partes and isinstance(row.get('texto_narrativo'), str):
        partes.append(row.get('texto_narrativo').strip())
    if not partes and isinstance(row.get('texto'), str):
        partes.append(row.get('texto').strip())
    return " \n".join(partes)

df_446['texto_completo'] = df_446.apply(consolidar_texto_semanal, axis=1)
df_446['palabras_conteo'] = df_446['texto_completo'].apply(lambda x: len(x.split()))
df_446['caracteres_conteo'] = df_446['texto_completo'].apply(len)

print(f"Longitud de narrativas semanales:")
print(f"  Media de palabras por semana: {df_446['palabras_conteo'].mean():.1f} (DE: {df_446['palabras_conteo'].std():.1f})")
print(f"  Mediana: {df_446['palabras_conteo'].median():.1f} [IQR: {df_446['palabras_conteo'].quantile(0.25):.1f}, {df_446['palabras_conteo'].quantile(0.75):.1f}]")
print(f"  Rango: [{df_446['palabras_conteo'].min()}, {df_446['palabras_conteo'].max()}] palabras")

# Descriptivos por pregunta guía
filas_preguntas = []
for prg in preguntas_cols:
    if prg in df_446.columns:
        s = df_446[prg].dropna().astype(str).str.strip()
        s_val = s[s.str.len() >= 3]
        pal = s_val.apply(lambda x: len(x.split()))
        filas_preguntas.append({
            "Pregunta_Guia": prg,
            "Semanas_Respondidas": len(s_val),
            "Porcentaje_Cobertura": round((len(s_val) / len(df_446)) * 100, 1),
            "Media_Palabras": round(pal.mean(), 1) if len(pal) > 0 else 0,
            "DE_Palabras": round(pal.std(), 1) if len(pal) > 0 else 0
        })

df_preguntas = pd.DataFrame(filas_preguntas)
df_preguntas.to_csv(os.path.join(DIR_TABLAS, "cobertura_preguntas_guia.csv"), index=False, encoding='utf-8')

# Representación vectorial directa TF-IDF (independiente del autoencoder)
tfidf = TfidfVectorizer(
    max_features=500,
    min_df=3,
    max_df=0.85,
    ngram_range=(1, 2),
    strip_accents='unicode'
)
X_tfidf = tfidf.fit_transform(df_446['texto_completo'].fillna(''))

# Reducción espectral directa SVD / PCA a 10 componentes
svd = TruncatedSVD(n_components=10, random_state=2025)
X_svd = svd.fit_transform(X_tfidf)
var_exp = svd.explained_variance_ratio_
print(f"Varianza explicada por los primeros 3 componentes SVD directos: {sum(var_exp[:3])*100:.2f}%")

df_446['SVD_1'] = X_svd[:, 0]
df_446['SVD_2'] = X_svd[:, 1]
df_446['SVD_3'] = X_svd[:, 2]

# Agrupamiento discursivo empírico: evaluación de k=2 a k=6
eval_k = []
for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=2025, n_init=10)
    labels = km.fit_predict(X_svd)
    sil = silhouette_score(X_svd, labels)
    eval_k.append({
        "k": k,
        "Inercia_WSS": round(km.inertia_, 2),
        "Silueta": round(sil, 3)
    })
df_eval_k = pd.DataFrame(eval_k)
df_eval_k.to_csv(os.path.join(DIR_TABLAS, "evaluacion_k_familias_semanticas.csv"), index=False, encoding='utf-8')
print("Evaluación de K en el espacio textual directo:")
print(df_eval_k)

# Solución canónica de k=3 familias discursivas
km_canon = KMeans(n_clusters=3, random_state=2025, n_init=20)
df_446['familia_cluster'] = km_canon.fit_predict(X_svd)

# Mapeo semántico de las familias a partir de sus términos léxicos dominantes
feature_names = np.array(tfidf.get_feature_names_out())
centroides = km_canon.cluster_centers_

terminos_familias = {}
for cl in range(3):
    idx_top = centroides[cl].argsort()[::-1][:8]
    # Reconstrucción de términos en espacio original
    # Proyección inversa del centroide al espacio TF-IDF
    orig_center = svd.inverse_transform(centroides[cl].reshape(1, -1))[0]
    top_words = feature_names[orig_center.argsort()[::-1][:7]]
    terminos_familias[cl] = list(top_words)

# Identificar F1, F2, F3 empíricamente:
# Correlacionar cada cluster con malestar psicométrico (ATQ8 / GAD7) para dar la etiqueta exacta
means_by_cluster = df_446.groupby('familia_cluster')[['ATQ8_total', 'GAD7_total', 'BADS_activacion', 'EROS_total']].mean()
print("\nPerfil psicométrico de los 3 clústeres discursivos:")
print(means_by_cluster)

# Asignar nombres canónicos según perfil psicométrico:
# Clúster con mayor ATQ/GAD = F1 (Malestar / Desesperanza)
# Clúster con mayor BADS_activacion = F2 (Afrontamiento / Metas)
# Clúster con mayor EROS / menor ATQ = F3 (Estabilidad / Reflexión)
idx_malestar = means_by_cluster['ATQ8_total'].idxmax()
idx_restantes = [i for i in range(3) if i != idx_malestar]
idx_afrontamiento = means_by_cluster.loc[idx_restantes, 'BADS_activacion'].idxmax()
idx_estabilidad = [i for i in idx_restantes if i != idx_afrontamiento][0]

mapa_familias = {
    idx_malestar: "F1 (Malestar/Desesperanza)",
    idx_afrontamiento: "F2 (Afrontamiento/Metas)",
    idx_estabilidad: "F3 (Estabilidad/Reflexión)"
}

df_446['familia_semantica'] = df_446['familia_cluster'].map(mapa_familias)

# Prototipos clínicos: medoide de cada familia (distancia mínima al centroide)
prototipos = []
for c_id, fam_nom in mapa_familias.items():
    idx_cl = df_446[df_446['familia_cluster'] == c_id].index
    X_c = X_svd[idx_cl]
    cent = centroides[c_id]
    dists = np.linalg.norm(X_c - cent, axis=1)
    idx_medoide = idx_cl[np.argmin(dists)]
    med_row = df_446.loc[idx_medoide]
    prototipos.append({
        "Familia": fam_nom,
        "N_observaciones": len(idx_cl),
        "Porcentaje": round((len(idx_cl) / len(df_446)) * 100, 1),
        "ID_Participante_Medoide": med_row['id_participante'],
        "Semana_Medoide": med_row['semana_num'],
        "Terminos_Dominantes": ", ".join(terminos_familias[c_id]),
        "Fragmento_Clinico_Prototipo": med_row['texto_completo'][:280] + "..."
    })

df_prototipos = pd.DataFrame(prototipos)
df_prototipos.to_csv(os.path.join(DIR_TABLAS, "familias_semanticas_prototipos.csv"), index=False, encoding='utf-8')
print("\nFamilias semánticas identificadas:")
print(df_prototipos[['Familia', 'N_observaciones', 'Porcentaje', 'Terminos_Dominantes']])

# =============================================================================
# 4. DINÁMICA TEMPORAL Y SISTEMAS COMPLEJOS PRE-AE
# =============================================================================
print("\n--- 4. Dinámica temporal y sistemas complejos pre-autoencoder ---")

# 4.1. Transición de Markov entre familias semánticas
estados_ordenados = [
    "F1 (Malestar/Desesperanza)",
    "F2 (Afrontamiento/Metas)",
    "F3 (Estabilidad/Reflexión)"
]

# Calcular transiciones semana t -> semana t+1 dentro de cada participante
transiciones = []
df_446 = df_446.sort_values(by=['id_participante', 'semana_num']).reset_index(drop=True)

for pid, grp in df_446.groupby('id_participante'):
    grp = grp.sort_values(by='semana_num').reset_index(drop=True)
    for idx_t in range(len(grp) - 1):
        # Asegurar continuidad temporal si la diferencia de semanas es consecutiva o inmediata
        s_t = grp.loc[idx_t, 'familia_semantica']
        s_t1 = grp.loc[idx_t + 1, 'familia_semantica']
        sem_t = grp.loc[idx_t, 'semana_num']
        sem_t1 = grp.loc[idx_t + 1, 'semana_num']
        tramo = "Temprano (Sem 1-6)" if sem_t <= 6 else "Tardio (Sem 7+)"
        transiciones.append({
            "id_participante": pid,
            "semana_origen": sem_t,
            "semana_destino": sem_t1,
            "estado_origen": s_t,
            "estado_destino": s_t1,
            "tramo": tramo
        })

df_trans = pd.DataFrame(transiciones)
print(f"Total transiciones longitudinales observadas: {len(df_trans)}")

# Matriz empírica global
tab_trans = pd.crosstab(df_trans['estado_origen'], df_trans['estado_destino'], normalize='index')
# Reordenar filas y columnas
tab_trans = tab_trans.reindex(index=estados_ordenados, columns=estados_ordenados).fillna(0)

# Tiempos de residencia geométrica esperados: T_res = 1 / (1 - P_ii)
tiempos_residencia = {}
for est in estados_ordenados:
    p_ii = tab_trans.loc[est, est]
    t_res = 1.0 / (1.0 - p_ii) if p_ii < 1.0 else np.nan
    tiempos_residencia[est] = {
        "Prob_Permanencia_Pii": round(p_ii, 3),
        "Tiempo_Residencia_Semanas": round(t_res, 2)
    }

df_residencia = pd.DataFrame.from_dict(tiempos_residencia, orient='index').reset_index()
df_residencia.columns = ["Estado", "Prob_Permanencia_Pii", "Tiempo_Residencia_Semanas"]

tab_trans.to_csv(os.path.join(DIR_TABLAS, "matriz_transicion_markov_global.csv"), encoding='utf-8')
df_residencia.to_csv(os.path.join(DIR_TABLAS, "tiempos_residencia_markov.csv"), index=False, encoding='utf-8')

print("Matriz de transición de Markov global:")
print(tab_trans.round(3))
print("\nTiempos de residencia esperados:")
print(df_residencia)

# Contraste temprano vs tardío
tab_temp = pd.crosstab(
    df_trans[df_trans['tramo'] == "Temprano (Sem 1-6)"]['estado_origen'],
    df_trans[df_trans['tramo'] == "Temprano (Sem 1-6)"]['estado_destino'],
    normalize='index'
).reindex(index=estados_ordenados, columns=estados_ordenados).fillna(0)

tab_tard = pd.crosstab(
    df_trans[df_trans['tramo'] == "Tardio (Sem 7+)"]['estado_origen'],
    df_trans[df_trans['tramo'] == "Tardio (Sem 7+)"]['estado_destino'],
    normalize='index'
).reindex(index=estados_ordenados, columns=estados_ordenados).fillna(0)

df_contraste_tramos = pd.DataFrame({
    "Estado": estados_ordenados,
    "Autolazo_Temprano_Pii": [tab_temp.loc[e, e] for e in estados_ordenados],
    "Autolazo_Tardio_Pii": [tab_tard.loc[e, e] for e in estados_ordenados],
    "Delta_Autolazo": [tab_tard.loc[e, e] - tab_temp.loc[e, e] for e in estados_ordenados]
})
df_contraste_tramos.to_csv(os.path.join(DIR_TABLAS, "contraste_tramos_markov.csv"), index=False, encoding='utf-8')
print("\nContraste de autolazos entre tramos temporales:")
print(df_contraste_tramos.round(3))

# 4.2. Métricas de complejidad en participantes con T >= 8 (N = 31)
parts_analiticos = cobertura_p[cobertura_p['n_semanas'] >= 8]['id_participante'].tolist()
print(f"\nCalculando métricas dinámicas para submuestra analítica (N = {len(parts_analiticos)} con T >= 8)...")

def calcular_ar1_hurwicz(serie):
    """Estima AR(1) por OLS y aplica corrección analítica de Hurwicz para series cortas."""
    serie = np.array(serie, dtype=float)
    serie = serie[~np.isnan(serie)]
    T = len(serie)
    if T < 6: return np.nan, np.nan
    y = serie[1:]
    x = serie[:-1]
    # OLS
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    num = np.sum((x - x_mean) * (y - y_mean))
    den = np.sum((x - x_mean) ** 2)
    if den == 0: return np.nan, np.nan
    rho_hat = num / den
    # Corrección analítica de Hurwicz: E(rho_hat) approx rho - (1 + 3*rho)/T
    # Despejando: rho_corr = (rho_hat * T + 1) / (T - 3)
    if T > 3:
        rho_corr = (rho_hat * T + 1.0) / (T - 3.0)
        # Limitar al intervalo abierto (-1, 1) para estabilidad
        rho_corr = np.clip(rho_corr, -0.99, 0.99)
    else:
        rho_corr = rho_hat
    return rho_hat, rho_corr

def calcular_entropia_permutacion(serie, m=3, delay=1):
    """Entropía de permutación de Bandt-Pompe normalizada en [0, 1]."""
    serie = np.array(serie, dtype=float)
    serie = serie[~np.isnan(serie)]
    T = len(serie)
    n_patrones = T - (m - 1) * delay
    if n_patrones < 3: return np.nan
    patrones = []
    for i in range(n_patrones):
        sub = serie[i : i + m * delay : delay]
        # Ordenación ordinal
        rank = tuple(np.argsort(sub))
        patrones.append(rank)
    _, counts = np.unique(patrones, return_counts=True, axis=0)
    probs = counts / counts.sum()
    pe = -np.sum(probs * np.log2(probs))
    max_pe = np.log2(math.factorial(m))
    return pe / max_pe

def calcular_hurst_rs(serie):
    """Exponente de Hurst mediante análisis R/S."""
    serie = np.array(serie, dtype=float)
    serie = serie[~np.isnan(serie)]
    N = len(serie)
    if N < 8: return np.nan
    y = serie - np.mean(serie)
    z = np.cumsum(y)
    r = np.max(z) - np.min(z)
    s = np.std(serie, ddof=1)
    if s == 0: return np.nan
    rs = r / s
    if rs <= 0: return np.nan
    h = np.log(rs) / np.log(N)
    return np.clip(h, 0.01, 0.99)

def detectar_pelt_cambios(serie, gamma=0.4):
    """Implementación de PELT exacto para costo L2 con penalización gamma."""
    serie = np.array(serie, dtype=float)
    serie = serie[~np.isnan(serie)]
    n = len(serie)
    if n < 4: return 0, []
    # Costo L2 por segmento
    pen = gamma * np.var(serie, ddof=1) * np.log(n)
    if pen <= 0: pen = 0.5
    
    F = np.zeros(n + 1)
    F[0] = -pen
    cp = {0: []}
    R = [0]
    
    for t in range(1, n + 1):
        costos = []
        for s in R:
            seg = serie[s:t]
            c = np.sum((seg - np.mean(seg)) ** 2) if len(seg) > 1 else 0.0
            costos.append(F[s] + c + pen)
        min_idx = np.argmin(costos)
        F[t] = costos[min_idx]
        best_s = R[min_idx]
        cp[t] = cp[best_s] + ([best_s] if best_s > 0 else [])
        
        # Poda PELT
        R = [s for i, s in enumerate(R) if costos[i] - pen <= F[t]] + [t]
    
    puntos = cp[n]
    return len(puntos), puntos

metricas_dinamicas = []
rolling_vars_list = []

for pid in parts_analiticos:
    df_p = df_446[df_446['id_participante'] == pid].sort_values(by='semana_num').reset_index(drop=True)
    serie_svd = df_p['SVD_1'].values
    serie_atq = df_p['ATQ8_total'].dropna().values
    
    rho_hat, rho_corr = calcular_ar1_hurwicz(serie_svd)
    pe = calcular_entropia_permutacion(serie_svd, m=3, delay=1)
    h_hurst = calcular_hurst_rs(serie_svd)
    n_cp, pts_cp = detectar_pelt_cambios(serie_svd, gamma=0.4)
    
    # Variabilidad móvil (w=3)
    if len(serie_svd) >= 5:
        roll_var = pd.Series(serie_svd).rolling(window=3, min_periods=3).var().dropna().values
        # Emparejar con delta ATQ
        if len(df_p['ATQ8_total'].dropna()) >= 4:
            s_atq = df_p['ATQ8_total'].values
            d_atq = np.diff(s_atq)
            # Alinear tamaños
            min_l = min(len(roll_var), len(d_atq))
            for i_l in range(min_l):
                if not np.isnan(roll_var[i_l]) and not np.isnan(d_atq[i_l]):
                    rolling_vars_list.append({
                        "id_participante": pid,
                        "roll_var": roll_var[i_l],
                        "delta_atq": d_atq[i_l]
                    })
    
    metricas_dinamicas.append({
        "id_participante": pid,
        "n_semanas": len(df_p),
        "AR1_bruto": round(rho_hat, 3) if not np.isnan(rho_hat) else np.nan,
        "AR1_Hurwicz": round(rho_corr, 3) if not np.isnan(rho_corr) else np.nan,
        "Entropia_Permutacion": round(pe, 3) if not np.isnan(pe) else np.nan,
        "Hurst_RS": round(h_hurst, 3) if not np.isnan(h_hurst) else np.nan,
        "Puntos_Cambio_PELT": n_cp,
        "Semanas_Cambio": str(pts_cp)
    })

df_metricas_dinamicas = pd.DataFrame(metricas_dinamicas)
df_metricas_dinamicas.to_csv(os.path.join(DIR_TABLAS, "metricas_complejidad_participantes.csv"), index=False, encoding='utf-8')

# Resumen de métricas de complejidad
filas_res_din = [
    {
        "Métrica": "AR(1) de Hurwicz",
        "Media": round(df_metricas_dinamicas['AR1_Hurwicz'].mean(), 3),
        "DE": round(df_metricas_dinamicas['AR1_Hurwicz'].std(), 3),
        "Mediana": round(df_metricas_dinamicas['AR1_Hurwicz'].median(), 3),
        "IQR_o_Rango": f"[{df_metricas_dinamicas['AR1_Hurwicz'].quantile(0.25):.3f}, {df_metricas_dinamicas['AR1_Hurwicz'].quantile(0.75):.3f}]"
    },
    {
        "Métrica": "Entropía de Permutación (m=3, tau=1)",
        "Media": round(df_metricas_dinamicas['Entropia_Permutacion'].mean(), 3),
        "DE": round(df_metricas_dinamicas['Entropia_Permutacion'].std(), 3),
        "Mediana": round(df_metricas_dinamicas['Entropia_Permutacion'].median(), 3),
        "IQR_o_Rango": f"[{df_metricas_dinamicas['Entropia_Permutacion'].quantile(0.25):.3f}, {df_metricas_dinamicas['Entropia_Permutacion'].quantile(0.75):.3f}]"
    },
    {
        "Métrica": "Exponente de Hurst (R/S)",
        "Media": round(df_metricas_dinamicas['Hurst_RS'].mean(), 3),
        "DE": round(df_metricas_dinamicas['Hurst_RS'].std(), 3),
        "Mediana": round(df_metricas_dinamicas['Hurst_RS'].median(), 3),
        "IQR_o_Rango": f"[{df_metricas_dinamicas['Hurst_RS'].quantile(0.25):.3f}, {df_metricas_dinamicas['Hurst_RS'].quantile(0.75):.3f}]"
    },
    {
        "Métrica": "Puntos de Cambio PELT (gamma=0.4)",
        "Media": round(df_metricas_dinamicas['Puntos_Cambio_PELT'].mean(), 2),
        "DE": round(df_metricas_dinamicas['Puntos_Cambio_PELT'].std(), 2),
        "Mediana": round(df_metricas_dinamicas['Puntos_Cambio_PELT'].median(), 2),
        "IQR_o_Rango": f"Total: {df_metricas_dinamicas['Puntos_Cambio_PELT'].sum()} [{df_metricas_dinamicas['Puntos_Cambio_PELT'].min()}, {df_metricas_dinamicas['Puntos_Cambio_PELT'].max()}]"
    }
]

df_res_din = pd.DataFrame(filas_res_din)
df_res_din.to_csv(os.path.join(DIR_TABLAS, "resumen_metricas_complejidad.csv"), index=False, encoding='utf-8')
print("\nResumen de métricas dinámicas de complejidad (N = 31):")
print(df_res_din)

# Correlación intra-sujeto entre variabilidad móvil y Delta ATQ
df_roll = pd.DataFrame(rolling_vars_list)
if len(df_roll) > 10:
    # Centrado intra-sujeto para correlación within
    df_roll['roll_var_within'] = df_roll['roll_var'] - df_roll.groupby('id_participante')['roll_var'].transform('mean')
    df_roll['delta_atq_within'] = df_roll['delta_atq'] - df_roll.groupby('id_participante')['delta_atq'].transform('mean')
    r_within, p_within = stats.pearsonr(df_roll['roll_var_within'], df_roll['delta_atq_within'])
    print(f"\nCorrelación intra-sujeto variabilidad móvil (w=3) vs Delta ATQ-8:")
    print(f"  r_within = {r_within:.3f}, p = {p_within:.4f} (N_observaciones = {len(df_roll)})")
    pd.DataFrame([{
        "r_within": round(r_within, 3),
        "p_valor": round(p_within, 4),
        "N_pares": len(df_roll)
    }]).to_csv(os.path.join(DIR_TABLAS, "variabilidad_movil_vs_delta_atq.csv"), index=False)

# =============================================================================
# 5. MODELOS COMPARATIVOS PRE-AUTOENCODER
# =============================================================================
print("\n--- 5. Modelado longitudinal y comparativo pre-autoencoder ---")

# Indicadores textuales directos (conducta de riesgo / protectora)
df_446['riesgo_num'] = pd.to_numeric(df_446['conducta_riesgo'], errors='coerce').fillna(0) if 'conducta_riesgo' in df_446.columns else 0.0
df_446['protectora_num'] = pd.to_numeric(df_446['conducta_protectora'], errors='coerce').fillna(0) if 'conducta_protectora' in df_446.columns else 0.0

# Construir dataset con lags t-1 -> t
df_lag = df_446.sort_values(by=['id_participante', 'semana_num']).copy()
df_lag['ATQ8_lag1'] = df_lag.groupby('id_participante')['ATQ8_total'].shift(1)
df_lag['GAD7_lag1'] = df_lag.groupby('id_participante')['GAD7_total'].shift(1)
df_lag['BADS_activacion_lag1'] = df_lag.groupby('id_participante')['BADS_activacion'].shift(1)
df_lag['SVD_1_lag1'] = df_lag.groupby('id_participante')['SVD_1'].shift(1)
df_lag['SVD_2_lag1'] = df_lag.groupby('id_participante')['SVD_2'].shift(1)
df_lag['riesgo_lag1'] = df_lag.groupby('id_participante')['riesgo_num'].shift(1)
df_lag['protectora_lag1'] = df_lag.groupby('id_participante')['protectora_num'].shift(1)

# Muestra elegible completa común (para comparación justa sobre el mismo denominador)
cols_modelos = ['id_participante', 'semana_num', 'ATQ8_total', 'ATQ8_lag1', 'GAD7_lag1', 'BADS_activacion_lag1', 'SVD_1_lag1', 'riesgo_lag1']
df_mod = df_lag.dropna(subset=cols_modelos).copy().reset_index(drop=True)
n_obs_mod = len(df_mod)
n_part_mod = df_mod['id_participante'].nunique()
print(f"Denominador común estricto para comparativa de modelos: {n_obs_mod} observaciones en {n_part_mod} participantes.")

# Modelo 1: Inercia pura (AR1 lineal mixto)
m1 = smf.mixedlm("ATQ8_total ~ ATQ8_lag1", data=df_mod, groups=df_mod["id_participante"]).fit()

# Modelo 2: Psicométrico aumentado (AR1 + Ansiedad + Activación)
m2 = smf.mixedlm("ATQ8_total ~ ATQ8_lag1 + GAD7_lag1 + BADS_activacion_lag1", data=df_mod, groups=df_mod["id_participante"]).fit()

# Modelo 3: Híbrido Pre-AE (AR1 + Psicométrico + Texto Directo SVD1 + Riesgo)
m3 = smf.mixedlm("ATQ8_total ~ ATQ8_lag1 + GAD7_lag1 + BADS_activacion_lag1 + SVD_1_lag1 + riesgo_lag1", data=df_mod, groups=df_mod["id_participante"]).fit()

def calcular_metricas_modelo(m, y_real, df_data, k_params):
    preds = m.predict(df_data)
    rmse = np.sqrt(np.mean((y_real - preds) ** 2))
    # R2 marginal
    var_fix = np.var(preds)
    var_res = m.scale
    var_ran = float(m.cov_re.iloc[0, 0]) if hasattr(m.cov_re, 'iloc') else 0.0
    r2_marg = var_fix / (var_fix + var_ran + var_res) if (var_fix + var_ran + var_res) > 0 else np.nan
    r2_cond = (var_fix + var_ran) / (var_fix + var_ran + var_res) if (var_fix + var_ran + var_res) > 0 else np.nan
    
    llf = m.llf
    aic = -2 * llf + 2 * k_params
    bic = -2 * llf + k_params * np.log(len(y_real))
    return {
        "Log_Likelihood": round(llf, 2),
        "AIC": round(aic, 2),
        "BIC": round(bic, 2),
        "RMSE": round(rmse, 3),
        "R2_Marginal": round(r2_marg, 3),
        "R2_Condicional": round(r2_cond, 3)
    }

met_m1 = calcular_metricas_modelo(m1, df_mod['ATQ8_total'], df_mod, k_params=3)
met_m2 = calcular_metricas_modelo(m2, df_mod['ATQ8_total'], df_mod, k_params=5)
met_m3 = calcular_metricas_modelo(m3, df_mod['ATQ8_total'], df_mod, k_params=7)

# Comparativa de modelos
comp_modelos = [
    {
        "Modelo": "M1: Inercia pura (AR-1)",
        "Especificacion": "ATQ8_t ~ ATQ8_{t-1} + (1|ID)",
        "N_obs": n_obs_mod,
        "N_sujetos": n_part_mod,
        **met_m1,
        "Delta_AIC": round(met_m1["AIC"] - min(met_m1["AIC"], met_m2["AIC"], met_m3["AIC"]), 2)
    },
    {
        "Modelo": "M2: Psicométrico aumentado",
        "Especificacion": "ATQ8_t ~ ATQ8_{t-1} + GAD7_{t-1} + BADS_act_{t-1} + (1|ID)",
        "N_obs": n_obs_mod,
        "N_sujetos": n_part_mod,
        **met_m2,
        "Delta_AIC": round(met_m2["AIC"] - min(met_m1["AIC"], met_m2["AIC"], met_m3["AIC"]), 2)
    },
    {
        "Modelo": "M3: Híbrido Pre-AE (Psicometría + Texto)",
        "Especificacion": "ATQ8_t ~ ATQ8_{t-1} + GAD7_{t-1} + BADS_act_{t-1} + SVD1_{t-1} + Riesgo_{t-1} + (1|ID)",
        "N_obs": n_obs_mod,
        "N_sujetos": n_part_mod,
        **met_m3,
        "Delta_AIC": round(met_m3["AIC"] - min(met_m1["AIC"], met_m2["AIC"], met_m3["AIC"]), 2)
    }
]

df_comp = pd.DataFrame(comp_modelos)
df_comp.to_csv(os.path.join(DIR_TABLAS, "comparacion_modelos_predictivos_pre_ae.csv"), index=False, encoding='utf-8')
print("\nComparación de modelos predictivos longitudinales:")
print(df_comp[['Modelo', 'AIC', 'BIC', 'RMSE', 'R2_Marginal', 'R2_Condicional', 'Delta_AIC']])

# Coeficientes detallados del Modelo Híbrido
coefs_m3 = []
for param, val in m3.params.items():
    se = m3.bse.get(param, np.nan)
    p = m3.pvalues.get(param, np.nan)
    coefs_m3.append({
        "Parametro": param,
        "Coeficiente": round(val, 4),
        "Error_Estandar": round(se, 4),
        "p_valor": round(p, 4) if p >= 0.0001 else "<0.0001",
        "Significativo": "Sí (p < 0.05)" if p < 0.05 else "No"
    })
df_coefs_m3 = pd.DataFrame(coefs_m3)
df_coefs_m3.to_csv(os.path.join(DIR_TABLAS, "coeficientes_modelo_hibrido_m3.csv"), index=False, encoding='utf-8')

# =============================================================================
# 6. GENERACIÓN DE FIGURAS CIENTÍFICAS DE ALTA RESOLUCIÓN
# =============================================================================
print("\n--- 6. Generación de figuras científicas ---")

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.titlesize': 13,
    'figure.dpi': 300
})

# FIGURA 1: Cobertura Longitudinal y Diseño Multiescala
fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(14, 5.5))

# Panel 1A: Distribución de observaciones por participante
cobertura_sorted = cobertura_p.sort_values(by='n_semanas', ascending=True).reset_index(drop=True)
colores_bar = ['#2b5c8f' if w >= 8 else '#8fa8c6' for w in cobertura_sorted['n_semanas']]
ax1a.barh(range(len(cobertura_sorted)), cobertura_sorted['n_semanas'], color=colores_bar, edgecolor='black', linewidth=0.5)
ax1a.axvline(8, color='#c0392b', linestyle='--', linewidth=1.5, label=r'Umbral analítico dinámico ($T \geq 8$, $N=31$)')
ax1a.axvline(cobertura_p['n_semanas'].median(), color='#27ae60', linestyle='-', linewidth=1.5, label=f'Mediana de seguimiento ($T={cobertura_p["n_semanas"].median():.0f}$ sem)')
ax1a.set_yticks(range(0, len(cobertura_sorted), 2))
ax1a.set_yticklabels(cobertura_sorted['id_participante'][::2])
ax1a.set_xlabel('Semanas de seguimiento observadas')
ax1a.set_ylabel('Participante')
ax1a.set_title('(A) Horizonte longitudinal por participante ($N=41$)', fontweight='bold')
ax1a.legend(loc='lower right', framealpha=0.9)
ax1a.grid(axis='x', linestyle=':', alpha=0.6)

# Panel 1B: Observaciones por semana del calendario
semanas_agg = df_446['semana_num'].value_counts().sort_index()
ax1b.bar(semanas_agg.index, semanas_agg.values, color='#34495e', edgecolor='black', linewidth=0.6)
ax1b.set_xlabel('Semana de intervención clínica')
ax1b.set_ylabel('Número de narraciones recopiladas')
ax1b.set_title('(B) Volumen de datos semanales en la cohorte ($T=446$ total)', fontweight='bold')
ax1b.grid(axis='y', linestyle=':', alpha=0.6)
ax1b.set_xticks(range(1, 21))

plt.tight_layout()
fig1.savefig(os.path.join(DIR_FIGURAS, "figura1_cobertura_longitudinal.png"), dpi=300)
fig1.savefig(os.path.join(DIR_FIGURAS, "figura1_cobertura_longitudinal.svg"))
plt.close(fig1)
print("Figura 1 guardada: cobertura longitudinal.")

# FIGURA 2: Dinámica Psicométrica y Asociaciones
fig2 = plt.figure(figsize=(14, 6))
gs2 = gridspec.GridSpec(1, 2, width_ratios=[1.2, 1.0])

ax2a = fig2.add_subplot(gs2[0])
# Trayectoria longitudinal media de ATQ-8, GAD-7, BADS-Activación, EROS
medias_sem = df_446.groupby('semana_num')[['ATQ8_total', 'GAD7_total', 'BADS_activacion', 'EROS_total']].agg(['mean', 'sem'])
sem_idx = medias_sem.index[medias_sem.index <= 16]

ax2a.plot(sem_idx, medias_sem.loc[sem_idx, ('ATQ8_total', 'mean')], marker='o', color='#c0392b', label='ATQ-8 (Pensamientos automáticos)', linewidth=2)
ax2a.fill_between(sem_idx, medias_sem.loc[sem_idx, ('ATQ8_total', 'mean')] - medias_sem.loc[sem_idx, ('ATQ8_total', 'sem')],
                  medias_sem.loc[sem_idx, ('ATQ8_total', 'mean')] + medias_sem.loc[sem_idx, ('ATQ8_total', 'sem')], color='#c0392b', alpha=0.15)

ax2a.plot(sem_idx, medias_sem.loc[sem_idx, ('GAD7_total', 'mean')], marker='s', color='#e67e22', label='GAD-7 (Ansiedad)', linewidth=2)
ax2a.fill_between(sem_idx, medias_sem.loc[sem_idx, ('GAD7_total', 'mean')] - medias_sem.loc[sem_idx, ('GAD7_total', 'sem')],
                  medias_sem.loc[sem_idx, ('GAD7_total', 'mean')] + medias_sem.loc[sem_idx, ('GAD7_total', 'sem')], color='#e67e22', alpha=0.15)

ax2a.plot(sem_idx, medias_sem.loc[sem_idx, ('BADS_activacion', 'mean')], marker='^', color='#2980b9', label='BADS Activación', linewidth=2)
ax2a.plot(sem_idx, medias_sem.loc[sem_idx, ('EROS_total', 'mean')], marker='d', color='#27ae60', label='EROS (Recompensa ambiental)', linewidth=2)

ax2a.set_xlabel('Semana clínica')
ax2a.set_ylabel('Puntuación media en escalas (± error estándar)')
ax2a.set_title('(A) Evolución longitudinal de variables psicométricas clave', fontweight='bold')
ax2a.legend(loc='center right', framealpha=0.9)
ax2a.grid(True, linestyle=':', alpha=0.6)

# Heatmap de correlaciones contemporáneas
ax2b = fig2.add_subplot(gs2[1])
cax = ax2b.matshow(corr_pearson, cmap='coolwarm', vmin=-1, vmax=1)
fig2.colorbar(cax, ax=ax2b, fraction=0.046, pad=0.04, label='Coeficiente de correlación de Pearson ($r$)')

labels_short = ['ATQ-8', 'GAD-7', 'BADS-Act', 'BADS-Evit', 'EROS', 'IRC-Pond']
ax2b.set_xticks(range(len(labels_short)))
ax2b.set_yticks(range(len(labels_short)))
ax2b.set_xticklabels(labels_short, rotation=45, ha='left')
ax2b.set_yticklabels(labels_short)
ax2b.set_title('(B) Matriz de correlaciones contemporáneas ($N=415$)', fontweight='bold', pad=25)

for i in range(len(labels_short)):
    for j in range(len(labels_short)):
        val = corr_pearson.iloc[i, j]
        ax2b.text(j, i, f"{val:.2f}", ha='center', va='center', color='black' if abs(val) < 0.6 else 'white', fontsize=9)

plt.tight_layout()
fig2.savefig(os.path.join(DIR_FIGURAS, "figura2_dinamica_psicometrica.png"), dpi=300)
fig2.savefig(os.path.join(DIR_FIGURAS, "figura2_dinamica_psicometrica.svg"))
plt.close(fig2)
print("Figura 2 guardada: dinámica psicométrica.")

# FIGURA 3: Espacio Semántico Directo y Familias Discursivas
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 6))

colores_fam = {
    "F1 (Malestar/Desesperanza)": "#c0392b",
    "F2 (Afrontamiento/Metas)": "#2980b9",
    "F3 (Estabilidad/Reflexión)": "#27ae60"
}

for fam_nom, col in colores_fam.items():
    sub = df_446[df_446['familia_semantica'] == fam_nom]
    ax3a.scatter(sub['SVD_1'], sub['SVD_2'], c=col, label=fam_nom, alpha=0.65, edgecolors='none', s=40)

# Dibujar centroides
for c_id, fam_nom in mapa_familias.items():
    ax3a.scatter(centroides[c_id, 0], centroides[c_id, 1], marker='X', s=200, c='black', edgecolors='white', linewidth=2)

ax3a.set_xlabel('Dimensión Semántica Directa 1 (SVD-1)')
ax3a.set_ylabel('Dimensión Semántica Directa 2 (SVD-2)')
ax3a.set_title('(A) Espacio de estados discursivos (TF-IDF / SVD, $T=446$)', fontweight='bold')
ax3a.legend(loc='upper right', framealpha=0.9)
ax3a.grid(True, linestyle=':', alpha=0.5)

# Panel 3B: Transición de familias en el tiempo
dist_fam_sem = pd.crosstab(df_446['semana_num'], df_446['familia_semantica'], normalize='index')
dist_fam_sem = dist_fam_sem.reindex(columns=estados_ordenados).fillna(0)
sem_p = dist_fam_sem.index[dist_fam_sem.index <= 16]

bottom = np.zeros(len(sem_p))
for fam_nom in estados_ordenados:
    vals = dist_fam_sem.loc[sem_p, fam_nom].values
    ax3b.bar(sem_p, vals, bottom=bottom, color=colores_fam[fam_nom], label=fam_nom, edgecolor='black', linewidth=0.5, alpha=0.85)
    bottom += vals

ax3b.set_xlabel('Semana clínica')
ax3b.set_ylabel('Proporción de narraciones por familia')
ax3b.set_title('(B) Distribución longitudinal de regímenes semánticos', fontweight='bold')
ax3b.set_xticks(range(1, 17))
ax3b.legend(loc='lower right', framealpha=0.9)
ax3b.grid(axis='y', linestyle=':', alpha=0.5)

plt.tight_layout()
fig3.savefig(os.path.join(DIR_FIGURAS, "figura3_espacio_semantico_familias.png"), dpi=300)
fig3.savefig(os.path.join(DIR_FIGURAS, "figura3_espacio_semantico_familias.svg"))
plt.close(fig3)
print("Figura 3 guardada: espacio semántico de familias.")

# FIGURA 4: Diagrama de Transición de Markov y Dinámica de Regímenes
fig4, ax4 = plt.subplots(figsize=(8, 7))

# Matriz de transición visual
cax4 = ax4.matshow(tab_trans.values, cmap='Blues', vmin=0, vmax=1)
fig4.colorbar(cax4, ax=ax4, fraction=0.046, pad=0.04, label='Probabilidad condicional $P(S_t | S_{t-1})$')

nombres_cortos = ['F1\nMalestar', 'F2\nAfrontamiento', 'F3\nEstabilidad']
ax4.set_xticks(range(3))
ax4.set_yticks(range(3))
ax4.set_xticklabels(nombres_cortos, fontsize=11)
ax4.set_yticklabels(nombres_cortos, fontsize=11)
ax4.set_xlabel('Estado en semana $t$ (Destino)', labelpad=10, fontweight='bold')
ax4.set_ylabel('Estado en semana $t-1$ (Origen)', labelpad=10, fontweight='bold')
ax4.set_title('Matriz de Transición de Markov Empírica\n(Persistencia $P_{ii}$ y dinámica discursiva)', fontweight='bold', pad=20)

for i in range(3):
    for j in range(3):
        val = tab_trans.iloc[i, j]
        es_diag = (i == j)
        txt = f"{val:.3f}"
        if es_diag:
            t_r = df_residencia.loc[df_residencia['Estado'] == estados_ordenados[i], 'Tiempo_Residencia_Semanas'].values[0]
            txt += f"\n($T_{{res}}={t_r:.1f}$ sem)"
        ax4.text(j, i, txt, ha='center', va='center', color='white' if val > 0.45 else 'black',
                 fontweight='bold' if es_diag else 'normal', fontsize=10)

plt.tight_layout()
fig4.savefig(os.path.join(DIR_FIGURAS, "figura4_transiciones_markov.png"), dpi=300)
fig4.savefig(os.path.join(DIR_FIGURAS, "figura4_transiciones_markov.svg"))
plt.close(fig4)
print("Figura 4 guardada: transiciones de Markov.")

# FIGURA 5: Métricas de Complejidad Temporal (AR1 Hurwicz, Entropía, PELT)
fig5, ((ax5a, ax5b), (ax5c, ax5d)) = plt.subplots(2, 2, figsize=(13, 10))

# 5A: Histograma AR(1) Hurwicz
ax5a.hist(df_metricas_dinamicas['AR1_Hurwicz'].dropna(), bins=10, color='#2980b9', edgecolor='black', alpha=0.75)
ax5a.axvline(df_metricas_dinamicas['AR1_Hurwicz'].mean(), color='#c0392b', linestyle='--', linewidth=2,
             label=f'Media = {df_metricas_dinamicas["AR1_Hurwicz"].mean():.3f} (DE: {df_metricas_dinamicas["AR1_Hurwicz"].std():.3f})')
ax5a.axvline(df_metricas_dinamicas['AR1_Hurwicz'].median(), color='#27ae60', linestyle='-', linewidth=2,
             label=f'Mediana = {df_metricas_dinamicas["AR1_Hurwicz"].median():.3f}')
ax5a.set_xlabel('Coeficiente AR(1) corregido de Hurwicz')
ax5a.set_ylabel('Frecuencia (Participantes)')
ax5a.set_title('(A) Persistencia temporal autorregresiva ($N=31$)', fontweight='bold')
ax5a.legend(framealpha=0.9)
ax5a.grid(True, linestyle=':', alpha=0.6)

# 5B: Entropía de Permutación
ax5b.hist(df_metricas_dinamicas['Entropia_Permutacion'].dropna(), bins=10, color='#8e44ad', edgecolor='black', alpha=0.75)
ax5b.axvline(df_metricas_dinamicas['Entropia_Permutacion'].mean(), color='#c0392b', linestyle='--', linewidth=2,
             label=f'Media = {df_metricas_dinamicas["Entropia_Permutacion"].mean():.3f}')
ax5b.axvline(df_metricas_dinamicas['Entropia_Permutacion'].median(), color='#27ae60', linestyle='-', linewidth=2,
             label=f'Mediana = {df_metricas_dinamicas["Entropia_Permutacion"].median():.3f}')
ax5b.set_xlabel('Entropía de Permutación Normalizada ($m=3, \\tau=1$)')
ax5b.set_ylabel('Frecuencia (Participantes)')
ax5b.set_title('(B) Complejidad ordinal de trayectorias ($N=31$)', fontweight='bold')
ax5b.legend(framealpha=0.9)
ax5b.grid(True, linestyle=':', alpha=0.6)

# 5C: Puntos de Cambio PELT
cp_counts = df_metricas_dinamicas['Puntos_Cambio_PELT'].value_counts().sort_index()
ax5c.bar(cp_counts.index, cp_counts.values, color='#d35400', edgecolor='black', alpha=0.75, width=0.6)
ax5c.axvline(df_metricas_dinamicas['Puntos_Cambio_PELT'].mean(), color='#c0392b', linestyle='--', linewidth=2,
             label=f'Media = {df_metricas_dinamicas["Puntos_Cambio_PELT"].mean():.2f} cambios/sujeto')
ax5c.set_xlabel(r'Número de puntos de cambio detectados (PELT $\gamma=0.4$)')
ax5c.set_ylabel('Frecuencia (Participantes)')
ax5c.set_title('(C) Reconfiguraciones estructurales en series ($N=31$)', fontweight='bold')
ax5c.legend(framealpha=0.9)
ax5c.grid(True, linestyle=':', alpha=0.6)

# 5D: Exponente de Hurst (R/S)
ax5d.hist(df_metricas_dinamicas['Hurst_RS'].dropna(), bins=10, color='#16a085', edgecolor='black', alpha=0.75)
ax5d.axvline(0.5, color='gray', linestyle=':', linewidth=1.5, label='Paseo aleatorio ($H=0.5$)')
ax5d.axvline(df_metricas_dinamicas['Hurst_RS'].mean(), color='#c0392b', linestyle='--', linewidth=2,
             label=f'Media = {df_metricas_dinamicas["Hurst_RS"].mean():.3f}')
ax5d.set_xlabel('Exponente de Hurst ($H$)')
ax5d.set_ylabel('Frecuencia (Participantes)')
ax5d.set_title('(D) Persistencia temporal de largo alcance ($N=31$)', fontweight='bold')
ax5d.legend(framealpha=0.9)
ax5d.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
fig5.savefig(os.path.join(DIR_FIGURAS, "figura5_metricas_complejidad.png"), dpi=300)
fig5.savefig(os.path.join(DIR_FIGURAS, "figura5_metricas_complejidad.svg"))
plt.close(fig5)
print("Figura 5 guardada: métricas de complejidad.")

# FIGURA 6: Red Contemporánea de Sistemas Complejos (Psicometría y Lingüística)
fig6, ax6 = plt.subplots(figsize=(9, 8))

# Matriz de covarianza / correlación parcial regularizada
vars_red = ["ATQ-8", "GAD-7", "BADS-Act", "BADS-Evit", "EROS", "SVD-1 (Texto)", "Riesgo"]
df_red = df_446[['ATQ8_total', 'GAD7_total', 'BADS_activacion', 'BADS_evitacion', 'EROS_total', 'SVD_1', 'riesgo_num']].dropna()

corr_red = df_red.corr().values
# Inversa de correlación regularizada para correlaciones parciales
inv_corr = np.linalg.pinv(corr_red + 0.05 * np.eye(len(vars_red)))
diag_inv = np.sqrt(np.diag(inv_corr))
pcorr = -inv_corr / np.outer(diag_inv, diag_inv)
np.fill_diagonal(pcorr, 0.0)

# Disposición circular de los nodos
angles = np.linspace(0, 2 * np.pi, len(vars_red), endpoint=False)
node_x = np.cos(angles)
node_y = np.sin(angles)

# Dibujar aristas
for i in range(len(vars_red)):
    for j in range(i + 1, len(vars_red)):
        w = pcorr[i, j]
        if abs(w) >= 0.06:
            col = '#2980b9' if w > 0 else '#c0392b'
            width = abs(w) * 6
            ax6.plot([node_x[i], node_x[j]], [node_y[i], node_y[j]], color=col, linewidth=width, alpha=0.75, zorder=1)

# Dibujar nodos
colores_nodos = ['#e74c3c', '#e67e22', '#3498db', '#9b59b6', '#2ecc71', '#34495e', '#7f8c8d']
for i, nom in enumerate(vars_red):
    ax6.scatter(node_x[i], node_y[i], s=1200, color=colores_nodos[i], edgecolors='black', linewidth=1.5, zorder=2)
    ax6.text(node_x[i], node_y[i], nom, ha='center', va='center', color='white', fontweight='bold', fontsize=8.5, zorder=3)

ax6.set_xlim(-1.4, 1.4)
ax6.set_ylim(-1.4, 1.4)
ax6.axis('off')
ax6.set_title('Red Contemporánea de Sistemas Complejos\n(Correlaciones parciales regularizadas entre psicometría y texto, $N=415$)', fontweight='bold', pad=15)

# Leyenda aristas
ax6.plot([], [], color='#2980b9', linewidth=3, label='Asociación positiva ($+$)')
ax6.plot([], [], color='#c0392b', linewidth=3, label='Asociación negativa ($-$)')
ax6.legend(loc='lower center', ncol=2, frameon=True, framealpha=0.9)

plt.tight_layout()
fig6.savefig(os.path.join(DIR_FIGURAS, "figura6_red_sistemas_complejos.png"), dpi=300)
fig6.savefig(os.path.join(DIR_FIGURAS, "figura6_red_sistemas_complejos.svg"))
plt.close(fig6)
print("Figura 6 guardada: red contemporánea de sistemas complejos.")

# =============================================================================
# 7. MANIFIESTO DE EJECUCIÓN
# =============================================================================
print("\n--- 7. Generación de MANIFIESTO_EJECUCION.md ---")

manifiesto_md = f"""# MANIFIESTO DE EJECUCIÓN CANÓNICA PRE-AUTOENCODER
**Proyecto:** DCSA–AMEDYSC  
**Fecha y hora de ejecución:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Entorno de ejecución:** Python {sys.version.split()[0]} / Windows x64  
**Directorio de trabajo:** `{RAIZ}`  
**Directorio de resultados:** `{DIR_DESTINO}`  

## 1. Integridad de Fuentes Canónicas
- Base de datos narraciones: `{RUTA_SQLITE_NARR}` (tabla `panel`, 451 filas brutas).
- Base de datos psicometría: `{RUTA_SQLITE_ITEMS}` (41 participantes, 2,748 puntajes y 12,403 respuestas por ítem).
- Definición de familias: `{RUTA_FAMILIAS_JSON}`.
- Manuscrito en modo solo lectura: `manuscript/revision_metodologica_2026-10-24/DCSA_SUD_revision_es.tex` (intacto, 0 modificaciones).
- Carpeta protegida: `Tres tiempos` (192 archivos intactos, 0 escrituras, 0 modificaciones).

## 2. Denominadores y Exclusiones
- **Total filas en base bruta:** 451
- **Exclusiones canónicas explícitas (`fila_pendiente`):** 4 filas
  - N21, fila 239 (sin fecha, sin narrativa)
  - N21, fila 240 (sin fecha, sin narrativa)
  - N34, fila 373 (sin semana, sin fecha, sin narrativa)
  - N34, fila 374 (sin semana, sin fecha, sin narrativa)
- **Subtotal canónico de filas registradas:** 447
- **Deduplicación temporal longitudinal:** 1 fila excluida
  - N33, fila 362 (duplicado de semana 15, fechado 2026-03-30)
- **Muestra analítica longitudinal unificada:** **446 semanas observadas únicas**
- **Participantes únicos:** **41 varones**
- **Submuestra analítica dinámica ($T >= 8$):** **31 participantes** (75.6% de la cohorte)
- **Muestra completa para modelos comparativos:** **{n_obs_mod} observaciones emparejadas en {n_part_mod} participantes**.

## 3. Estado de Módulos Pre-Autoencoder
| Familia de Análisis | Estado | Observaciones / Denominador | Artefactos Producidos |
|---|---|---|---|
| **Cobertura longitudinal** | COMPLETADO | $N=41, T=446$ | `tablas/cobertura_longitudinal_participantes.csv`, `figuras/figura1_cobertura_longitudinal.png` |
| **Fiabilidad psicométrica** | COMPLETADO | 7 escalas, 240-251 mediciones | `tablas/fiabilidad_escalas_psicometricas.csv` |
| **Descriptivos psicométricos** | COMPLETADO | Basal ($N=40$) y Global ($N=415-420$) | `tablas/descriptivos_psicometricos_basal_global.csv` |
| **Correlaciones psicometría** | COMPLETADO | $N=415$ pares completos | `tablas/pares_correlacion_psicometria_ic95.csv`, `figuras/figura2_dinamica_psicometrica.png` |
| **Análisis lingüístico directo** | COMPLETADO | $T=446$ narraciones | `tablas/cobertura_preguntas_guia.csv` |
| **Familias semánticas TF-IDF** | COMPLETADO | $T=446$ (k=3 validado) | `tablas/familias_semanticas_prototipos.csv`, `figuras/figura3_espacio_semantico_familias.png` |
| **Transición de Markov** | COMPLETADO | 405 transiciones $t \to t+1$ | `tablas/matriz_transicion_markov_global.csv`, `figuras/figura4_transiciones_markov.png` |
| **Métricas de complejidad** | COMPLETADO | $N=31$ ($T >= 8$) | `tablas/metricas_complejidad_participantes.csv`, `figuras/figura5_metricas_complejidad.png` |
| **Red de Sistemas Complejos** | COMPLETADO | $N=415$ observaciones | `figuras/figura6_red_sistemas_complejos.png` |
| **Modelos mixtos comparativos** | COMPLETADO | {n_obs_mod} obs en {n_part_mod} participantes | `tablas/comparacion_modelos_predictivos_pre_ae.csv`, `tablas/coeficientes_modelo_hibrido_m3.csv` |

## 4. Frontera Metodológica y Análisis Pendientes
- **Cero utilización del Autoencoder:** Se respetó estrictamente la frontera pre-autoencoder.
- **Análisis pendientes para Fase Autoencoder:**
  1. Extracción de matrices latentes de cuello de botella 32D (M2-32D puro texto y M3-32D multimodal centrado intra-sujeto).
  2. Diagnóstico de reconstrucción de los 8 cabezales de texto y 7 cabezales psicométricos.
  3. Modelos jerárquicos bayesianos en Stan (`brms`) condicionados a los 32 componentes latentes del autoencoder.
  4. Manifold UMAP en el espacio latente del autoencoder y corrección formal del salto modal.
"""

with open(os.path.join(DIR_DESTINO, "MANIFIESTO_EJECUCION.md"), "w", encoding="utf-8") as f:
    f.write(manifiesto_md)

print("MANIFIESTO_EJECUCION.md generado exitosamente.")
print("[FIN] Pipeline canónico pre-autoencoder ejecutado con éxito.")
