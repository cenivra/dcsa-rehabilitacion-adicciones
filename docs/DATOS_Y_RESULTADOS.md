# Inventario de Tablas, Figuras y Artefactos Científicos Agregados

Este documento detalla el inventario real y verificado de tablas estadísticas y figuras científicas presentes en las carpetas `results/tables/` y `results/figures/` del repositorio **DCSA-reproducibilidad**.

Todos los materiales incorporados corresponden exclusivamente a resúmenes agregados, modelos estadísticos y representaciones gráficas libres de datos sensibles o microdatos individuales.

---

## 1. Inventario de tablas estadísticas agregadas (`results/tables/`)

Se han copiado e incorporado **19 tablas en formato CSV**, estructuradas bajo un esquema de nomenclatura descriptiva y unívoca:

### Módulo Pre-Autoencoder (15 tablas)

| Archivo | Dimensiones | Función científica y descripción de contenido |
| :--- | :--- | :--- |
| `pre_ae_fiabilidad_escalas_psicometricas.csv` | 7 filas $\times$ 6 cols | Consistencia interna ($\alpha$ de Cronbach y $\omega$ de McDonald vía CFA) sobre 12\,376 respuestas válidas de protocolos completos en SQLite. |
| `pre_ae_descriptivos_psicometricos_basal_global.csv` | 8 filas $\times$ 13 cols | Estadísticos descriptivos basales (semana 1, $n=39/38$) y globales ($N=258$ a $263$) calculados por instrumento sobre evaluaciones activas sin ceros estructurales: media, DE, mediana, IQR y rango. |
| `pre_ae_correlaciones_pearson_psicometria.csv` | 6 filas $\times$ 7 cols | Matriz de correlaciones lineales de Pearson entre constructos clínicos en evaluaciones activas completas ($N=256$). |
| `pre_ae_correlaciones_spearman_psicometria.csv` | 6 filas $\times$ 7 cols | Matriz de correlaciones no paramétricas de Spearman entre constructos clínicos en evaluaciones activas completas ($N=256$). |
| `pre_ae_pares_correlacion_psicometria_ic95.csv` | 15 filas $\times$ 8 cols | Coeficientes de correlación de Pearson y Spearman con intervalos de confianza del 95% calculados vía transformación de Fisher y valores $p$. |
| `pre_ae_cobertura_preguntas_guia.csv` | 8 filas $\times$ 5 cols | Volumen discursivo y tasa de respuesta de las 8 preguntas reflexivas estructuradas sobre las 446 semanas del panel. |
| `pre_ae_evaluacion_k_familias_semanticas.csv` | 5 filas $\times$ 3 cols | Diagnósticos de agrupamiento *k-means* para soluciones de $k=2$ a $k=6$: inercia intra-clúster (WSS) y coeficiente de silueta. |
| `pre_ae_familias_semanticas_caracterizacion.csv` | 3 filas $\times$ 6 cols | Proporción de narrativas asignadas a las familias F1 (Malestar/Desesperanza, $N=129$), F2 (Afrontamiento/Metas, $N=209$) y F3 (Estabilidad/Reflexión, $N=108$), sin identificadores de participantes. |
| `pre_ae_matriz_transicion_markov_global.csv` | 3 filas $\times$ 4 cols | Probabilidades de transición empíricas globales de la cadena de Markov de primer orden ($N=405$ transiciones observadas). |
| `pre_ae_contraste_tramos_markov.csv` | 3 filas $\times$ 4 cols | Comparación de las probabilidades de permanencia (autolazos $P_{ii}$) entre el tramo temprano ($t \le 6$, $N=235$) y el tramo tardío ($t \ge 7$, $N=170$). |
| `pre_ae_tiempos_residencia_markov.csv` | 3 filas $\times$ 3 cols | Tiempos geométricos medios de permanencia en semanas ($1 / (1 - P_{ii})$): F1 ($3.13$ sem), F2 ($3.61$ sem) y F3 ($2.71$ sem). |
| `pre_ae_resumen_metricas_complejidad.csv` | 4 filas $\times$ 5 cols | Resumen agregado de persistencia autorregresiva de Hurwicz ($\rho_{\mathrm{corr}} = 0.250$), entropía de permutación ($0.843$), exponente de Hurst ($0.490$) y puntos de corte PELT ($N=31$). |
| `pre_ae_variabilidad_movil_vs_delta_atq.csv` | 1 fila $\times$ 3 cols | Asociación contemporánea intra-sujeto entre inestabilidad discursiva móvil ($w=3$ en SVD-1) y cambio inmediato en malestar cognitivo ($r_{\mathrm{within}} = +0.056, p = 0.3315, N=308$). |
| `pre_ae_comparacion_modelos_predictivos_pre_ae.csv` | 3 filas $\times$ 10 cols | Comparación formal de modelos lineales mixtos M1 (Inercia), M2 (Psicométrico) y M3 (Híbrido): Log-Lik, AIC, BIC, RMSE muestral y $R^2$ marginal/condicional ($N=218$ pares). |
| `pre_ae_coeficientes_modelo_hibrido_m3.csv` | 7 filas $\times$ 6 cols | Parámetros estimados, errores estándar asintóticos, estadísticos $z$ y valores $p$ del modelo mixto aumentado M3. |

### Módulo Autoencoder Multimodal MFCA-N1 (4 tablas)

| Archivo | Dimensiones | Función científica y descripción de contenido |
| :--- | :--- | :--- |
| `autoencoder_reconstruccion_psicometrica_oof.csv` | 7 filas $\times$ 12 cols | Desempeño reconstructivo fuera de pliegue (OOF) de las 7 cabezas psicométricas en la validación cruzada de 5 pliegues: $n_{\mathrm{obs}}$ ($227$ a $236$), MSE, MAE, $R^2_{\mathrm{OOF}}$ y porcentaje de mejora frente a la media de entrenamiento. |
| `autoencoder_reconstruccion_narrativa_oof.csv` | 8 filas $\times$ 12 cols | Desempeño reconstructivo fuera de pliegue de las 8 cabezas residuales de texto: MSE, MAE, $R^2_{\mathrm{OOF}}$ entre pliegues, similitud coseno media ($0.52$ a $0.63$) y porcentaje de mejora vs. media. |
| `autoencoder_metricas_por_cabeza_pliegue.csv` | 75 filas $\times$ 12 cols | Desglose desagregado por pliegue de validación ($1$ a $5$) de las métricas de error y correlación para cada uno de los 15 cabezales del modelo. |
| `autoencoder_diagnosticos_espacio_latente.csv` | 5 filas $\times$ 9 cols | Diagnósticos geométricos del espacio latente de 64 dimensiones por pliegue: dimensiones activas ($\mathrm{SD} > 10^{-3}$), rango efectivo de Roy-Vetterli (media $35.4$), varianza explicada por los 5 primeros componentes principales y correlaciones de preservación geométrica. |

---

## 2. Inventario de figuras científicas y diagramas (`results/figures/`)

Se han incorporado **22 archivos de figuras** en alta resolución (PNG a 300 DPI), formato vectorial para escalabilidad editorial (SVG) y documentos vectoriales (PDF):

### Figuras de resultados del manuscrito (16 archivos)

- `figura1_cobertura_longitudinal.png` / `.svg`: Diagrama de barras con el horizonte temporal individual ($T=4$ a $19$) y curva de volumen acumulado de semanas ($N=446$).
- `figura2_dinamica_psicometrica.png` / `.svg`: Trayectorias longitudinales y matriz de dispersión bivariada con densidades marginales para las escalas clínicas ($N=256$).
- `figura3_espacio_semantico_familias.png` / `.svg`: Proyección bidimensional SVD de las narrativas semanales con elipses de dispersión y centroides de las familias F1, F2 y F3.
- `figura4_transiciones_markov.png` / `.svg`: Diagrama de estados y probabilidades de transición de Markov con tiempos geométricos de residencia.
- `figura5_metricas_complejidad.png` / `.svg`: Histogramas y diagramas de caja de los estimadores de dinámica temporal ($\rho_{\mathrm{Hurwicz}}$, entropía de permutación y exponente de Hurst, $N=31$).
- `figura6_red_sistemas_complejos.png` / `.svg`: Red de correlaciones parciales regularizadas entre variables psicométricas y dimensiones discursivas directas ($N=256$).
- `figura7_autoencoder_convergencia.png` / `.svg`: Curvas de aprendizaje de pérdida total, psicométrica y textual a lo largo de las épocas de entrenamiento en los 5 pliegues de validación.
- `figura8_autoencoder_ganancia_cabezales.png` / `.svg`: Gráfico de barras comparativo de la reducción relativa del error de reconstrucción frente a la media de entrenamiento en los 15 cabezales.

### Diagramas metodológicos rediseñados (`diagram-design`) (6 archivos)

- `diagrama1_arquitectura_mfca.pdf` / `.png` / `.svg`: Diagrama técnico de la arquitectura neuronal del Autoencoder Multimodal Dual Recalibrado MFCA-N1 (codificadores especializados, fusión latente en 64D y 15 cabezales decodificadores).
- `diagrama5_modelo_integrado_complejidad.pdf` / `.png` / `.svg`: Diagrama conceptual multiescala que sintetiza la dinámica de recuperación en adicciones como un sistema adaptativo complejo.

---

## 3. Delimitación de exclusión de datos primarios

En estricta observancia del principio de privacidad de datos clínicos, se deja constancia de que los siguientes archivos **NO forman parte de este repositorio**:
1. `narraciones_colab.sqlite` y `escalas_completas_items.sqlite` (bases SQLite con registros clínicos brutos).
2. `panel_canonico_446.csv` (panel longitudinal fila por fila con narrativas completas de participantes).
3. `embeddings_semanales.parquet` (vectores densos individuales de 768D por participante y semana).
4. `ae_presente_latents.parquet` (coordenadas latentes individuales de 64D por participante y semana).
5. Checkpoints binarios `.pt` de los modelos de redes neuronales entrenados.
