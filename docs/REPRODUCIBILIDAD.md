# Protocolo de Reproducibilidad Computacional y Pipelines Canónicos

Este documento describe la trazabilidad técnica, las dependencias de software, los puntos de entrada analíticos, la portabilidad de rutas y los procedimientos de validación implementados en el repositorio **DCSA-reproducibilidad**.

---

## 1. Entorno de software y versiones verificadas

Los análisis han sido ejecutados y validados en un entorno estándar de computación científica:

### Intérprete y dependencias principales (`requirements.txt`)
- **Python:** Versión `3.10` a `3.14` (64-bit).
- `numpy`: $\ge 1.24.0, < 2.0.0$ (operaciones tensoriales y álgebra lineal).
- `scipy`: $\ge 1.10.0$ (estadística inferencial y pruebas de hipótesis).
- `pandas`: $\ge 2.0.0$ (estructuras tabulares y series temporales).
- `pyarrow`: $\ge 14.0.0$ (manejo eficiente de columnas vectoriales).
- `scikit-learn`: $\ge 1.3.0$ (vectorización TF-IDF, TruncatedSVD, *k-means*, *GroupKFold*).
- `statsmodels`: $\ge 0.14.0$ (modelos lineales mixtos y pruebas asintóticas de Wald).
- `torch` (PyTorch): $\ge 2.0.0$ (entrenamiento de redes neuronales profundas).
- `matplotlib`: $\ge 3.7.0$ y `seaborn`: $\ge 0.12.0$ (gráficos científicos).
- `networkx`: $\ge 3.0$ (análisis y visualización de redes de sistemas complejos).

### Control determinista de procesos estocásticos
- Semilla global en análisis pre-AE: `seed = 2025`
- Semilla de partición y optimización en autoencoder: `seed = 20261023`

---

## 2. Portabilidad de rutas y configuración de variables de entorno

Para evitar dependencias rígidas de rutas absolutas locales de un entorno particular (como rutas con nombres de usuario o unidades de disco fijas), las fuentes copiadas a `src/` han sido adaptadas para resolver sus rutas de forma relativa al propio repositorio o mediante variables de entorno:

| Variable de entorno | Valor predeterminado (fallback) | Descripción |
| :--- | :--- | :--- |
| `DCSA_DATA_ROOT` | Directorio padre del repositorio (`../../`) | Raíz donde se ubica el proyecto original y sus bases de datos clínicas. |
| `DCSA_OUTPUT_DIR` | `results/` relativo a la raíz del repositorio | Directorio de destino para tablas y figuras exportadas. |
| `DCSA_SQLITE_NARR` | `$DCSA_DATA_ROOT/datos/colab/narraciones_colab.sqlite` | Base SQLite con las narrativas semanales estructuradas. |
| `DCSA_SQLITE_ITEMS`| `$DCSA_DATA_ROOT/datos/escalas_completas_items.sqlite`| Base SQLite con respuestas a nivel de ítem psicométrico. |
| `DCSA_FAMILIAS_JSON`| `src/autoencoder/config/familias.json` | Archivo JSON con la configuración de escalas y preguntas guía. |
| `DCSA_MODO` | `"normal"` | Modo de ejecución del notebook (`"smoke"` para prueba rápida de 2 épocas, `"normal"` para entrenamiento completo). |

---

## 3. Pipeline de análisis pre-autoencoder (`src/pre_ae/`)

- **Script principal:** [`src/pre_ae/ejecutar_analisis_pre_ae_canonico.py`](src/pre_ae/ejecutar_analisis_pre_ae_canonico.py) (1,177 líneas).
- **Procedencia:** Copia directa adaptada para portabilidad desde `src/DCSA_AMEDYSC/ejecutar_analisis_pre_ae_canonico.py`.

### Flujo de etapas analíticas:
1. **Auditoría de denominadores y construcción del panel:**  
   Lee la tabla `panel` (451 registros), excluye 4 filas de `fila_pendiente` y resuelve la duplicidad temporal de N33, consolidando el panel analítico de 446 semanas observadas de 41 participantes.
2. **Filtrado de evaluaciones psicométricas activas:**  
   Excluye las 27 semanas sin registro psicométrico y las 156 semanas con ceros estructurales, identificando 263 ocasiones con evaluación activa. Calcula descriptivos por instrumento ($N=263$ para BADS y EROS, $N=262$ para GAD-7, $N=260$ para ATQ-8, $N=258$ para IRC-W).
3. **Asociaciones contemporáneas:**  
   Calcula la matriz de Pearson y Spearman con intervalos de confianza del 95% sobre la intersección de casos completos ($N=256$).
4. **Vectorización textual y agrupamiento semántico:**  
   Aplica `TfidfVectorizer(max_features=500, ngram_range=(1,2))` y `TruncatedSVD(n_components=10)`. Ajusta `KMeans(n_clusters=3)` generando las familias F1, F2 y F3.
5. **Modelado de Markov y tiempos de residencia:**  
   Computa matrices de transición empíricas global ($N=405$), temprana ($N=235$) y tardía ($N=170$), junto con tiempos de permanencia geométrica.
6. **Métricas de complejidad en series temporales:**  
   En participantes con $T \ge 8$ ($N=31$), estima $\rho_{\mathrm{Hurwicz}}$, entropía de permutación ($m=3, \tau=1$), exponente de Hurst y puntos de cambio PELT.
7. **Modelos lineales mixtos longitudinales:**  
   Ajusta M1, M2 y M3 mediante `statsmodels` sobre 218 pares consecutivos activos, extrayendo AIC, BIC, RMSE de ajuste muestral y coeficientes de Wald.

---

## 4. Pipeline del autoencoder multimodal MFCA-N1 (`src/autoencoder/`)

- **Cuaderno principal:** [`src/autoencoder/MFCA_N1_Familias_Multihead_RECALIBRADO_FOCALIZADO_2026-10-23.ipynb`](src/autoencoder/MFCA_N1_Familias_Multihead_RECALIBRADO_FOCALIZADO_2026-10-23.ipynb).
- **Configuración metodológica:** [`src/autoencoder/config/familias.json`](src/autoencoder/config/familias.json).
- **Procedencia:** Copia directa adaptada desde `src/DCSA_AMEDYSC/DCSA_octubre/ae_multihead/MFCA_N1_Familias_Multihead_RECALIBRADO_FOCALIZADO_2026-10-23.ipynb`.
- **Estado de outputs:** Todas las celdas de código se distribuyen limpias (`outputs: []` y `execution_count: null`) para asegurar que el cuaderno funcione como código reproducible y no como volcado de ejecución previa.

### Componentes de diseño del Autoencoder:
1. **Entrada dual y representación:**  
   - Texto: Vector de 768 dimensiones obtenido promediando los embeddings de las preguntas respondidas.
   - Psicometría: Vector de 14 dimensiones (7 puntajes continuos escalados en $[0, 1]$ concatenados con 7 máscaras binarias de disponibilidad).
2. **Fusión en espacio latente:**  
   Codificador textual ($768 \to 256$) y psicométrico ($14 \to 32$) fusionados mediante proyección lineal con LayerNorm a un cuello de botella de 64 dimensiones ($\mathbf{z}_t \in \mathbb{R}^{64}$).
3. **Decodificación en 15 cabezales especializados:**  
   - 7 cabezas psicométricas con activación Sigmoid para acotar predicciones en $[0, 1]$.
   - 8 cabezas textuales residuales ancladas en la media de entrenamiento del pliegue $\mu_{\mathrm{base}}$ ($64 \to 256 \to 768$).
4. **Función de pérdida balanceada y blindada ante ausencias:**  
   Ponderación $w_{\mathrm{fam}} = 3.0$ y $w_{\mathrm{txt}} = 100.0$ con regularización temporal de suavidad ($\lambda = 0.05$). Enmascaramiento condicional mediante `torch.where` y máscaras booleanas que garantizan gradiente nulo ($0.00$) ante datos ausentes y anulan el riesgo de corrupción por `NaN * 0.0`.
5. **Validación cruzada agrupada (*GroupKFold*, 5 pliegues):**  
   Partición externa de 5 pliegues por participante con validación interna del 20% para parada temprana (*early stopping*, paciencia 15 épocas).

---

## 5. Pruebas unitarias sintéticas autónomas

El cuaderno del autoencoder incluye en su Celda 8 la función `ejecutar_pruebas_unitarias_sinteticas()`, la cual puede ejecutarse **sin necesidad de contar con las bases de datos originales ni con GPU**:

- **Test 1:** Invarianza exacta de la función de pérdida ante perturbaciones extremas en posiciones con máscara `0`.
- **Test 1b:** Invarianza y gradiente exactamente nulo ante tensores con `NaN` crudos en observaciones ausentes.
- **Test 2:** Normas finitas y acotadas ($< 5.0$) en predicciones textuales residuales.
- **Test 3:** Acotamiento estricto en $[0, 1]$ de todas las salidas psicométricas sigmoideas.
- **Test 4:** Paridad dimensional de tensores latentes ($(B, T, 64)$) y de decodificación ($(B, T, 7)$ y $(B, T, 8, 768)$).

---

## 6. Delimitación de reproducibilidad de extremo a extremo

Es imprescindible documentar que la ejecución completa de extremo a extremo de estos pipelines requiere acceso autorizado a:
1. `datos/colab/narraciones_colab.sqlite`
2. `datos/escalas_completas_items.sqlite`
3. `results/DCSA_octubre/embeddings_semanales.parquet`

Dichos archivos contienen microdatos individuales de pacientes en tratamiento residencial y se encuentran legítimamente restringidos por motivos bioéticos y normativos de confidencialidad médica. No obstante, las 19 tablas agregadas en `results/tables/` y las 22 figuras en `results/figures/` permiten auditar y verificar íntegramente todas las cifras, coeficientes, diagnósticos y modelos presentados en el manuscrito científico.
