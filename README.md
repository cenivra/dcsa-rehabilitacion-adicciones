# DCSA: Dinámica Compleja y Sistemas Adaptativos en Trastornos por Uso de Sustancias

<p align="center">
  <img
    src="assets/dcsa_portada.jpg"
    alt="DCSA: visualización conceptual de sistemas clínicos dinámicos, trayectorias longitudinales, transiciones y representaciones multimodales"
    width="100%"
  />
</p>

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-Restringida-lightgrey.svg)](LICENSE.md)

Este repositorio alberga el código analítico, la configuración metodológica, la documentación reproducible y las tablas y figuras científicas agregadas del proyecto **Análisis de sistemas clínicos dinámicos** (**DCSA**) en rehabilitación de adicciones: índices de inestabilidad, transiciones y  resiliencia, enfocado en el modelado computacional y longitudinal de procesos de recuperación en personas con trastornos por uso de sustancias (TUS) en internamiento residencial.

---

## 1. Propósito científico y enfoque metodológico

El proyecto DCSA investiga la recuperación del trastorno por uso de sustancias no como un resultado estático o unipolar, sino como un **sistema adaptativo complejo** caracterizado por interacciones multiescala entre dos corrientes de información temporal:

1. **Monitoreo discursivo de alta frecuencia (semanal):** Narrativas clínicas abiertas guiadas por ocho dimensiones reflexivas estructuradas (descripción semanal, retos, afrontamiento, emociones, logros, reflexiones, metas y comentarios).
2. **Evaluación psicométrica quincenal de constructos clave:** Cuestionarios estandarizados que miden malestar cognitivo (ATQ-8), sintomatología de ansiedad (GAD-7), activación conductual y sus afectaciones asociadas (BADS en sus cuatro subescalas: Activación, Evitación/Rumia, Afectación Laboral y Afectación Social), recompensa ambiental percibida (EROS) y rigidez conductual (Índice de Rigidez Conductual ponderado, IRC-W).

El objetivo es caracterizar la estabilidad, la persistencia, los puntos de cambio estadístico y los patrones de acoplamiento contemporáneo y retardado entre las narrativas libres y el estado psicométrico reportado.

---

## 2. Vías analíticas del repositorio

El repositorio implementa dos vías metodológicas complementarias y reproducibles:

### Vía 1: Análisis directo pre-autoencoder (`src/pre_ae/`)

- **Script principal:** [`src/pre_ae/ejecutar_analisis_pre_ae_canonico.py`](src/pre_ae/ejecutar_analisis_pre_ae_canonico.py).
- **Representación textual interpretable:** Vectorización TF-IDF (unigramas y bigramas, max 500 términos) y reducción con *Truncated SVD* a 10 dimensiones.
- **Familias semánticas y dinámicas de Markov:** Agrupamiento *k-means* ($k=3$) que identifica tres colectores discursivos (*F1: Malestar/Desesperanza*, *F2: Afrontamiento/Metas*, *F3: Estabilidad/Reflexión*), matrices de transición global ($N=405$), temprana ($t \le 6$, $N=235$) y tardía ($t \ge 7$, $N=170$), junto con tiempos geométricos de permanencia.
- **Complejidad temporal de señales continuas:** Persistencia autorregresiva AR(1) con corrección de muestra finita de Hurwicz ($\rho_{\mathrm{corr}}$), entropía de permutación de Bandt-Pompe, exponente de Hurst y segmentación PELT ($N=31$ participantes con $T \ge 8$).
- **Modelos mixtos longitudinales (M1–M3):** Comparación de inercia pura (M1), predictores psicométricos multivariados rezagados (M2) y modelo híbrido aumentado con texto SVD y conductas de riesgo (M3) sobre 218 pares consecutivos activos ($40$ participantes).

### Vía 2: Autoencoder multimodal dual recalibrado MFCA-N1 (`src/autoencoder/`)

- **Cuaderno canónico:** [`src/autoencoder/MFCA_N1_Familias_Multihead_RECALIBRADO_FOCALIZADO_2026-10-23.ipynb`](src/autoencoder/MFCA_N1_Familias_Multihead_RECALIBRADO_FOCALIZADO_2026-10-23.ipynb).
- **Configuración de familias:** [`src/autoencoder/config/familias.json`](src/autoencoder/config/familias.json).
- **Arquitectura neuronal:** Cuello de botella latente de 64 dimensiones ($\mathbf{z}_t \in \mathbb{R}^{64}$) con decodificación en 15 cabezales (7 cabezas psicométricas sigmoideas en $[0, 1]$ y 8 cabezas residuales de texto ancladas en la media de entrenamiento).
- **Enmascaramiento y pérdida balanceada:** Pérdida cuadrática ponderada 50/50 y aplicación de máscara booleana estricta que excluye las semanas sin cuestionario activo (1,633 celdas observadas válidas), evitando corrupciones por ceros estructurales o indeterminaciones `NaN`.
- **Validación cruzada agrupada:** 5 pliegues *GroupKFold* por participante con parada temprana interna para evaluar métricas fuera de pliegue (*Out-of-Fold*, OOF) y diagnósticos espectrales de variedad latente.

---

## 3. Estructura del repositorio

```text
DCSA-reproducibilidad/
├── README.md                      # Presentación científica, estructura y guías del proyecto
├── .gitignore                     # Filtros de privacidad, datos clínicos y archivos pesados
├── requirements.txt               # Especificación de dependencias numéricas y librerías
├── docs/                          # Documentación metodológica y metadatos
│   ├── DESCRIPCION_MUESTRA.md     # Métricas agregadas de la cohorte y diseño observacional
│   ├── DATOS_Y_RESULTADOS.md      # Catálogo de tablas, figuras y artefactos canónicos
│   └── REPRODUCIBILIDAD.md        # Protocolo de ejecución, dependencias y trazabilidad
├── src/                           # Código fuente canónico y reproducible
│   ├── pre_ae/                    # Script canónico del análisis directo pre-autoencoder
│   │   └── ejecutar_analisis_pre_ae_canonico.py
│   └── autoencoder/               # Cuaderno y configuración del autoencoder MFCA-N1
│       ├── MFCA_N1_Familias_Multihead_RECALIBRADO_FOCALIZADO_2026-10-23.ipynb
│       └── config/
│           └── familias.json
└── results/                       # Salidas científicas agregadas reproducibles
    ├── tables/                    # 19 tablas agregadas en formato CSV (Pre-AE y Autoencoder)
    └── figures/                   # 22 figuras científicas y diagramas (PNG, SVG, PDF)
```

---

## 4. Instrucciones generales de reproducción

Para ejecutar los pipelines de análisis en un entorno local:

1. **Instalación de dependencias:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Configuración de rutas de datos:**  
   Los scripts están adaptados para resolver rutas de forma portable mediante variables de entorno cuando se disponga de acceso a las fuentes de datos primarias (restringidas por privacidad):
   - `DCSA_DATA_ROOT`: Directorio base donde se ubican las bases de datos originales.
   - `DCSA_SQLITE_NARR`: Ruta a la base SQLite con narrativas semanales.
   - `DCSA_SQLITE_ITEMS`: Ruta a la base SQLite con ítems psicométricos.
   - `DCSA_OUTPUT_DIR`: Directorio de destino de tablas y figuras generadas.
3. **Ejecución del pipeline Pre-AE:**

   ```bash
   python src/pre_ae/ejecutar_analisis_pre_ae_canonico.py
   ```

4. **Inspección del Autoencoder MFCA-N1:**  
   Abrir [`src/autoencoder/MFCA_N1_Familias_Multihead_RECALIBRADO_FOCALIZADO_2026-10-23.ipynb`](src/autoencoder/MFCA_N1_Familias_Multihead_RECALIBRADO_FOCALIZADO_2026-10-23.ipynb) en Jupyter Notebook o VS Code. El cuaderno se distribuye limpio de outputs de ejecuciones previas y contiene pruebas unitarias sintéticas (`ejecutar_pruebas_unitarias_sinteticas`) que pueden ejecutarse de manera autónoma sin necesidad de bases de datos.

---

## 5. Delimitación metodológica y limitaciones científicas

Es fundamental interpretar los resultados de este proyecto bajo sus justas delimitaciones científicas:

1. **Diseño observacional de un solo brazo:** El estudio no incluye un grupo control aleatorizado ni evalúa la eficacia de una intervención terapéutica específica frente a un estándar activo.
2. **Reconstrucción representacional frente a predicción clínica:** La capacidad del autoencoder MFCA-N1 para reconstruir variables psicométricas a través de un espacio latente común refleja integración geométrica multimodal; no constituye una predicción prospectiva fuera de muestra ni un biomarcador validado de recaída.
3. **Naturaleza exploratoria de los estados semánticos:** Las familias discursivas identificadas (F1, F2, F3) representan colectores estadísticos de uso léxico y prototipos discursivos; no constituyen categorías diagnósticas ni estados psicopatológicos formales.
4. **Muestra y contexto:** Los hallazgos corresponden a una cohorte clínica masculina en tratamiento residencial en Jalisco, México, por lo que su transferibilidad a otros contextos institucionales, modalidades ambulatorias o poblaciones femeninas/mixtas debe ser evaluada empíricamente.

---

## 6. Estado del proyecto y gobernanza de datos

Este repositorio contiene materiales de un **proyecto de investigación científica en desarrollo**. Por consideraciones éticas y de confidencialidad médica:

- **No se distribuyen registros narrativos individuales, citas textuales ni microdatos fila por fila.**
- Los datos compartidos se restringen a resultados agregados, descriptivos globales, matrices de asociación y métricas de desempeño computacional.
- La publicación, versión o distribución de los códigos y artefactos contenidos en este espacio permanece sujeta a la autorización expresa de los autores e investigadores responsables del proyecto.

---

## 7. Cómo citar este repositorio

Si utilizas este código o los resultados agregados en tu investigación, cita como:

> Análisis de sistemas clínicos dinámicos (DCSA) en rehabilitación de adicciones: índices de inestabilidad, transiciones y resiliencia. (2026).

Archivo de citación: [`CITATION.cff`](./CITATION.cff)

Formato BibTeX:

```bibtex
@software{dcsa-rehabilitacion-adicciones_2026,
  author = {Lauro Gutiérrez Castro},
  title = {Análisis de sistemas clínicos dinámicos (DCSA) en rehabilitación de adicciones: índices de inestabilidad, transiciones y resiliencia},
  year = {2026},
  url = {https://github.com/cenivra/dcsa-rehabilitacion-adicciones}
}
```

---

## 8. Contacto y contribución

**Mantenimiento:** Equipo investigador del proyecto DCSA.

**Acceso a datos restringidos:** Solicita autorización ante los investigadores responsables del proyecto.

**Contribuciones:** Abre un issue para reportar errores o proponer cambios. Los pull requests son bienvenidos previo consenso con el equipo mantenedor.

**Código de conducta:** Se espera respeto y rigor académico en todas las interacciones.
