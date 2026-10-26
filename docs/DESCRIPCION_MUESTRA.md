# Descripción de la Muestra y Diseño Observacional

Este documento presenta la caracterización agregada y el diseño metodológico de la cohorte analizada en el proyecto **DCSA**. Todos los datos se reportan a nivel de grupo para salvaguardar el anonimato y la confidencialidad de los participantes.

---

## 1. Características generales de la cohorte clínica

- **Tamaño de la cohorte:** $N = 41$ participantes adultos en tratamiento residencial.
- **Identidad de género:** 100% masculino ($41/41$).
- **Distribución de edad:**
  - Media: $33.98$ años.
  - Desviación estándar (DE): $11.64$ años.
  - Mediana: $31.0$ años.
  - Rango: $18$ a $60$ años.
- **Contexto asistencial:** Centro de rehabilitación y tratamiento residencial para trastornos por uso de sustancias (TUS) ubicado en el estado de Jalisco, México.
- **Criterios generales de elegibilidad:**
  - Pacientes adultos ( $\ge 18$ años) ingresados para tratamiento residencial voluntario por TUS.
  - Capacidad para comprender los procedimientos del estudio y consentir su participación.
  - Disponibilidad de lectoescritura suficiente para responder semanalmente a las preguntas estructuradas y a las escalas psicométricas mediante formularios digitales.
  - Ausencia de deterioro cognitivo grave o estados psicóticos activos que impidieran la interacción con el protocolo.

---

## 2. Diseño longitudinal y multiescala

El protocolo combinó dos resoluciones temporales complementarias para modelar la dinámica de recuperación:

1. **Monitoreo discursivo semanal (alta frecuencia):** Respuestas reflexivas breves a 8 preguntas guía estructuradas:
   - Q1: Descripción general de la semana.
   - Q2: Desafíos o dificultades experimentadas.
   - Q3: Estrategias de afrontamiento implementadas.
   - Q4: Emociones predominantes.
   - Q5: Progresos, logros o avances percibidos.
   - Q6: Reflexiones personales y aprendizajes.
   - Q7: Metas para la semana siguiente.
   - Q8: Comentarios adicionales (sección optativa).
2. **Evaluación psicométrica quincenal (baja frecuencia):** Batería de cuestionarios administrados aproximadamente cada 14 días:
   - **ATQ-8 (Automatic Thoughts Questionnaire - 8 items):** Frecuencia de pensamientos automáticos negativos asociados a malestar cognitivo y depresión (rango teórico: 8 a 40).
   - **GAD-7 (Generalized Anxiety Disorder - 7 items):** Severidad de sintomatología de ansiedad generalizada (rango teórico: 0 a 21).
   - **BADS (Behavioral Activation for Depression Scale):** Activación conductual y afectación funcional dividida en cuatro subescalas:
     - *Activación Conductual* (7 ítems; rango teórico: 0 a 42).
     - *Evitación/Rumia* (8 ítems; rango teórico: 0 a 48).
     - *Afectación Trabajo/Escuela* (5 ítems; rango teórico: 0 a 30).
     - *Afectación Social* (5 ítems; rango teórico: 0 a 30).
   - **EROS (Environmental Reward Observation Scale):** Grado de recompensa ambiental percibida y acceso a reforzadores placenteros (10 ítems; rango teórico: 10 a 40).
   - **IRC-W (Índice de Rigidez Conductual ponderado):** Índice compuesto estandarizado derivado de los componentes de evitación, rumiación y déficit de activación ($0.58 \times \text{rumia} + 0.63 \times Z_{\text{evitación}} - 0.52 \times Z_{\text{activación}}$).

---

## 3. Cobertura longitudinal y métricas temporales

- **Total de semanas observadas en el panel analítico:** $446$ semanas.
- **Horizonte de seguimiento por participante:**
  - Mediana: $11.0$ semanas.
  - Rango intercuartílico (IQR): $[8.0, 14.0]$ semanas.
  - Rango total: $4$ a $19$ semanas de internamiento.
- **Submuestra de series largas ($T \ge 8$ semanas):**
  - $31$ participantes ($75.6\%$ de la cohorte).
  - Aportan $387$ semanas observadas ($86.8\%$ del volumen longitudinal total).
  - Este subconjunto se utilizó como base para los estimadores de persistencia temporal (AR(1) con corrección de Hurwicz, entropía de permutación y exponente de Hurst).
- **Volumen discursivo semanal:**
  - Media de palabras por narrativa consolidada: $121.9 \pm 78.1$ palabras.
  - Mediana: $102.5$ palabras (IQR: $[65.0, 163.2]$).
  - Cobertura de respuesta superior al $95\%$ en 7 de las 8 preguntas guía estructuradas ($76.5\%$ en la sección optativa Q8).

---

## 4. Trazabilidad de denominadores analíticos

Para asegurar rigor metodológico y transparencia computacional, los análisis no utilizan un denominador arbitrario o idéntico para todos los cálculos, sino que respetan la naturaleza de cada medición:

| Conjunto analítico / Procedimiento | Denominador ($N$) | Descripción y justificación metodológica |
| :--- | :--- | :--- |
| **Panel longitudinal narrativo (NLP)** | $446$ semanas ($41$ participantes) | Total de semanas válidas con texto clínico consolidado analizadas mediante TF-IDF, Truncated SVD y asignación de familias semánticas. |
| **Dinámica estocástica de Markov** | $405$ transiciones consecutivas | Transiciones semanales adyacentes ($t \to t+1$) observadas dentro del horizonte longitudinal de los participantes ($235$ en tramo temprano $t \le 6$; $170$ en tramo tardío $t \ge 7$). |
| **Dinámica de series temporales** | $31$ participantes ($387$ semanas) | Restringido a participantes que alcanzaron el umbral mínimo operativo de estabilidad temporal ($T \ge 8$). |
| **Evaluaciones clínicas basales (Semana 1)** | $n = 39$ participantes ($n = 38$ para IRC-W) | Primera evaluación psicométrica activa registrada en el ingreso al centro residencial (2 participantes no completaron batería en la semana 1). |
| **Ocasiones con evaluación psicométrica activa** | $263$ semanas ($41$ participantes) | Semanas con administración efectiva de escalas en el panel longitudinal, tras descartar $27$ semanas sin registro y $156$ semanas intermedias sin administración (ceros estructurales). |
| **Descriptivos globales de BADS (4 subescalas) y EROS** | $N = 263$ ocasiones | Todas las ocasiones activas disponen de administración completa verificada a nivel de ítem. |
| **Descriptivos globales de GAD-7** | $N = 262$ ocasiones | Excluye 1 ocasión de administración parcial (N40 semana 10) donde únicamente se aplicaron BADS y EROS. |
| **Descriptivos globales de ATQ-8** | $N = 260$ ocasiones | Excluye 3 ocasiones de administración parcial donde no se administró ATQ-8 (N35 sem 7, N36 sem 11 y N40 sem 10). |
| **Descriptivos globales de IRC-W** | $N = 258$ ocasiones | Ocasiones con puntuaciones z estandarizadas completas para el cálculo ponderado. |
| **Matriz de correlaciones contemporáneas** | $N = 256$ ocasiones | Intersección de casos completos que cuentan simultáneamente con las 6 variables psicométricas analizadas. |
| **Modelos mixtos longitudinales (M1–M3)** | $N = 218$ pares consecutivos ($40$ participantes) | Pares de evaluaciones psicométricas activas consecutivas con rezago temporal válido ($t-1 \to t$). |
| **Entrenamiento y validación del autoencoder MFCA-N1** | $1\,633$ celdas escala-semana observadas | Celdas psicométricas observadas empalmadas en ventana $\pm 7$ días (BADS: 236 c/u; EROS: 233; GAD-7: 229; ATQ-8: 227), procesadas bajo máscara binaria estricta. |

---

## 5. Consideraciones éticas y no reidentificación

En estricto cumplimiento con los principios de bioética médica y la normativa aplicable a investigaciones clínicas observacionales:

- **Ningún nombre propio, inicial, identificador institucional o dato personal sensible forma parte de este repositorio.**
- Los identificadores utilizados en el pipeline interno son códigos alfanuméricos arbitrarios (`N1` a `N41`) que no guardan relación con números de expediente clínico.
- Las semanas temporales son relativas al ingreso individual ($t=1, 2, \dots$) y no contienen fechas de calendario gregoriano.
- No se incluyen citas narrativas textuales literales que pudieran contener detalles biográficos o contextuales conducentes a reidentificación por terceros.
