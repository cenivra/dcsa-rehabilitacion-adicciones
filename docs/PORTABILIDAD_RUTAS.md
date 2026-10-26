# Manifiesto de Portabilidad y Estandarización de Rutas

Este documento registra las modificaciones de infraestructura implementadas en el repositorio **DCSA-reproducibilidad** para eliminar dependencias rígidas de rutas absolutas locales, suprimir nombres de usuario y rutas personales del entorno de desarrollo de origen, y garantizar la portabilidad del código en múltiples plataformas (Windows, Linux, macOS y Google Colab).

---

## 1. Principios de portabilidad y arquitectura de configuración

Para asegurar que cualquier investigador independiente pueda ejecutar los análisis en su propio entorno sin modificar el código fuente:

1. **Resolución relativa por defecto:** Los scripts y notebooks calculan sus rutas a partir de la ubicación de su propio archivo fuente (`Path(__file__).resolve().parents[...]` o `Path.cwd()`).
2. **Sobreescritura jerárquica mediante variables de entorno:** Se establecieron nombres de variables homogéneos que permiten redirigir la raíz del proyecto, las carpetas de datos y las rutas de salida sin alterar los archivos de código.
3. **Validación explícita de entradas:** En lugar de fallos crípticos o uso silencioso de rutas antiguas, el código verifica la existencia de los archivos requeridos y emite un mensaje de error claro indicando la variable de configuración necesaria.
4. **Plantilla de configuración sin secretos (`.env.example`):** Se incorporó un archivo de ejemplo documentando las variables configurables.

---

## 2. Inventario de variables de entorno adoptadas

| Variable de entorno | Valor predeterminado (fallback) | Función y componente asociado |
| :--- | :--- | :--- |
| `DCSA_PROJECT_ROOT` | Raíz relativa del repositorio (`../../`) | Directorio base de `DCSA-reproducibilidad`. Utilizado por scripts y notebooks. |
| `DCSA_DATA_DIR` | `$DCSA_PROJECT_ROOT/../datos` | Directorio de entrada donde residen las bases SQLite primarias. |
| `DCSA_RESULTS_DIR` | `$DCSA_PROJECT_ROOT/results` | Directorio raíz para la exportación de tablas, figuras y diagnósticos. |
| `DCSA_SQLITE_NARR` | `$DCSA_DATA_DIR/colab/narraciones_colab.sqlite` | Base de datos SQLite con las 446 narraciones semanales estructuradas. |
| `DCSA_SQLITE_ITEMS`| `$DCSA_DATA_DIR/escalas_completas_items.sqlite`| Base de datos SQLite con las respuestas psicométricas a nivel de ítem. |
| `DCSA_FAMILIES_CONFIG`| `src/autoencoder/config/familias.json` | Archivo JSON con la configuración de escalas clínicas y preguntas guía. |
| `DCSA_EMBEDDINGS_PATH`| `$DCSA_RESULTS_DIR/embeddings_semanales.parquet` | Archivo Parquet con embeddings semánticos de 768 dimensiones. |
| `DCSA_AE_OUTPUT_DIR`| `$DCSA_RESULTS_DIR/autoencoder` | Directorio específico de artefactos derivados del autoencoder MFCA-N1. |
| `DCSA_MODO` | `"normal"` | Modo de ejecución del notebook (`"smoke"` para prueba rápida, `"normal"` para completo). |
| `DCSA_ARQ` | `"multimodal_dual_recalibrado"` | Selección de la arquitectura neuronal del autoencoder. |
| `DCSA_MOUNT_DRIVE` | `0` | En Google Colab, bandera para montar `/content/drive` (`1` = montar, `0` = no montar). |

---

## 3. Archivos intervenidos y tipos de rutas sustituidas

### 3.1. `src/pre_ae/ejecutar_analisis_pre_ae_canonico.py`
- **Ruta original sustituida:** Asignación fija `RAIZ = r"C:\Users\saraq\Mi unidad\DCSA_SISTEMAS_COMPLEJOS"` y subdirectorios construidos rígidamente hacia carpetas locales de Windows.
- **Modificación implementada:**
  - `RAIZ_REPO` se resuelve dinámicamente mediante `Path(__file__)`.
  - Se añadieron resolutores basados en `DCSA_PROJECT_ROOT`, `DCSA_DATA_DIR`, `DCSA_RESULTS_DIR`, `DCSA_SQLITE_NARR`, `DCSA_SQLITE_ITEMS` y `DCSA_FAMILIAS_JSON`.
  - Se incorporaron aserciones previas con mensajes de error explícitos si los archivos SQLite requeridos no están disponibles en la ruta configurada.

### 3.2. `src/autoencoder/MFCA_N1_Familias_Multihead_RECALIBRADO_FOCALIZADO_2026-10-23.ipynb`
- **Rutas originales sustituidas:**
  - Celda 4: Lista rígida de candidatos que incluía `Path(r"C:\Users\saraq\Mi unidad\DCSA_SISTEMAS_COMPLEJOS")` y `Path("/content/drive/MyDrive/DCSA_SISTEMAS_COMPLEJOS")`, junto con intento ciego de montar Drive sin control del usuario.
  - Celda 10: Rutas fijas hacia `RAIZ / "datos/..."` y hacia `RAIZ / "src/mfca_n1_autoencoder/familias.json"`.
  - Celda 12: Ruta estática `RAIZ / "results/DCSA_octubre/embeddings_semanales.parquet"`.
- **Modificación implementada:**
  - Celda 4: Se eliminaron todas las rutas personales y fijas de Windows y Google Drive. Se implementó resolución prioritaria vía `DCSA_DATA_ROOT` o `DCSA_PROJECT_ROOT`, seguida de búsqueda relativa ascendente no rígida. La función `_montar_drive_si_colab()` ahora respeta la bandera condicional `DCSA_MOUNT_DRIVE`.
  - Celda 10: Rutas de bases SQLite y de salida configurables mediante `DCSA_DATA_DIR`, `DCSA_SQLITE_ITEMS`, `DCSA_SQLITE_NARR` y `DCSA_AE_OUTPUT_DIR`. `familias.json` se resuelve con preferencia local en `src/autoencoder/config/familias.json` o mediante `DCSA_FAMILIES_CONFIG`.
  - Celda 12: Embeddings configurables mediante `DCSA_EMBEDDINGS_PATH` con validación de existencia `PQ_EMB.is_file()` y excepción descriptiva.
  - Celdas de código: Verificadas con `outputs: []` y `execution_count: null`.

### 3.3. `docs/REPRODUCIBILIDAD.md`
- **Texto original sustituido:** Mención a rutas absolutas personales en ejemplos textuales (`C:\Users\saraq\...`).
- **Modificación implementada:** Redacción genérica centrada en principios de diseño ("rutas absolutas locales de un entorno particular como nombres de usuario o unidades fijas").

---

## 4. Guía de configuración por entorno

### 4.1. En Windows (PowerShell)
```powershell
$env:DCSA_PROJECT_ROOT = "C:\ruta\a\DCSA-reproducibilidad"
$env:DCSA_DATA_DIR = "C:\ruta\a\datos_clinicos"
$env:DCSA_EMBEDDINGS_PATH = "C:\ruta\a\embeddings_semanales.parquet"
python src\pre_ae\ejecutar_analisis_pre_ae_canonico.py
```

### 4.2. En Linux / macOS (Bash / Zsh)
```bash
export DCSA_PROJECT_ROOT="/home/usuario/DCSA-reproducibilidad"
export DCSA_DATA_DIR="/home/usuario/datos_clinicos"
export DCSA_EMBEDDINGS_PATH="/home/usuario/embeddings_semanales.parquet"
python src/pre_ae/ejecutar_analisis_pre_ae_canonico.py
```

### 4.3. En Google Colab
Si se dispone de acceso a los datos originales en Google Drive:
```python
import os
os.environ["DCSA_MOUNT_DRIVE"] = "1"
os.environ["DCSA_DATA_ROOT"] = "/content/drive/MyDrive/ruta_autorizada_datos"
os.environ["DCSA_EMBEDDINGS_PATH"] = "/content/drive/MyDrive/ruta_autorizada_datos/embeddings_semanales.parquet"
```
Si se ejecuta el notebook únicamente con fines de verificación sintética, las pruebas de la Celda 8 (`ejecutar_pruebas_unitarias_sinteticas()`) se ejecutan sin requerir datos externos ni montaje de almacenamiento.

---

## 5. Auditoría final de limpieza y privacidad

Una auditoría estricta mediante expresiones regulares sobre todos los archivos del repositorio confirmó:
- **0 coincidencias** de rutas absolutas de Windows (`C:\...`).
- **0 coincidencias** de nombres de usuario o carpetas de sincronización personal.
- **0 coincidencias** de rutas fijas de Google Drive (`/content/drive/...`).
- **0 secretos, tokens o credenciales** expuestas.
- Las tablas en `results/tables/` son 100% agregadas y no contienen identificadores individuales ni narraciones de pacientes.
