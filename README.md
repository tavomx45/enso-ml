# ENSO-ML: Pronóstico estacional de precipitación en la costa de Ecuador

Proyecto de Machine Learning orientado a estudiar la relación entre **ENSO (El Niño–Southern Oscillation)** y la **precipitación estacional en una región costera seleccionada de Ecuador**, utilizando índices climáticos históricos, reanálisis **ERA5-Land**, validación temporal y una aplicación web desarrollada con **Streamlit**.

> **Estado actual del proyecto:** el modelo académico fue validado con datos históricos y posteriormente se construyó una extensión operacional con datos actualizados hasta **agosto de 2026**, capaz de generar un outlook para **septiembre–noviembre de 2026**.

---

## Demo

La aplicación fue desplegada con **Streamlit Community Cloud**.

> Añadir aquí la URL pública de la app:
>
> `https://enso-ml-tavomx45.streamlit.app/'

---

## Objetivo del proyecto

La pregunta inicial fue:

> **¿Pueden los indicadores ENSO ayudar a anticipar la precipitación futura en la costa de Ecuador?**

El proyecto evolucionó desde una predicción mensual puntual a una formulación estacional, después de comprobar experimentalmente que la precipitación de un único mes a tres meses de horizonte contenía demasiado ruido para ser modelada de forma consistente.

El objetivo final quedó definido como:

> **Predecir la anomalía media de precipitación durante los próximos tres meses.**

Matemáticamente:

$
y_t =
\frac{
A_{t+1} + A_{t+2} + A_{t+3}
}{3}
$

donde \(A_t\) representa la anomalía mensual de precipitación respecto de una climatología de referencia.

---

# 1. Datos utilizados

## 1.1 Índices ENSO históricos

Durante la fase académica se trabajó con:

- Niño 1+2
- Niño 3
- Niño 3.4
- Niño 4
- SOI
- TNI
- MEI.v2 y ONI como variables exploradas durante el análisis inicial

Las series fueron limpiadas, normalizadas a frecuencia mensual y unificadas por fecha.

Los valores sentinela utilizados para representar datos faltantes, como `-999`, se transformaron a `NaN`.

---

## 1.2 ERA5-Land

La precipitación y otras variables climáticas se obtuvieron de **ERA5-Land**.

Periodo histórico principal:

```text
1950-01 → 2025-12
```

Región espacial aproximada:

```text
Norte:  1.5
Sur:   -5.0
Oeste: -81.5
Este:  -79.0
```

El dominio representa una región costera / occidental de Ecuador y no una división administrativa exacta.

Se trabajó inicialmente con:

```text
total_precipitation
2m_temperature
2m_dewpoint_temperature
10m_u_component_of_wind
10m_v_component_of_wind
surface_pressure
total_evaporation
runoff
volumetric_soil_water_layer_1
volumetric_soil_water_layer_2
volumetric_soil_water_layer_3
volumetric_soil_water_layer_4
```

Después del análisis experimental, el modelo final dejó de utilizar la mayoría de las variables atmosféricas e hidrológicas adicionales.

---

# 2. Preparación de la precipitación

La precipitación ERA5-Land fue convertida a milímetros mensuales.

Para el producto mensual utilizado originalmente:

\[
P_{mm} =
P_{ERA5}
\times 1000
\times días\ del\ mes
\]

Después se calculó un promedio espacial sobre las celdas terrestres válidas de la región seleccionada.

La serie histórica resultante contiene observaciones mensuales continuas.

---

# 3. ¿Por qué usar anomalías?

La precipitación costera ecuatoriana presenta una fuerte estacionalidad.

Los meses lluviosos y secos tienen comportamientos climatológicos muy diferentes. Si el modelo utilizara directamente precipitación absoluta, podría aprender simplemente el calendario.

Por ello se definió:

\[
A_t =
P_t -
\bar{P}_{mes}
\]

donde:

- \(P_t\) es la precipitación observada;
- \(\bar{P}_{mes}\) es la climatología histórica para ese mes.

Interpretación:

```text
A = 0      → condiciones cercanas a lo normal
A > 0      → más húmedo que lo normal
A < 0      → más seco que lo normal
```

La climatología quedó fijada usando:

```text
1950-01 → 2010-12
```

Esto evita utilizar información futura durante la evaluación histórica y mantiene una definición estable de "normal".

---

# 4. Primer problema supervisado

La primera formulación fue:

\[
y_t = A_{t+3}
\]

Es decir:

> predecir la anomalía de precipitación de un mes exacto a tres meses de distancia.

Se probaron diferentes grupos de variables.

## Experimento A — ENSO

Se utilizaron:

- seis índices ENSO actuales;
- lags de 1, 2 y 3 meses;
- codificación estacional del mes objetivo.

## Experimento B — ENSO + precipitación reciente

Se añadieron:

```text
precip_anomaly
precip_anomaly_lag1
precip_anomaly_lag2
precip_anomaly_lag3
```

## Experimento C — ENSO + precipitación + atmósfera

Se incorporaron variables como:

```text
temperatura
punto de rocío
viento
presión
```

## Experimento D — atmósfera + hidrología

Se añadieron:

```text
evaporación
runoff
humedad del suelo
```

El experimento más grande llegó aproximadamente a **74 features**.

---

# 5. ¿Por qué usar lags?

ENSO presenta una fuerte autocorrelación temporal.

Por ejemplo, Niño 3.4 mostró aproximadamente:

```text
lag 1 ≈ 0.93
lag 2 ≈ 0.85
lag 3 ≈ 0.76
lag 6 ≈ 0.44
```

Los lags permiten que el modelo observe no solo el estado actual de ENSO, sino también su evolución reciente.

Sin embargo, aumentar el número de lags también introdujo:

- alta correlación entre variables;
- mayor dimensionalidad;
- más riesgo de sobreajuste.

---

# 6. Estacionalidad cíclica

El mes del año es una variable circular.

Diciembre y enero son meses consecutivos, aunque numéricamente sean `12` y `1`.

Por ello se utilizaron:

\[
\sin\left(\frac{2\pi m}{12}\right)
\]

y:

\[
\cos\left(\frac{2\pi m}{12}\right)
\]

Esto permite representar correctamente la estructura cíclica anual.

---

# 7. Resultado del primer target

La predicción mensual puntual a \(t+3\) no produjo resultados suficientemente buenos.

Ejemplo de validación:

```text
Climatología RMSE ≈ 62.89
Linear      RMSE ≈ 62.53
Ridge       RMSE ≈ 62.81
```

Los modelos más complejos, como Random Forest y Gradient Boosting, tendieron a empeorar por sobreajuste.

La conclusión fue:

> añadir más variables no resolvía el problema.

El principal problema era la **formulación del target**.

---

# 8. Cambio de target

ENSO es un fenómeno que evoluciona en escalas de meses y estaciones.

Intentar predecir un único mes exacto resultaba demasiado sensible al ruido meteorológico local.

Por ello el target fue reformulado como:

\[
y_t =
\frac{
A_{t+1} + A_{t+2} + A_{t+3}
}{3}
\]

En lugar de preguntar:

> ¿cuánto se desviará la precipitación exactamente dentro de tres meses?

la nueva pregunta fue:

> **¿los próximos tres meses tenderán a ser más húmedos o más secos de lo normal?**

Este cambio produjo una mejora clara en los resultados.

---

# 9. Feature Engineering final S2

La ingeniería de variables final redujo el número de features y priorizó información temporal más compacta.

Para cada índice ENSO se utilizó:

```text
valor actual
media móvil de 3 meses
```

La media móvil se define como:

\[
MA3_t =
\frac{
x_t + x_{t-1} + x_{t-2}
}{3}
\]

Esto resume el estado reciente de ENSO sin crear múltiples lags altamente correlacionados.

El conjunto S2 final contiene **16 features**:

```text
nino12
nino12_ma3

nino3
nino3_ma3

nino34
nino34_ma3

nino4
nino4_ma3

soi
soi_ma3

tni
tni_ma3

target_month_sin
target_month_cos

precip_anomaly
precip_anomaly_ma3
```

---

# 10. Validación temporal

El proyecto evita realizar splits aleatorios porque se trabaja con series de tiempo.

Durante la fase principal se utilizaron periodos cronológicos:

```text
Train       → hasta 2010
Validation  → 2011–2017
Test        → 2018+
```

Posteriormente se utilizó:

```python
TimeSeriesSplit(n_splits=5)
```

para comparar modelos sobre la historia pre-2018.

En la versión operacional se hizo una validación más estricta:

```python
TimeSeriesSplit(
    n_splits=5,
    gap=2,
)
```

El `gap=2` ayuda a separar las ventanas futuras utilizadas por el target estacional.

---

# 11. Resultados del modelo estacional

En validación 2011–2017:

```text
S2 Linear Regression
MAE  ≈ 31.71
RMSE ≈ 42.52
R²   ≈ 0.172

Climatología
RMSE ≈ 48.55
R²   ≈ -0.080
```

La reformulación estacional consiguió por primera vez una mejora clara sobre climatología.

En el periodo 2018–2025:

```text
3M Persistence
RMSE ≈ 49.31
R²   ≈ 0.255

S2 Linear
RMSE ≈ 52.25
R²   ≈ 0.164

Climatología
RMSE ≈ 64.91
R²   ≈ -0.291
```

La persistencia se convirtió en un baseline especialmente fuerte.

---

# 12. Benchmark con PyCaret

Se utilizó PyCaret para comparar múltiples modelos de regresión.

Posteriormente, los candidatos fueron reevaluados manualmente con validación temporal para evitar depender del ranking automático.

Los mejores candidatos fueron principalmente modelos lineales o regularizados:

```text
Kernel Ridge
Bayesian Ridge
Theil-Sen
ARD
Ridge
Linear Regression
Lasso
ElasticNet
```

Kernel Ridge obtuvo el menor RMSE medio entre los modelos ML en el benchmark académico.

Pipeline final académico:

```text
SimpleImputer(strategy="mean")
        ↓
StandardScaler()
        ↓
KernelRidge(
    alpha=1,
    kernel="linear"
)
```

El modelo utiliza 16 features S2.

Resultados aproximados:

```text
CV pre-2018
Kernel Ridge RMSE ≈ 49.60
R² ≈ 0.349

Holdout 2018–2025
Kernel Ridge RMSE ≈ 51.55
R² ≈ 0.186
```

---

# 13. Baseline de persistencia

Se utilizó un baseline simple:

\[
\hat{y}_t =
\frac{
A_t + A_{t-1} + A_{t-2}
}{3}
\]

Interpretación:

> los próximos tres meses se parecerán a los últimos tres meses.

Este baseline no es un modelo entrenado.

Sin embargo, resultó extremadamente competitivo y en varios periodos superó a los modelos ML.

Esto se mantiene explícitamente en los resultados y en la aplicación Streamlit.

---

# 14. Arquitectura del proyecto

El proyecto fue reorganizado para separar experimentación, lógica reutilizable, entrenamiento y aplicación.

```text
enso-ml/
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── notebooks/
│
├── scripts/
│   ├── download_era5.py
│   ├── update_enso_operational.py
│   ├── update_precipitation.py
│   ├── build_operational_dataset.py
│   ├── retrain_operational.py
│   ├── benchmark_operational_models.py
│   └── update_all.py
│
├── src/
│   └── enso_ml/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── features.py
│       ├── model.py
│       ├── pipeline.py
│       └── train.py
│
├── tests/
│   ├── test_features.py
│   └── test_model.py
│
├── .streamlit/
│   └── config.toml
│
├── .gitignore
├── pyproject.toml
├── uv.lock
├── requirements.txt
└── README.md
```

---

# 15. Paquete `src/enso_ml`

La lógica final fue movida fuera de los notebooks.

## `data.py`

Responsable de:

- cargar datos;
- validar fechas;
- verificar continuidad mensual;
- unir ENSO y precipitación.

## `features.py`

Implementa:

- climatología;
- anomalías;
- medias móviles;
- codificación estacional;
- target estacional;
- features de entrenamiento;
- features de inferencia.

## `model.py`

Reproduce el pipeline final sin depender de PyCaret:

```text
SimpleImputer
StandardScaler
Kernel Ridge
```

También incluye:

- entrenamiento;
- predicción;
- métricas;
- persistencia;
- guardado y carga con `joblib`.

## `pipeline.py`

Orquesta:

```text
datos
→ feature engineering
→ entrenamiento
→ guardado
→ inferencia
```

---

# 16. Tests

Se añadieron tests con `pytest`.

Se verifican, entre otros:

```text
número correcto de features
medias móviles
target t+1:t+3
fechas de forecast
codificación cíclica
entrenamiento
predicción
SimpleImputer
métricas
persistencia del modelo con joblib
```

Ejecutar:

```bash
PYTHONPATH=src uv run pytest -v
```

---

# 17. Streamlit

La aplicación permite presentar:

- outlook actual;
- comparación entre modelos;
- baseline de persistencia;
- métricas históricas;
- índices ENSO;
- precipitación observada;
- metodología;
- diferencia entre modelo académico y extensión operacional.

Ejecutar localmente:

```bash
PYTHONPATH=src uv run streamlit run app/streamlit_app.py
```

---

# 18. Extensión operacional 2026

Después de completar el modelo académico, se extendió el proyecto para trabajar con datos actualizados.

La precipitación ERA5-Land se actualizó hasta:

```text
2026-08
```

Los índices ENSO operacionales se reconstruyeron utilizando fuentes CPC consistentes.

La versión operacional utiliza:

```text
Niño 1+2
Niño 3
Niño 3.4
Niño 4
SOI
```

TNI fue eliminado porque su disponibilidad operacional es más retrasada.

El dataset operacional quedó aproximadamente en:

```text
1982-01 → 2026-08
531 muestras etiquetadas
14 features
```

---

# 19. ¿Por qué fue necesario reentrenar?

Actualizar simplemente el modelo académico con los nuevos valores no era correcto.

El cambio hacia fuentes operacionales implicó:

- una serie histórica diferente;
- menor longitud temporal;
- eliminación de TNI;
- nueva distribución de algunas variables;
- nueva definición del espacio de entrenamiento.

Por ello se volvió a realizar:

```text
feature engineering
validación temporal
benchmark de modelos
evaluación holdout
```

---

# 20. Benchmark operacional final

Se compararon:

```text
Linear Regression
Ridge
Bayesian Ridge
Kernel Ridge
ElasticNet
Huber
```

más:

```text
3M Persistence
Climatología
```

Todos los modelos ML utilizaron:

```text
SimpleImputer(mean)
        ↓
StandardScaler
        ↓
regresor
```

Regla de selección:

> menor RMSE medio en `TimeSeriesSplit(n_splits=5, gap=2)` utilizando únicamente datos pre-2018.

Resultados de CV:

```text
ElasticNet       RMSE ≈ 51.85
Bayesian Ridge   RMSE ≈ 52.02
Ridge            RMSE ≈ 52.94
Huber            RMSE ≈ 56.35
Linear           RMSE ≈ 56.37
Persistence      RMSE ≈ 58.50
Climatología     RMSE ≈ 59.72
Kernel Ridge     RMSE ≈ 60.81
```

Por esta regla, **ElasticNet fue seleccionado como modelo operacional destacado**.

---

# 21. Evaluación 2018–2025 de la versión operacional

```text
3M Persistence
RMSE ≈ 51.07
R²   ≈ 0.176

Huber
RMSE ≈ 52.45
R²   ≈ 0.130

Kernel Ridge
RMSE ≈ 56.23
R²   ≈ 0.000

ElasticNet
RMSE ≈ 60.71
R²   ≈ -0.165

Climatología
RMSE ≈ 63.97
R²   ≈ -0.293
```

Aunque ElasticNet fue seleccionado por CV pre-2018, la persistencia resultó más robusta en el periodo reciente.

Por ello el sistema operacional se presenta como un **outlook experimental**, no como una predicción determinista.

---

# 22. Outlook septiembre–noviembre de 2026

Último mes observado:

```text
Agosto 2026
```

Ventana objetivo:

```text
Septiembre 2026
Octubre 2026
Noviembre 2026
```

Pronósticos del benchmark:

```text
Linear Regression     +86.38 mm/mes
Ridge                 +83.45 mm/mes
Kernel Ridge          +76.59 mm/mes
Bayesian Ridge        +72.89 mm/mes
ElasticNet            +64.63 mm/mes
Huber                 +50.01 mm/mes
3M Persistence         +1.41 mm/mes
```

Los seis modelos ML coinciden en una anomalía positiva, aunque con diferencias importantes en magnitud.

El resultado destacado de ElasticNet es:

\[
\boxed{
+64.63\text{ mm/mes}
}
\]

Interpretación:

> el modelo estima que, en promedio, la precipitación de septiembre–noviembre de 2026 podría situarse alrededor de **64.6 mm por mes por encima de la climatología mensual de referencia**.

Este valor corresponde aproximadamente al percentil histórico 89 de la distribución del target.

Debe interpretarse con cautela debido a:

- fuerte divergencia con persistencia;
- variabilidad temporal de la relación ENSO–precipitación;
- valores ENSO recientes extremos;
- menor número de observaciones en la versión operacional.

---

# 23. Actualización automática

Una vez descargada la precipitación ERA5 reciente, puede ejecutarse:

```bash
uv run python scripts/update_all.py \
    --through 2026-08 \
    --skip-precip
```

Para actualizar todo:

```bash
uv run python scripts/update_all.py --through 2026-08
```

El flujo operacional realiza:

```text
NOAA CPC ENSO
        ↓
ERA5-Land
        ↓
dataset operacional
        ↓
feature engineering
        ↓
validación / reentrenamiento
        ↓
outlook actualizado
```

---

# 24. Instalación

El proyecto utiliza `uv`.

Clonar:

```bash
git clone https://github.com/TU_USUARIO/enso-ml.git
cd enso-ml
```

Crear / sincronizar entorno:

```bash
uv sync
```

Si se utilizan scripts CDS / ERA5:

```bash
uv add cdsapi xarray netcdf4
```

Para Streamlit:

```bash
uv add streamlit plotly
```

Para tests:

```bash
uv add --dev pytest
```

---

# 25. Entrenar el modelo académico

Desde la raíz:

```bash
PYTHONPATH=src uv run python -m enso_ml.train
```

Se generan artefactos en:

```text
models/
```

---

# 26. Ejecutar tests

```bash
PYTHONPATH=src uv run pytest -v
```

---

# 27. Ejecutar la aplicación

```bash
PYTHONPATH=src uv run streamlit run app/streamlit_app.py
```

---

# 28. Datos y credenciales

Los datos crudos y credenciales no se versionan.

Ejemplo de `.gitignore`:

```gitignore
.venv/
__pycache__/
.ipynb_checkpoints/
.vscode/

.cdsapirc
.env

data/raw/*
models/*
```

Para Streamlit Cloud se versionan únicamente los CSV procesados pequeños requeridos por la aplicación.

Las credenciales de CDS deben mantenerse fuera de GitHub.

---

# 29. Principales conclusiones

El proyecto produjo varias conclusiones importantes:

1. **Predecir un mes exacto a +3 meses fue demasiado ruidoso.**
2. **Cambiar el target a una media estacional t+1:t+3 mejoró claramente el problema.**
3. **Feature engineering temporal fue más útil que simplemente añadir variables.**
4. **Modelos complejos no necesariamente superaron a regresores lineales regularizados.**
5. **Persistencia fue un baseline extremadamente competitivo.**
6. **La relación ENSO–precipitación no parece completamente estable entre periodos históricos.**
7. **Actualizar un modelo operacional requiere volver a validar, no solo añadir observaciones nuevas.**
8. **Los modelos ML operacionales coinciden actualmente en una señal húmeda para Sep–Nov 2026, pero con alta incertidumbre.**

---

# 30. Trabajo futuro

Posibles extensiones:

```text
comparar ventanas t+1:t+3, t+4:t+6 y t+7:t+9
incorporar predicciones probabilísticas
cuantificar incertidumbre
evaluar otros dominios espaciales de Ecuador
incorporar índices atmosféricos adicionales
automatizar completamente actualizaciones periódicas
integrar nuevas versiones de ERA5 / CPC
evaluar modelos por eventos El Niño / La Niña separados
```

Una extensión particularmente interesante sería estudiar cómo disminuye la habilidad predictiva con el horizonte:

\[
t+1:t+3
\]

\[
t+4:t+6
\]

\[
t+7:t+9
\]

---

# 31. Tecnologías

```text
Python 3.12
uv
pandas
NumPy
Xarray
scikit-learn
PyCaret
ERA5-Land
NOAA CPC / PSL
cdsapi
joblib
pytest
Streamlit
Plotly
Git / GitHub
```

---

# Autor

**Gustavo Paredes**

Proyecto desarrollado como parte del curso de Fundamentos de Inteligencia Artificial.

---

## Nota

Este proyecto tiene fines **académicos y experimentales**.

Los pronósticos mostrados por la aplicación no deben interpretarse como productos meteorológicos oficiales ni utilizarse para decisiones de gestión de riesgos, agricultura, infraestructura o emergencias sin contrastarlos con fuentes meteorológicas oficiales.
