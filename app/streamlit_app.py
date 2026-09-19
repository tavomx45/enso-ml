from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

PROCESSED = ROOT / "data" / "processed"

OPERATIONAL_DATA = PROCESSED / "enso_precip_operational.csv"
CV_FILE = PROCESSED / "operational_model_benchmark_cv.csv"
HOLDOUT_FILE = PROCESSED / "operational_model_benchmark_holdout.csv"
FORECAST_FILE = PROCESSED / "operational_model_benchmark_latest_forecasts.csv"


# ---------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="ENSO Rainfall Outlook · Ecuador",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 3rem;
            max-width: 1450px;
        }

        .hero {
            padding: 2.1rem 2.3rem;
            border-radius: 26px;
            margin-bottom: 1.2rem;
            background:
                radial-gradient(circle at 85% 15%, rgba(56,189,248,0.28), transparent 35%),
                linear-gradient(120deg, rgba(15,118,110,0.98), rgba(30,64,175,0.95));
            color: white;
            box-shadow: 0 18px 50px rgba(15,23,42,0.22);
        }

        .hero .eyebrow {
            font-size: 0.78rem;
            letter-spacing: 0.15em;
            font-weight: 800;
            opacity: 0.85;
            margin-bottom: 0.55rem;
        }

        .hero h1 {
            margin: 0;
            font-size: clamp(2rem, 4.2vw, 3.6rem);
            line-height: 1.02;
        }

        .hero p {
            margin: 0.9rem 0 0 0;
            max-width: 950px;
            font-size: 1.06rem;
            opacity: 0.94;
        }

        .info-card {
            border: 1px solid rgba(148,163,184,0.22);
            border-radius: 18px;
            padding: 1rem 1.2rem;
            background: rgba(255,255,255,0.025);
            margin-bottom: 0.8rem;
        }

        .wet-card {
            border-left: 5px solid #38bdf8;
        }

        .warning-card {
            border-left: 5px solid #f59e0b;
        }

        .success-card {
            border-left: 5px solid #22c55e;
        }

        .muted {
            opacity: 0.78;
            font-size: 0.9rem;
        }

        div[data-testid="stMetric"] {
            border: 1px solid rgba(148,163,184,0.20);
            border-radius: 18px;
            padding: 0.8rem 1rem;
            background: rgba(255,255,255,0.02);
        }

        div[data-testid="stMetricValue"] {
            font-size: 1.65rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_operational_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


required_files = [
    OPERATIONAL_DATA,
    CV_FILE,
    HOLDOUT_FILE,
    FORECAST_FILE,
]

missing = [
    str(path.relative_to(ROOT))
    for path in required_files
    if not path.exists()
]

if missing:
    st.error(
        "Faltan archivos de la versión operacional necesarios para la app."
    )
    st.code(
        "\n".join(missing),
        language="text",
    )
    st.info(
        "Ejecuta primero el flujo operacional y el mini-benchmark, "
        "y luego agrega estos CSV específicos al repositorio."
    )
    st.stop()


operational = load_operational_data(
    str(OPERATIONAL_DATA)
)
cv = load_csv(str(CV_FILE))
holdout = load_csv(str(HOLDOUT_FILE))
forecasts = load_csv(str(FORECAST_FILE))


# ---------------------------------------------------------------------
# Derived presentation values
# ---------------------------------------------------------------------
selected_model = "ElasticNet"

elastic_row = forecasts.loc[
    forecasts["model"] == selected_model
]

if elastic_row.empty:
    st.error(
        f"No se encontró {selected_model} en el archivo de pronósticos."
    )
    st.stop()

elastic_row = elastic_row.iloc[0]

persistence_row = forecasts.loc[
    forecasts["model"] == "3M Persistence"
].iloc[0]

ml_forecasts = forecasts.loc[
    forecasts["model"] != "3M Persistence"
].copy()

selected_forecast = float(
    elastic_row["forecast_anomaly_mm_per_month"]
)
persistence_forecast = float(
    persistence_row["forecast_anomaly_mm_per_month"]
)
selected_percentile = float(
    elastic_row["historical_percentile"]
)

ml_min = float(
    ml_forecasts["forecast_anomaly_mm_per_month"].min()
)
ml_max = float(
    ml_forecasts["forecast_anomaly_mm_per_month"].max()
)

observation_month = pd.Timestamp(
    elastic_row["observation_month"]
)
target_start = pd.Timestamp(
    elastic_row["target_start"]
)
target_end = pd.Timestamp(
    elastic_row["target_end"]
)

latest = operational.sort_values("date").tail(1).iloc[0]

elastic_cv = cv.loc[
    cv["model"] == selected_model
].iloc[0]

elastic_holdout = holdout.loc[
    holdout["model"] == selected_model
].iloc[0]

persistence_holdout = holdout.loc[
    holdout["model"] == "3M Persistence"
].iloc[0]


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
with st.sidebar:
    st.title("🌊 ENSO · Ecuador")
    st.caption(
        "Outlook estacional de precipitación con datos actualizados "
        "hasta agosto de 2026."
    )

    st.divider()

    history_years = st.select_slider(
        "Historia visible",
        options=[3, 5, 10, 20, 30],
        value=10,
    )

    st.markdown("**Último mes observado**")
    st.write(observation_month.strftime("%B %Y"))

    st.markdown("**Ventana del outlook**")
    st.write(
        f"{target_start.strftime('%b %Y')} – "
        f"{target_end.strftime('%b %Y')}"
    )

    st.divider()

    st.markdown("**Modelo operacional seleccionado**")
    st.write("ElasticNet")

    st.caption(
        "Selección por menor RMSE medio en validación temporal pre-2018. "
        "La persistencia permanece como baseline obligatorio."
    )

    if st.button(
        "↻ Limpiar caché",
        use_container_width=True,
    ):
        st.cache_data.clear()
        st.rerun()


# ---------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero">
        <div class="eyebrow">OUTLOOK OPERACIONAL · ENSO · ERA5-LAND · ECUADOR</div>
        <h1>Precipitación estacional Sep–Nov 2026</h1>
        <p>
            El sistema fue actualizado con información climática disponible
            hasta agosto de 2026. Tras reconstruir una versión operacional
            consistente y repetir la validación temporal, ElasticNet obtuvo
            el menor RMSE medio entre los candidatos ML en validación cruzada.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Main metrics
# ---------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "ElasticNet · Outlook ML",
        f"{selected_forecast:+.1f} mm/mes",
        help=(
            "Anomalía media mensual estimada para Sep–Nov 2026. "
            "Valor positivo = más húmedo que la climatología de referencia."
        ),
    )

with c2:
    st.metric(
        "Persistencia 3M",
        f"{persistence_forecast:+.1f} mm/mes",
        help=(
            "Baseline simple basado en la anomalía media de los 3 meses recientes."
        ),
    )

with c3:
    st.metric(
        "Rango modelos ML",
        f"{ml_min:+.0f} a {ml_max:+.0f}",
        help=(
            "Rango de pronósticos de los seis modelos ML del mini-benchmark."
        ),
    )

with c4:
    st.metric(
        "Percentil histórico",
        f"{selected_percentile:.1f}%",
        help=(
            "Posición del pronóstico ElasticNet dentro de la distribución "
            "histórica del target."
        ),
    )


st.markdown(
    f"""
    <div class="info-card wet-card">
        <strong>Lectura del outlook</strong><br>
        ElasticNet estima una anomalía media de
        <strong>{selected_forecast:+.1f} mm/mes</strong> durante
        septiembre–noviembre de 2026. Los seis modelos ML coinciden en una
        señal positiva, con valores entre <strong>{ml_min:+.0f}</strong> y
        <strong>{ml_max:+.0f} mm/mes</strong>.
        <div class="muted">
            Esto no significa que cada mes tendrá exactamente esa anomalía.
            El target es el promedio de las anomalías de los tres meses futuros.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="info-card warning-card">
        <strong>Incertidumbre importante</strong><br>
        La persistencia de 3 meses proyecta solo
        <strong>{persistence_forecast:+.1f} mm/mes</strong> y fue más robusta
        que los modelos ML en el holdout 2018–2025. Por ello, el valor de
        ElasticNet se presenta como <strong>outlook ML experimental</strong>,
        no como una predicción determinista.
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------
tab_outlook, tab_validation, tab_context, tab_academic, tab_method = st.tabs(
    [
        "🌦 Outlook 2026",
        "📊 Validación operacional",
        "📈 Contexto climático",
        "🧠 Modelo académico",
        "🧪 Metodología",
    ]
)


# ---------------------------------------------------------------------
# Tab 1: Outlook
# ---------------------------------------------------------------------
with tab_outlook:
    st.subheader(
        f"Ventana objetivo: "
        f"{target_start.strftime('%B %Y')} – "
        f"{target_end.strftime('%B %Y')}"
    )

    left, right = st.columns(
        [1.15, 1],
        gap="large",
    )

    with left:
        chart_df = forecasts.copy()
        chart_df["Tipo"] = chart_df["model"].apply(
            lambda value: (
                "Baseline"
                if value == "3M Persistence"
                else "Machine Learning"
            )
        )

        fig = px.bar(
            chart_df,
            x="model",
            y="forecast_anomaly_mm_per_month",
            color="Tipo",
            text_auto=".1f",
            title="Pronóstico por modelo",
            labels={
                "model": "Modelo",
                "forecast_anomaly_mm_per_month": "Anomalía (mm/mes)",
            },
        )
        fig.add_hline(
            y=0,
            line_dash="dash",
        )
        fig.update_layout(
            xaxis_tickangle=-30,
            legend_title_text="",
            margin=dict(
                l=10,
                r=10,
                t=55,
                b=80,
            ),
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with right:
        st.markdown("#### ¿Por qué destacamos ElasticNet?")

        st.write(
            "Al actualizar las fuentes climáticas para llegar hasta agosto "
            "de 2026, fue necesario reconstruir un dataset operacional "
            "consistente y volver a comparar los modelos."
        )

        st.metric(
            "ElasticNet · CV RMSE",
            f"{float(elastic_cv['RMSE']):.2f}",
            delta=(
                f"{float(elastic_cv['RMSE']) - float(cv.iloc[0]['RMSE']):+.2f}"
                if cv.iloc[0]["model"] != selected_model
                else "mejor ML por CV"
            ),
            delta_color="off",
        )

        st.write(
            "ElasticNet obtuvo el menor RMSE medio entre los candidatos ML "
            "durante la validación temporal pre-2018. Sin embargo, su "
            "desempeño en 2018–2025 fue más débil que la persistencia."
        )

        st.info(
            "La selección del modelo se hizo por CV temporal pre-2018. "
            "El holdout se conserva como evaluación posterior y no se usa "
            "para escoger retroactivamente otro modelo."
        )

    st.markdown("#### Consenso y dispersión de los modelos")

    st.dataframe(
        forecasts[
            [
                "model",
                "forecast_anomaly_mm_per_month",
                "historical_percentile",
            ]
        ].rename(
            columns={
                "model": "Modelo",
                "forecast_anomaly_mm_per_month": "Outlook (mm/mes)",
                "historical_percentile": "Percentil histórico",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )


# ---------------------------------------------------------------------
# Tab 2: Validation
# ---------------------------------------------------------------------
with tab_validation:
    st.subheader("Validación temporal de la versión operacional")

    st.markdown(
        """
        Al cambiar a fuentes operacionales consistentes, el dataset ENSO
        comienza en 1982 y se redujo el número de muestras. Para evitar
        sobreestimar el desempeño, se repitió la comparación con
        `TimeSeriesSplit(n_splits=5, gap=2)`.
        """
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "ElasticNet · CV RMSE",
            f"{float(elastic_cv['RMSE']):.2f}",
        )

    with c2:
        st.metric(
            "ElasticNet · Holdout RMSE",
            f"{float(elastic_holdout['RMSE']):.2f}",
        )

    with c3:
        st.metric(
            "Persistencia · Holdout RMSE",
            f"{float(persistence_holdout['RMSE']):.2f}",
        )

    st.markdown("#### CV temporal pre-2018")

    cv_display = cv.copy()
    st.dataframe(
        cv_display,
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("#### Holdout 2018–2025")

    st.dataframe(
        holdout,
        hide_index=True,
        use_container_width=True,
    )

    fig_holdout = px.bar(
        holdout.sort_values("RMSE"),
        x="model",
        y="RMSE",
        text_auto=".1f",
        title="RMSE en holdout 2018–2025",
        labels={
            "model": "Modelo",
            "RMSE": "RMSE",
        },
    )
    fig_holdout.update_layout(
        xaxis_tickangle=-30,
        margin=dict(
            l=10,
            r=10,
            t=55,
            b=80,
        ),
    )
    st.plotly_chart(
        fig_holdout,
        use_container_width=True,
    )

    st.markdown(
        """
        **Conclusión:** ElasticNet fue el mejor candidato ML según la regla
        de selección por CV, pero la persistencia fue más robusta durante
        2018–2025. La versión operacional se presenta por tanto como una
        extensión experimental del proyecto, no como un sistema determinista.
        """
    )


# ---------------------------------------------------------------------
# Tab 3: Context
# ---------------------------------------------------------------------
with tab_context:
    st.subheader("Contexto climático hasta agosto de 2026")

    cutoff = (
        operational["date"].max()
        - pd.DateOffset(
            years=history_years
        )
    )

    recent = operational.loc[
        operational["date"] >= cutoff
    ].copy()

    latest_metrics = st.columns(5)

    labels = [
        ("Niño 1+2", "nino12"),
        ("Niño 3", "nino3"),
        ("Niño 3.4", "nino34"),
        ("Niño 4", "nino4"),
        ("SOI", "soi"),
    ]

    for column, (label, key) in zip(
        latest_metrics,
        labels,
    ):
        with column:
            st.metric(
                label,
                f"{float(latest[key]):+.2f}",
            )

    c1, c2 = st.columns(2)

    with c1:
        fig_nino = px.line(
            recent,
            x="date",
            y=[
                "nino3",
                "nino34",
                "nino4",
            ],
            title="Índices Niño recientes",
            labels={
                "date": "Fecha",
                "value": "Anomalía",
                "variable": "Índice",
            },
        )
        fig_nino.add_hline(
            y=0,
            line_dash="dash",
        )
        st.plotly_chart(
            fig_nino,
            use_container_width=True,
        )

    with c2:
        fig_soi = px.line(
            recent,
            x="date",
            y="soi",
            title="Southern Oscillation Index (SOI)",
            labels={
                "date": "Fecha",
                "soi": "SOI",
            },
        )
        fig_soi.add_hline(
            y=0,
            line_dash="dash",
        )
        st.plotly_chart(
            fig_soi,
            use_container_width=True,
        )

    # Rebuild precipitation anomalies for visualization using 1950-2010 reference
    # is not possible from this shortened operational file alone. We therefore
    # show observed precipitation directly here.
    fig_precip = px.line(
        recent,
        x="date",
        y="precipitation_mm",
        title="Precipitación mensual observada · región costera seleccionada",
        labels={
            "date": "Fecha",
            "precipitation_mm": "Precipitación (mm)",
        },
    )
    st.plotly_chart(
        fig_precip,
        use_container_width=True,
    )

    st.caption(
        "La actualización operacional combina índices CPC consistentes con "
        "precipitación ERA5-Land agregada hasta agosto de 2026."
    )


# ---------------------------------------------------------------------
# Tab 4: Academic model
# ---------------------------------------------------------------------
with tab_academic:
    st.subheader("Modelo académico principal")

    st.markdown(
        """
        La versión académica original fue desarrollada con una serie histórica
        más larga y 16 variables S2, incluyendo TNI. En ese contexto,
        **Kernel Ridge** fue el mejor modelo ML por RMSE temporal.
        """
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Modelo",
            "Kernel Ridge",
        )

    with c2:
        st.metric(
            "Features",
            "16",
        )

    with c3:
        st.metric(
            "Target",
            "media t+1:t+3",
        )

    academic_results = pd.DataFrame(
        [
            {
                "Método": "3M Persistence",
                "RMSE 2018–2025": 49.31,
                "R² 2018–2025": 0.255,
            },
            {
                "Método": "Kernel Ridge",
                "RMSE 2018–2025": 51.55,
                "R² 2018–2025": 0.186,
            },
            {
                "Método": "Linear Regression",
                "RMSE 2018–2025": 52.25,
                "R² 2018–2025": 0.164,
            },
            {
                "Método": "Climatología",
                "RMSE 2018–2025": 64.91,
                "R² 2018–2025": -0.291,
            },
        ]
    )

    st.dataframe(
        academic_results,
        hide_index=True,
        use_container_width=True,
    )

    st.markdown(
        """
        El modelo académico y la versión operacional responden a dos objetivos
        relacionados pero distintos:

        - **Académico:** evaluar si ENSO contiene señal útil para anticipar
          precipitación estacional.
        - **Operacional:** actualizar el sistema con fuentes disponibles en
          tiempo casi real y producir un outlook actual.
        """
    )


# ---------------------------------------------------------------------
# Tab 5: Methodology
# ---------------------------------------------------------------------
with tab_method:
    st.subheader("Evolución completa del proyecto")

    st.markdown(
        """
        **1. Datos iniciales.** Se integraron índices ENSO históricos con
        precipitación ERA5-Land para una región costera seleccionada de Ecuador.

        **2. Anomalías.** La precipitación se expresó respecto de una
        climatología mensual de referencia 1950–2010 para separar el ciclo
        estacional normal de desviaciones potencialmente asociadas a ENSO.

        **3. Primera formulación.** Se intentó predecir la anomalía de un mes
        exacto a +3 meses. Los modelos no superaron de manera consistente a
        climatología.

        **4. Cambio del target.** Se reformuló el problema como la anomalía
        media de los próximos tres meses, una escala más coherente con la
        persistencia temporal de ENSO.

        **5. Features S2.** Se incorporaron estado actual de ENSO, medias
        móviles de 3 meses, estacionalidad cíclica y memoria reciente de
        precipitación.

        **6. Modelo académico.** Kernel Ridge fue el mejor modelo ML del
        benchmark histórico, aunque persistencia siguió siendo un baseline
        extremadamente competitivo.

        **7. Actualización 2026.** Para llegar hasta agosto de 2026 se
        actualizaron los índices ENSO con fuentes CPC consistentes y la
        precipitación mediante ERA5-Land.

        **8. Reentrenamiento operacional.** El cambio de fuente y la reducción
        del período histórico obligaron a repetir la validación. En el
        mini-benchmark final, ElasticNet obtuvo el menor RMSE medio en CV
        temporal pre-2018.

        **9. Outlook Sep–Nov 2026.** ElasticNet produce una señal húmeda de
        aproximadamente +64.6 mm/mes. Los demás modelos ML también proyectan
        anomalías positivas, aunque la persistencia permanece cerca de neutral
        y fue más robusta en el holdout reciente.
        """
    )

    st.markdown("#### Target estacional")
    st.latex(
        r"y_t = \frac{A_{t+1}+A_{t+2}+A_{t+3}}{3}"
    )

    st.markdown("#### Modelo operacional destacado")
    st.code(
        """SimpleImputer(strategy="mean")
    ↓
StandardScaler()
    ↓
ElasticNet(alpha=1.0, l1_ratio=0.5)""",
        language="text",
    )

    st.warning(
        "El outlook 2026 debe interpretarse como una señal experimental. "
        "La validación temporal muestra que la relación ENSO–precipitación "
        "no es completamente estable entre períodos."
    )


# ---------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------
st.divider()

st.caption(
    "Proyecto académico y extensión operacional · ENSO + ERA5-Land · Ecuador · "
    "Datos operacionales hasta agosto de 2026"
)
