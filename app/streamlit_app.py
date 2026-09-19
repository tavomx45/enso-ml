from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from enso_ml.config import ProjectPaths
from enso_ml.data import load_enso_precipitation
from enso_ml.features import MODEL_FEATURES, build_inference_frame
from enso_ml.model import load_model, predict
from enso_ml.pipeline import load_climatology


# ---------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="ENSO Rainfall Forecast",
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
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            max-width: 1400px;
        }

        .hero {
            padding: 2rem 2.2rem;
            border-radius: 24px;
            margin-bottom: 1.2rem;
            background:
                linear-gradient(120deg, rgba(14, 116, 144, 0.96), rgba(30, 64, 175, 0.92));
            color: white;
            box-shadow: 0 18px 50px rgba(15, 23, 42, 0.18);
        }

        .hero .eyebrow {
            font-size: 0.78rem;
            letter-spacing: 0.15em;
            font-weight: 700;
            opacity: 0.85;
            margin-bottom: 0.6rem;
        }

        .hero h1 {
            margin: 0;
            font-size: clamp(2rem, 4vw, 3.4rem);
            line-height: 1.02;
        }

        .hero p {
            margin-top: 0.8rem;
            margin-bottom: 0;
            max-width: 850px;
            font-size: 1.05rem;
            opacity: 0.92;
        }

        .forecast-card {
            padding: 1.15rem 1.3rem;
            border-radius: 18px;
            border: 1px solid rgba(148, 163, 184, 0.25);
            background: rgba(255, 255, 255, 0.03);
            margin-bottom: 0.8rem;
        }

        .forecast-card strong {
            font-size: 1.05rem;
        }

        .positive {
            border-left: 5px solid #0284c7;
        }

        .negative {
            border-left: 5px solid #f59e0b;
        }

        .neutral {
            border-left: 5px solid #64748b;
        }

        .small-note {
            font-size: 0.88rem;
            opacity: 0.78;
        }

        div[data-testid="stMetric"] {
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 18px;
            padding: 0.75rem 1rem;
            background: rgba(255, 255, 255, 0.02);
        }

        div[data-testid="stMetricValue"] {
            font-size: 1.65rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_artifacts(
    model_path: str,
    model_mtime: float,
    climatology_path: str,
    climatology_mtime: float,
):
    # mtimes are intentionally part of the cache key.
    del model_mtime, climatology_mtime

    model = load_model(model_path)
    climatology = load_climatology(climatology_path)

    return model, climatology


@st.cache_data(show_spinner=False)
def load_project_data(
    processed_dir: str,
    climatology_items: tuple[tuple[int, float], ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    climatology = dict(climatology_items)

    raw = load_enso_precipitation(processed_dir)
    inference = build_inference_frame(
        raw,
        climatology=climatology,
    )

    return raw, inference


def read_metadata(path: Path) -> dict:
    if not path.exists():
        return {}

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def anomaly_message(value: float) -> tuple[str, str]:
    if value < 0:
        return (
            "negative",
            "El modelo proyecta una ventana más seca que la climatología de referencia.",
        )

    if value > 0:
        return (
            "positive",
            "El modelo proyecta una ventana más húmeda que la climatología de referencia.",
        )

    return (
        "neutral",
        "El modelo proyecta una ventana cercana a la climatología de referencia.",
    )


# ---------------------------------------------------------------------
# Load model and data
# ---------------------------------------------------------------------
paths = ProjectPaths.from_root(ROOT)

missing_artifacts = [
    path
    for path in [
        paths.model_file,
        paths.climatology_file,
    ]
    if not path.exists()
]

if missing_artifacts:
    st.error(
        "No encuentro los artefactos del modelo. "
        "Entrena primero el pipeline desde la raíz del proyecto."
    )
    st.code(
        "PYTHONPATH=src uv run python -m enso_ml.train",
        language="bash",
    )
    st.stop()

model, climatology = load_artifacts(
    str(paths.model_file),
    paths.model_file.stat().st_mtime,
    str(paths.climatology_file),
    paths.climatology_file.stat().st_mtime,
)

raw_df, inference_df = load_project_data(
    str(paths.data_processed),
    tuple(sorted(climatology.items())),
)

metadata = read_metadata(paths.metadata_file)

if inference_df.empty:
    st.error("No fue posible construir una fila válida de inferencia.")
    st.stop()

latest = inference_df.iloc[-1:]
latest_row = latest.iloc[0]

forecast = float(
    predict(
        model,
        latest,
    )[0]
)

persistence = float(
    latest_row["precip_anomaly_ma3"]
)

observation_date = pd.Timestamp(
    latest_row["date"]
)
target_start = pd.Timestamp(
    latest_row["target_start"]
)
target_center = pd.Timestamp(
    latest_row["target_center"]
)
target_end = pd.Timestamp(
    latest_row["target_end"]
)

status_class, interpretation = anomaly_message(
    forecast
)


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
with st.sidebar:
    st.title("🌊 ENSO Forecast")
    st.caption(
        "Predicción estacional de precipitación para la región costera seleccionada de Ecuador."
    )

    st.divider()

    history_years = st.select_slider(
        "Historia visible",
        options=[3, 5, 10, 20, 30],
        value=10,
        help="Número de años mostrados en los gráficos históricos.",
    )

    show_persistence = st.toggle(
        "Mostrar baseline de persistencia",
        value=True,
    )

    st.divider()

    st.markdown("**Última observación**")
    st.write(
        observation_date.strftime("%B %Y")
    )

    st.markdown("**Ventana pronosticada**")
    st.write(
        f"{target_start.strftime('%b %Y')} – "
        f"{target_end.strftime('%b %Y')}"
    )

    if st.button(
        "↻ Limpiar caché",
        use_container_width=True,
    ):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    st.divider()
    st.caption(
        "Modelo: Kernel Ridge lineal · 16 variables S2"
    )


# ---------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">MACHINE LEARNING · ENSO · ECUADOR</div>
        <h1>Pronóstico estacional de precipitación</h1>
        <p>
            Estimación de la anomalía media mensual de precipitación durante
            los próximos tres meses, combinando indicadores ENSO,
            estacionalidad y memoria reciente de precipitación.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Top metrics
# ---------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Kernel Ridge",
        f"{forecast:+.1f} mm/mes",
        help="Anomalía media mensual prevista para los próximos 3 meses.",
    )

with col2:
    st.metric(
        "Persistencia 3M",
        f"{persistence:+.1f} mm/mes",
        help="Baseline: supone que los próximos 3 meses se parecerán a los 3 meses recientes.",
    )

with col3:
    st.metric(
        "Diferencia modelo–baseline",
        f"{forecast - persistence:+.1f} mm/mes",
    )

with col4:
    st.metric(
        "Horizonte",
        "3 meses",
        help="Ventana objetivo: t+1, t+2 y t+3.",
    )

st.markdown(
    f"""
    <div class="forecast-card {status_class}">
        <strong>Lectura del pronóstico</strong><br>
        {interpretation}
        <div class="small-note">
            Un valor de {forecast:+.1f} mm/mes no significa lluvia negativa:
            expresa la diferencia respecto de la climatología mensual de referencia.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------
tab_forecast, tab_history, tab_model, tab_method = st.tabs(
    [
        "🔭 Pronóstico",
        "📈 Contexto histórico",
        "🧠 Modelo",
        "🧪 Metodología",
    ]
)


# ---------------------------------------------------------------------
# Forecast tab
# ---------------------------------------------------------------------
with tab_forecast:
    st.subheader(
        f"Ventana objetivo: "
        f"{target_start.strftime('%b %Y')} – "
        f"{target_end.strftime('%b %Y')}"
    )

    left, right = st.columns(
        [1, 1.45],
        gap="large",
    )

    with left:
        comparison_df = pd.DataFrame(
            {
                "Método": [
                    "Kernel Ridge",
                    "Persistencia 3M",
                ],
                "Anomalía (mm/mes)": [
                    forecast,
                    persistence,
                ],
            }
        )

        fig_bar = px.bar(
            comparison_df,
            x="Método",
            y="Anomalía (mm/mes)",
            text_auto=".1f",
            title="Modelo vs baseline",
        )
        fig_bar.add_hline(
            y=0,
            line_dash="dash",
        )
        fig_bar.update_layout(
            showlegend=False,
            margin=dict(
                l=10,
                r=10,
                t=55,
                b=10,
            ),
        )
        st.plotly_chart(
            fig_bar,
            use_container_width=True,
        )

    with right:
        cutoff = (
            observation_date
            - pd.DateOffset(
                years=history_years
            )
        )

        history = inference_df.loc[
            inference_df["date"] >= cutoff
        ].copy()

        fig_history = go.Figure()

        fig_history.add_trace(
            go.Scatter(
                x=history["date"],
                y=history["precip_anomaly"],
                mode="lines",
                name="Anomalía observada",
            )
        )

        fig_history.add_trace(
            go.Scatter(
                x=[target_center],
                y=[forecast],
                mode="markers",
                marker=dict(size=13),
                name="Kernel Ridge",
            )
        )

        if show_persistence:
            fig_history.add_trace(
                go.Scatter(
                    x=[target_center],
                    y=[persistence],
                    mode="markers",
                    marker=dict(size=12),
                    name="Persistencia 3M",
                )
            )

        fig_history.add_hline(
            y=0,
            line_dash="dash",
        )

        fig_history.update_layout(
            title=(
                "Anomalía histórica y pronóstico "
                "en el centro de la ventana futura"
            ),
            xaxis_title="Fecha",
            yaxis_title="Anomalía (mm/mes)",
            hovermode="x unified",
            margin=dict(
                l=10,
                r=10,
                t=55,
                b=10,
            ),
        )

        st.plotly_chart(
            fig_history,
            use_container_width=True,
        )

    st.info(
        "El pronóstico representa el promedio de las anomalías de "
        "los meses t+1, t+2 y t+3. No es una predicción de un único mes."
    )


# ---------------------------------------------------------------------
# Historical context tab
# ---------------------------------------------------------------------
with tab_history:
    st.subheader("Contexto climático reciente")

    cutoff = (
        observation_date
        - pd.DateOffset(
            years=history_years
        )
    )

    recent = raw_df.loc[
        raw_df["date"] >= cutoff
    ].copy()

    recent_inference = inference_df.loc[
        inference_df["date"] >= cutoff
    ].copy()

    fig_precip = px.line(
        recent_inference,
        x="date",
        y="precip_anomaly",
        title="Anomalía mensual de precipitación",
        labels={
            "date": "Fecha",
            "precip_anomaly": "Anomalía (mm)",
        },
    )
    fig_precip.add_hline(
        y=0,
        line_dash="dash",
    )
    fig_precip.update_layout(
        hovermode="x unified",
    )
    st.plotly_chart(
        fig_precip,
        use_container_width=True,
    )

    c1, c2 = st.columns(2)

    with c1:
        fig_nino = px.line(
            recent,
            x="date",
            y="nino34",
            title="Niño 3.4",
            labels={
                "date": "Fecha",
                "nino34": "Anomalía Niño 3.4",
            },
        )
        fig_nino.add_hline(
            y=0,
            line_dash="dash",
        )
        fig_nino.update_layout(
            hovermode="x unified",
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
        fig_soi.update_layout(
            hovermode="x unified",
        )
        st.plotly_chart(
            fig_soi,
            use_container_width=True,
        )


# ---------------------------------------------------------------------
# Model tab
# ---------------------------------------------------------------------
with tab_model:
    st.subheader("Modelo seleccionado")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Algoritmo",
            "Kernel Ridge",
        )

    with c2:
        st.metric(
            "Kernel",
            "Linear",
        )

    with c3:
        st.metric(
            "Features",
            str(len(MODEL_FEATURES)),
        )

    st.markdown(
        """
        El pipeline final reproduce el mejor candidato encontrado durante
        el benchmark temporal con PyCaret:
        """
    )

    st.code(
        """SimpleImputer(strategy="mean")
    ↓
StandardScaler()
    ↓
KernelRidge(alpha=1, kernel="linear")""",
        language="text",
    )

    st.markdown("#### Variables utilizadas")

    feature_groups = {
        "ENSO actual": [
            "nino12",
            "nino3",
            "nino34",
            "nino4",
            "soi",
            "tni",
        ],
        "ENSO media móvil 3M": [
            "nino12_ma3",
            "nino3_ma3",
            "nino34_ma3",
            "nino4_ma3",
            "soi_ma3",
            "tni_ma3",
        ],
        "Estacionalidad": [
            "target_month_sin",
            "target_month_cos",
        ],
        "Precipitación local": [
            "precip_anomaly",
            "precip_anomaly_ma3",
        ],
    }

    for group, values in feature_groups.items():
        with st.expander(
            f"{group} · {len(values)} variables"
        ):
            st.write(values)

    st.markdown("#### Evaluación histórica")

    evaluation_df = pd.DataFrame(
        [
            {
                "Método": "3M Persistence",
                "RMSE": 49.31,
                "R²": 0.255,
            },
            {
                "Método": "Kernel Ridge",
                "RMSE": 51.55,
                "R²": 0.186,
            },
            {
                "Método": "Linear Regression",
                "RMSE": 52.25,
                "R²": 0.164,
            },
            {
                "Método": "Climatología",
                "RMSE": 64.91,
                "R²": -0.291,
            },
        ]
    )

    st.dataframe(
        evaluation_df,
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "Resultados del benchmark post-hoc 2018–2025. "
        "La persistencia fue un baseline muy competitivo; Kernel Ridge "
        "fue el mejor modelo de Machine Learning."
    )

    if metadata:
        with st.expander("Metadata del modelo entrenado"):
            st.json(metadata)


# ---------------------------------------------------------------------
# Methodology tab
# ---------------------------------------------------------------------
with tab_method:
    st.subheader("Cómo llegamos al modelo final")

    st.markdown(
        """
        **1. Datos climáticos.** Se integraron índices ENSO mensuales con
        precipitación media espacial de ERA5-Land para la región costera
        seleccionada de Ecuador.

        **2. Anomalías.** La precipitación se expresó como diferencia respecto
        de una climatología mensual de referencia. Esto evita que el modelo
        se limite a aprender que ciertos meses son naturalmente más lluviosos.

        **3. Primera formulación.** Se intentó predecir la anomalía de un mes
        específico a +3 meses. ENSO, lags, precipitación reciente y variables
        locales no superaron de forma consistente a la climatología.

        **4. Cambio de target.** El problema se reformuló como la anomalía media
        de los próximos tres meses. Esta escala estacional está mejor alineada
        con la dinámica temporal de ENSO.

        **5. Features S2.** Se utilizaron valores ENSO actuales, medias móviles
        de tres meses, estacionalidad cíclica y memoria reciente de precipitación.

        **6. Validación temporal.** Los modelos se compararon respetando el
        orden cronológico mediante validación temporal y `TimeSeriesSplit`.

        **7. Modelo final.** Kernel Ridge lineal fue el mejor modelo ML del
        benchmark. El baseline de persistencia permanece visible porque fue
        especialmente competitivo en 2018–2025.
        """
    )

    st.markdown("#### Definición del target")
    st.latex(
        r"y_t = \frac{A_{t+1}+A_{t+2}+A_{t+3}}{3}"
    )

    st.caption(
        "A representa la anomalía mensual de precipitación respecto de la climatología."
    )


# ---------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------
st.divider()

st.caption(
    "Proyecto académico de predicción estacional · "
    "ENSO + ERA5-Land · Ecuador"
)
