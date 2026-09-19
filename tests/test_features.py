import numpy as np
import pandas as pd

from enso_ml.features import (
    ENSO_VARIABLES,
    MODEL_FEATURES,
    TARGET_COLUMN,
    add_rolling_features,
    add_seasonal_target,
    add_forecast_window,
)


def make_feature_dataframe():
    dates = pd.date_range(
        "2000-01-01",
        periods=8,
        freq="MS",
    )

    data = {
        "date": dates,
        "precip_anomaly": np.arange(
            8,
            dtype=float,
        ),
    }

    for variable in ENSO_VARIABLES:
        data[variable] = np.arange(
            8,
            dtype=float,
        )

    return pd.DataFrame(data)


def test_model_has_16_features():
    assert len(MODEL_FEATURES) == 16
    assert len(MODEL_FEATURES) == len(
        set(MODEL_FEATURES)
    )


def test_rolling_mean_uses_current_and_past_only():
    df = make_feature_dataframe()

    result = add_rolling_features(
        df,
        window=3,
    )

    # En la tercera observación:
    # (0 + 1 + 2) / 3 = 1
    assert result.loc[
        2,
        "precip_anomaly_ma3",
    ] == 1.0

    # ENSO debe funcionar igual.
    assert result.loc[
        2,
        "nino34_ma3",
    ] == 1.0


def test_seasonal_target_is_next_three_month_mean():
    df = make_feature_dataframe()

    result = add_seasonal_target(df)

    # Para t=0:
    # meses futuros = 1, 2, 3
    # promedio = 2
    assert result.loc[
        0,
        TARGET_COLUMN,
    ] == 2.0

    # Para t=1:
    # meses futuros = 2, 3, 4
    # promedio = 3
    assert result.loc[
        1,
        TARGET_COLUMN,
    ] == 3.0


def test_last_rows_have_no_target():
    df = make_feature_dataframe()

    result = add_seasonal_target(df)

    # Las últimas tres filas no tienen
    # tres meses futuros disponibles.
    assert result[
        TARGET_COLUMN
    ].tail(3).isna().all()


def test_forecast_window_dates_are_correct():
    df = make_feature_dataframe()

    result = add_forecast_window(df)

    first = result.iloc[0]

    assert first["target_start"] == pd.Timestamp(
        "2000-02-01"
    )

    assert first["target_center"] == pd.Timestamp(
        "2000-03-01"
    )

    assert first["target_end"] == pd.Timestamp(
        "2000-04-01"
    )


def test_cyclic_month_features_are_bounded():
    df = make_feature_dataframe()

    result = add_forecast_window(df)

    assert (
        result["target_month_sin"].abs()
        <= 1
    ).all()

    assert (
        result["target_month_cos"].abs()
        <= 1
    ).all()