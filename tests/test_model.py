import numpy as np
import pandas as pd

from enso_ml.features import (
    MODEL_FEATURES,
    TARGET_COLUMN,
)

from enso_ml.model import (
    build_model,
    load_model,
    predict,
    regression_metrics,
    save_model,
    train_model,
)


def make_training_dataframe(
    n_rows=30,
):
    rng = np.random.default_rng(42)

    data = {
        feature: rng.normal(
            size=n_rows
        )
        for feature in MODEL_FEATURES
    }

    data[TARGET_COLUMN] = (
        10
        + 2 * data["nino34"]
        - 3 * data["precip_anomaly"]
        + rng.normal(
            scale=0.1,
            size=n_rows,
        )
    )

    return pd.DataFrame(data)


def test_build_model_has_expected_steps():
    model = build_model()

    assert "preprocess" in model.named_steps
    assert "kr" in model.named_steps

    kr = model.named_steps["kr"]

    assert kr.kernel == "linear"
    assert kr.alpha == 1.0


def test_model_can_train_and_predict():
    df = make_training_dataframe()

    model = train_model(df)

    predictions = predict(
        model,
        df,
    )

    assert len(predictions) == len(df)

    assert np.isfinite(
        predictions
    ).all()


def test_model_handles_missing_feature_value():
    df = make_training_dataframe()

    df.loc[
        0,
        "nino34",
    ] = np.nan

    model = train_model(df)

    predictions = predict(
        model,
        df,
    )

    assert np.isfinite(
        predictions
    ).all()


def test_regression_metrics():
    y_true = np.array([
        1.0,
        2.0,
        3.0,
    ])

    y_pred = np.array([
        1.0,
        2.0,
        3.0,
    ])

    metrics = regression_metrics(
        y_true,
        y_pred,
    )

    assert metrics["MAE"] == 0
    assert metrics["RMSE"] == 0
    assert metrics["R2"] == 1


def test_model_save_and_load(
    tmp_path,
):
    df = make_training_dataframe()

    model = train_model(df)

    path = (
        tmp_path
        / "test_model.joblib"
    )

    save_model(
        model,
        path,
    )

    loaded = load_model(path)

    pred_original = predict(
        model,
        df,
    )

    pred_loaded = predict(
        loaded,
        df,
    )

    np.testing.assert_allclose(
        pred_original,
        pred_loaded,
    )