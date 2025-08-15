"""
training_pipeline.py (Lab-4 aligned)
------------------------------------
Automates EDA, preprocessing (numeric-only), feature selection (simple correlation filter),
hyperparameter tuning (RandomizedSearchCV), model training (LR, RF, XGB),
evaluation (ROC/Calibration/Brier/F1), SHAP analysis (best model), and PSI (train vs test).
"""

from __future__ import annotations
import re
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score, roc_curve, auc
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight


try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import shap
except ImportError:
    shap = None

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def safe_filename(s: str) -> str:
    """Sanitize string to be safe for filenames."""
    return re.sub(r"[^\w\d-]", "_", s)


def setup_logging(out_dir: Path) -> None:
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "logs" / "pipeline.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.info("Logging initialized: %s", log_file)


def eda_plots(df: pd.DataFrame, target: str, fig_dir: Path) -> List[str]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: List[str] = []

    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target]

    for col in num_cols:
        plt.figure()
        df[col].hist(bins=30)
        plt.title(f"Histogram: {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        p = fig_dir / f"hist_{safe_filename(col)}.png"
        plt.tight_layout()
        plt.savefig(p, dpi=150)
        plt.close()
        paths.append(str(p))

    uniq = set(df[target].dropna().unique())
    if uniq.issubset({0, 1}):
        for col in num_cols:
            plt.figure()
            data0 = df[df[target] == 0][col].dropna()
            data1 = df[df[target] == 1][col].dropna()
            plt.boxplot([data0, data1], tick_labels=["Class 0", "Class 1"])
            plt.title(f"Boxplot: {col} by {target}")
            plt.ylabel(col)
            p = fig_dir / f"box_{safe_filename(col)}.png"
            plt.tight_layout()
            plt.savefig(p, dpi=150)
            plt.close()
            paths.append(str(p))

    if len(num_cols) >= 2:
        corr = df[num_cols].corr(numeric_only=True)
        plt.figure(figsize=(8, 6))
        plt.imshow(corr, interpolation="nearest")
        plt.colorbar()
        plt.xticks(range(len(num_cols)), num_cols, rotation=90)
        plt.yticks(range(len(num_cols)), num_cols)
        plt.title("Correlation heatmap (numeric)")
        p = fig_dir / "corr_heatmap.png"
        plt.tight_layout()
        plt.savefig(p, dpi=150)
        plt.close()
        paths.append(str(p))

    return paths


def drop_high_corr(df: pd.DataFrame, numeric_cols: List[str], threshold: float = 0.95) -> List[str]:
    if not numeric_cols:
        return []
    corr = df[numeric_cols].corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > threshold)]
    kept = [c for c in numeric_cols if c not in to_drop]
    return kept


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(brier_score_loss(y_true, y_prob))


def calibration_and_roc(y_true_tr: np.ndarray, y_prob_tr: np.ndarray,
                        y_true_te: np.ndarray, y_prob_te: np.ndarray,
                        prefix: str, fig_dir: Path) -> Tuple[str, str]:
    fig_dir.mkdir(parents=True, exist_ok=True)

    def rel_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10):
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        ids = np.digitize(y_prob, bins) - 1
        px, fx = [], []
        for i in range(n_bins):
            m = ids == i
            if np.any(m):
                px.append(y_prob[m].mean())
                fx.append(y_true[m].mean())
            else:
                px.append(np.nan)
                fx.append(np.nan)
        return np.array(px), np.array(fx)

    pt, ft = rel_curve(y_true_tr, y_prob_tr)
    pv, fv = rel_curve(y_true_te, y_prob_te)

    plt.figure()
    plt.plot([0, 1], [0, 1], "--", label="Perfect")
    plt.plot(pt, ft, "o-", label="Train")
    plt.plot(pv, fv, "o-", label="Test")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration curve (Train vs Test)")
    plt.legend()
    cal_path = fig_dir / f"{prefix}_calibration.png"
    plt.tight_layout()
    plt.savefig(cal_path, dpi=150)
    plt.close()

    fpr_t, tpr_t, _ = roc_curve(y_true_tr, y_prob_tr)
    fpr_v, tpr_v, _ = roc_curve(y_true_te, y_prob_te)
    auc_t = auc(fpr_t, tpr_t)
    auc_v = auc(fpr_v, tpr_v)

    plt.figure()
    plt.plot([0, 1], [0, 1], "--")
    plt.plot(fpr_t, tpr_t, label=f"Train AUC={auc_t:.3f}")
    plt.plot(fpr_v, tpr_v, label=f"Test AUC={auc_v:.3f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC (Train vs Test)")
    plt.legend()
    roc_path = fig_dir / f"{prefix}_roc.png"
    plt.tight_layout()
    plt.savefig(roc_path, dpi=150)
    plt.close()

    return str(cal_path), str(roc_path)


def calculate_psi(train_vals: np.ndarray, test_vals: np.ndarray, n_bins: int = 10) -> float:
    edges = np.quantile(train_vals[~np.isnan(train_vals)], np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) <= 2:
        edges = np.linspace(np.nanmin(train_vals), np.nanmax(train_vals), n_bins + 1)

    def counts(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
        hist, _ = np.histogram(values[~np.isnan(values)], bins=bins)
        return hist.astype(float)

    tr, te = counts(train_vals, edges), counts(test_vals, edges)

    def to_share(arr: np.ndarray) -> np.ndarray:
        arr = arr + 1e-6
        return arr / arr.sum()

    ps, qs = to_share(tr), to_share(te)
    psi = np.sum((ps - qs) * np.log(ps / qs))
    return float(psi)


def psi_dataframe(X_train: pd.DataFrame, X_test: pd.DataFrame, fig_dir: Path, prefix: str = "psi"):
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    rows = []
    for col in num_cols:
        val = calculate_psi(X_train[col].values, X_test[col].values)
        rows.append({"feature": col, "psi": val})
    df = pd.DataFrame(rows).sort_values("psi", ascending=False)

    plt.figure(figsize=(8, max(4, 0.25 * len(df))))
    plt.barh(df["feature"], df["psi"])
    plt.xlabel("PSI")
    plt.title("Population Stability Index (Train vs Test)")
    p = fig_dir / f"{prefix}_bar.png"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(p, dpi=150)
    plt.close()
    return df, str(p)


def shap_summary(model: object, X_train_trans: np.ndarray, X_sample_trans: np.ndarray,
                 feature_names: List[str], fig_dir: Path, is_tree: bool):
    if shap is None:
        logging.warning("shap not installed; skipping.")
        return None

    try:
        if is_tree and hasattr(model, "predict_proba"):
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_sample_trans)
            if isinstance(shap_vals, list) and len(shap_vals) == 2:
                shap_vals = shap_vals[1]
        else:
            explainer = shap.Explainer(model, X_train_trans)
            exp = explainer(X_sample_trans)
            shap_vals = exp.values if hasattr(exp, "values") else exp
    except Exception as exc:
        logging.warning("SHAP failed: %s", exc)
        return None

    plt.figure(figsize=(8, 6))
    try:
        shap.summary_plot(shap_vals, X_sample_trans, feature_names=feature_names, show=False, max_display=20)
    except Exception:
        import numpy as np
        m = np.mean(np.abs(shap_vals), axis=0)
        idx = np.argsort(m)[::-1][:20]
        plt.barh([feature_names[i] for i in idx][::-1], m[idx][::-1])
        plt.xlabel("mean |SHAP|")
        plt.title("SHAP summary (bar)")

    p = fig_dir / "shap_summary.png"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(p, dpi=150)
    plt.close()
    return str(p)


class ModelSpec:
    def __init__(self, name: str, estimator: object, params: Dict[str, Iterable], is_tree: bool = False):
        self.name = name
        self.estimator = estimator
        self.params = params
        self.is_tree = is_tree


def registry(class_weight: Optional[Dict[int, float]], scale_pos_weight: Optional[float]):
    return [
        ModelSpec(
            "LogisticRegression",
            LogisticRegression(max_iter=5000, class_weight=class_weight, random_state=RANDOM_STATE),
            {
                "C": np.logspace(-3, 3, 5),
                "penalty": ["l2"],
                "solver": ["lbfgs"],
            },
            is_tree=False,
        ),
        ModelSpec(
            "RandomForestClassifier",
            RandomForestClassifier(random_state=RANDOM_STATE, class_weight=class_weight),
            {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 10, None],
                "min_samples_split": [2, 5, 10],
            },
            is_tree=True,
        ),
        ModelSpec(
            "XGBClassifier",
            xgb.XGBClassifier(
                random_state=RANDOM_STATE,
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
            ),
            {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.2],
            },
            is_tree=True,
        ) if xgb else None,
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--target_column", type=str, required=True)
    args = parser.parse_args()

    out_dir = Path("outputs")
    fig_dir = out_dir / "figures" / "eda"
    setup_logging(out_dir)

    df = pd.read_csv(args.input_csv)
    target = args.target_column

    eda_paths = eda_plots(df, target, fig_dir)
    logging.info("EDA plots generated: %d", len(eda_paths))

    X = df.drop(columns=[target])
    y = df[target]

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = drop_high_corr(X, numeric_cols, threshold=0.95)
    X = X[numeric_cols]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=RANDOM_STATE)

    class_weight_dict = dict(enumerate(compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)))
    scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

    pipeline = ColumnTransformer(
        [("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_cols)]
    )

    X_train_trans = pipeline.fit_transform(X_train)
    X_test_trans = pipeline.transform(X_test)

    models = [m for m in registry(class_weight_dict, scale_pos_weight) if m is not None]

    results = {}
    for mspec in models:
        logging.info("Training model: %s", mspec.name)
        search = RandomizedSearchCV(
            mspec.estimator,
            mspec.params,
            n_iter=5,
            cv=StratifiedKFold(n_splits=3),
            scoring="roc_auc",
            n_jobs=1,  # memory safe
            random_state=RANDOM_STATE,
            verbose=1,
        )
        search.fit(X_train_trans, y_train)
        best_model = search.best_estimator_

        y_train_prob = best_model.predict_proba(X_train_trans)[:, 1]
        y_test_prob = best_model.predict_proba(X_test_trans)[:, 1]

        cal_path, roc_path = calibration_and_roc(y_train, y_train_prob, y_test, y_test_prob, mspec.name, out_dir / "figures")

        results[mspec.name] = {
            "best_params": search.best_params_,
            "roc_train": roc_auc_score(y_train, y_train_prob),
            "roc_test": roc_auc_score(y_test, y_test_prob),
            "brier_train": brier_score(y_train, y_train_prob),
            "brier_test": brier_score(y_test, y_test_prob),
            "f1_train": f1_score(y_train, best_model.predict(X_train_trans)),
            "f1_test": f1_score(y_test, best_model.predict(X_test_trans)),
            "roc_plot": roc_path,
            "calibration_plot": cal_path,
        }

        if shap is not None:
            sample_idx = np.random.choice(len(X_train_trans), min(100, len(X_train_trans)), replace=False)
            shap_path = shap_summary(best_model, X_train_trans, X_train_trans[sample_idx], numeric_cols, out_dir / "figures", mspec.is_tree)
            results[mspec.name]["shap_plot"] = shap_path

    psi_df, psi_path = psi_dataframe(X_train, X_test, out_dir / "figures")
    results["psi"] = {"df": psi_df.to_dict(orient="records"), "plot": psi_path}

    with open(out_dir / "report.json", "w") as f:
        json.dump(results, f, indent=4)
    logging.info("Pipeline complete. Report saved to outputs/report.json")


if __name__ == "__main__":
    main()
