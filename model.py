"""
ITAD Enterprise Asset Depreciation & Security Rule Engine
"""
from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

RANDOM_STATE = 42

CATEGORY_LIFE = {"Laptop": 5, "Desktop": 6, "Server": 7, "Network": 6}
CATEGORY_MARKET = {"Laptop": 0.92, "Desktop": 0.82, "Server": 1.08, "Network": 0.88}
CATEGORY_COST = {
    "Laptop": (700, 2600),
    "Desktop": (600, 2200),
    "Server": (2500, 12000),
    "Network": (800, 7000),
}

def create_demo_dataset(rows: int = 1200, introduce_nulls: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    categories = rng.choice(list(CATEGORY_LIFE), size=rows, p=[0.42, 0.25, 0.20, 0.13])
    purchase_cost = np.array([rng.uniform(*CATEGORY_COST[cat]) for cat in categories]).round(2)
    useful_life = np.array([CATEGORY_LIFE[cat] for cat in categories])
    age_years = np.array([rng.uniform(0.1, CATEGORY_LIFE[cat] * 1.6) for cat in categories]).round(1)
    condition = rng.choice(["Good", "Fair", "Poor"], size=rows, p=[0.48, 0.35, 0.17])
    
    condition_factor = pd.Series(condition).map({"Good": 1.0, "Fair": 0.78, "Poor": 0.52}).to_numpy()
    repair_cost = (
        purchase_cost * rng.beta(1.4, 8.0, size=rows) * np.where(condition == "Poor", 1.8, 1.0)
    ).round(2)
    
    age_fraction = np.clip(age_years / useful_life, 0, 1.5)
    depreciation_fraction = np.maximum(0, 1 - age_fraction) ** 1.65
    category_factor = np.array([CATEGORY_MARKET[item] for item in categories])
    
    remaining_value = (
        purchase_cost * depreciation_fraction * condition_factor * category_factor
        - repair_cost * 0.55
        + rng.normal(0, 0.045, size=rows) * purchase_cost
    )
    estimated_remaining_value = np.clip(remaining_value, 0, purchase_cost).round(2)

    df = pd.DataFrame(
        {
            "asset_id": [f"IT-{number:04d}" for number in range(1, rows + 1)],
            "asset_type": categories,
            "age_years": age_years,
            "useful_life_years": useful_life,
            "purchase_cost": purchase_cost,
            "condition": condition,
            "repair_cost_last_year": repair_cost,
            "storage_type": rng.choice(["HDD", "SSD", "NVMe", "No internal storage"], size=rows, p=[0.30, 0.46, 0.20, 0.04]),
            "data_classification": rng.choice(["Public", "Internal", "Confidential", "Restricted"], size=rows, p=[0.10, 0.43, 0.35, 0.12]),
            "estimated_remaining_value": estimated_remaining_value,
        }
    )

    if introduce_nulls:
        mask_repair = rng.random(rows) < 0.03
        mask_storage = rng.random(rows) < 0.02
        df.loc[mask_repair, "repair_cost_last_year"] = np.nan
        df.loc[mask_storage, "storage_type"] = np.nan

    return df


class ITADModelManager:
    def __init__(self, model_dir: Path | str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.model_dir / "remaining_value_pipeline.joblib"
        self.pipeline: Pipeline | None = None
        self.metrics: dict = {}
        self.feature_names: list[str] = []

    def build_pipeline(self) -> Pipeline:
        numeric_features = ["age_years", "useful_life_years", "purchase_cost", "repair_cost_last_year"]
        categorical_features = ["asset_type", "condition", "storage_type", "data_classification"]

        num_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        preprocessing = ColumnTransformer(
            transformers=[
                ("numeric", num_pipeline, numeric_features),
                ("categorical", cat_pipeline, categorical_features),
            ]
        )

        return Pipeline([
            ("preprocessing", preprocessing),
            ("regressor", RandomForestRegressor(n_estimators=180, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1))
        ])

    def train_and_save(self) -> dict:
        df = create_demo_dataset()
        features = df.drop(columns=["asset_id", "estimated_remaining_value"])
        target = df["estimated_remaining_value"]

        x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.20, random_state=RANDOM_STATE)

        self.pipeline = self.build_pipeline()
        self.pipeline.fit(x_train, y_train)

        predictions = self.pipeline.predict(x_test)
        
        ohe_cat_cols = self.pipeline.named_steps["preprocessing"].named_transformers_["categorical"].named_steps["encoder"].get_feature_names_out()
        num_cols = ["age_years", "useful_life_years", "purchase_cost", "repair_cost_last_year"]
        self.feature_names = list(num_cols) + list(ohe_cat_cols)

        importances = self.pipeline.named_steps["regressor"].feature_importances_
        feature_importance_dict = dict(zip(self.feature_names, [float(x) for x in importances]))

        self.metrics = {
            "r2_score": float(r2_score(y_test, predictions)),
            "mae": float(mean_absolute_error(y_test, predictions)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
            "feature_importance": dict(sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True)[:6])
        }

        joblib.dump(self.pipeline, self.model_path)
        return self.metrics

    def load_or_train(self) -> None:
        if self.model_path.exists():
            self.pipeline = joblib.load(self.model_path)
        else:
            self.train_and_save()

    def evaluate_governance_rules(self, row: pd.Series | dict, predicted_value: float) -> dict[str, str]:
        purchase_cost = float(row.get("purchase_cost", 0.0))
        age = float(row.get("age_years", 0.0))
        useful_life = float(row.get("useful_life_years", 1.0))
        repair_cost = float(row.get("repair_cost_last_year", 0.0) or 0.0)
        data_class = str(row.get("data_classification", "Public"))
        storage = str(row.get("storage_type", "HDD"))

        low_value_threshold = max(100.0, 0.10 * purchase_cost)
        due_for_retirement = (
            age >= useful_life
            or predicted_value <= low_value_threshold
            or repair_cost >= 0.25 * purchase_cost
        )

        if storage == "No internal storage":
            sanitization = "Clear: Basic Factory Reset"
        elif data_class in ["Confidential", "Restricted"]:
            sanitization = "NIST SP 800-88 Purge: Cryptographic Erasure / Destruction"
        else:
            sanitization = "NIST SP 800-88 Clear: Logical Overwrite Pass"

        return {
            "disposition_status": "Review for Retirement" if due_for_retirement else "Retain / Reassess",
            "sanitization_protocol": sanitization
        }

model_manager = ITADModelManager() 