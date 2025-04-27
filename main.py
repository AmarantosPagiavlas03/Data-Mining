# refactored_pipeline.py

import os
import time
from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
import polars as pl
import polars.selectors as cs
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import joblib

# --- Configuration ---
DEFAULT_DATA_FILEPATH = "training_set_VU_DM.csv"
DEFAULT_PLOT_DIR = "eda_plots"

# === Data Loading ===

class DataLoader:
    """Loads data into a Polars DataFrame."""

    def __init__(self, filepath: str = DEFAULT_DATA_FILEPATH):
        self.filepath = filepath
        self.df: Optional[pl.DataFrame] = None

    def load(self, **kwargs) -> Optional[pl.DataFrame]:
        print(f"Loading data from {self.filepath}")
        # map pandas-like args if needed
        if "nrows" in kwargs:
            kwargs["n_rows"] = kwargs.pop("nrows")
        if "usecols" in kwargs:
            kwargs["columns"] = kwargs.pop("usecols")
        kwargs.setdefault("infer_schema_length", 10000)

        try:
            self.df = pl.read_csv(self.filepath, **kwargs)
            print(f"Loaded: {self.df.shape}")
        except Exception as e:
            print(f"Error loading data: {e}")
            self.df = None
        return self.df

# === Plotting ===

class Plotter:
    """Handles all plotting and saving/display logic."""

    def __init__(self, save: bool = False, out_dir: str = DEFAULT_PLOT_DIR):
        self.save = save
        self.out_dir = out_dir

    def correlation_heatmap(self, corr_df, title: str = "Correlation Matrix"):
        plt.figure(figsize=(min(15, corr_df.shape[1]*0.6+2),
                            min(12, corr_df.shape[0]*0.6+1)))
        sns.heatmap(corr_df, cmap="coolwarm", linewidths=.2, annot=False)
        plt.title(title)
        self._finalize("correlation_matrix.png")

    def histogram_qq(self, arr: np.ndarray, name: str):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        sns.histplot(arr, kde=True, ax=ax1)
        ax1.set_title(f"Distribution of {name}")
        stats.probplot(arr, dist="norm", plot=ax2)
        ax2.set_title(f"Q–Q Plot of {name}")
        self._finalize(f"dist_{name}.png")

    def categorical_countplot(self, data_pd, column: str, order: List[str]):
        plt.figure(figsize=(min(12, len(order)*0.4),
                             max(5, min(8, len(order)*0.25 + 2))))
        if len(order) > 10:
            sns.countplot(y=data_pd[column], order=order)
            plt.xlabel("Count")
            plt.ylabel(column)
        else:
            sns.countplot(x=data_pd[column], order=order)
            plt.xlabel(column)
            plt.ylabel("Count")
            plt.xticks(rotation=30, ha="right")
        plt.title(f"Distribution of {column}")
        self._finalize(f"cat_{column}.png")

    def _finalize(self, filename: str):
        os.makedirs(self.out_dir, exist_ok=True)
        path = os.path.join(self.out_dir, filename)
        if self.save:
            plt.savefig(path, bbox_inches="tight", dpi=150)
            plt.close()
        else:
            plt.show()
            plt.close()


# === EDA Steps ===

class EdaStep(ABC):
    @abstractmethod
    def run(self, df: pl.DataFrame):
        """Execute this EDA step on the Polars DataFrame."""
        pass


class BasicInfoStep(EdaStep):
    def run(self, df: pl.DataFrame):
        print("--- Basic Information ---")
        print(f"Shape: {df.shape}")
        print("Columns:", df.columns)
        print("Schema:", df.schema)
        print("First 5 rows:")
        print(df.head())


class MissingValuesStep(EdaStep):
    def run(self, df: pl.DataFrame):
        print("\n--- Missing Values ---")
        mi = df.null_count().melt(variable_name="column", value_name="missing")
        mi = mi.filter(pl.col("missing") > 0)
        if mi.is_empty():
            print("No missing values.")
            return
        total = df.height
        mi = mi.with_columns((pl.col("missing") / total * 100).round(2).alias("pct"))
        mi = mi.sort("missing", descending=True)
        print(mi)


class NumericSummaryStep(EdaStep):
    def __init__(self, plotter: Plotter, plot: bool = True, corr: bool = True,
                 corr_sample_frac: float = 1.0, max_distributions: Optional[int] = 10):
        self.plotter = plotter
        self.plot = plot
        self.corr = corr
        self.corr_sample_frac = corr_sample_frac
        self.max_distributions = max_distributions

    def run(self, df: pl.DataFrame):
        num_cols = df.select(cs.numeric()).columns
        if not num_cols:
            print("\n--- No numerical features found ---")
            return

        print(f"\n--- Numerical Features ({len(num_cols)}) ---")
        print(df.select(num_cols).describe())

        if self.corr and len(num_cols) > 1:
            df_corr = df.select(num_cols)
            if 0 < self.corr_sample_frac < 1.0:
                n = max(2, int(df_corr.height * self.corr_sample_frac))
                df_corr = df_corr.sample(n=n, seed=42, shuffle=True)
            corr_pd = df_corr.corr().to_pandas()
            corr_pd.index = num_cols
            corr_pd.columns = num_cols
            self.plotter.correlation_heatmap(corr_pd)

        if self.plot:
            cols = num_cols[:self.max_distributions] if self.max_distributions else num_cols
            for col in cols:
                arr = df[col].drop_nulls().to_numpy()
                if arr.size > 0 and np.unique(arr).size > 1:
                    self.plotter.histogram_qq(arr, col)
                else:
                    print(f"Skipping distribution/Q-Q for '{col}' (constant or empty).")


class CategoricalAnalysisStep(EdaStep):
    def __init__(self, plotter: Plotter, plot: bool = True,
                 threshold: int = 50, max_plots: Optional[int] = 10):
        self.plotter = plotter
        self.plot = plot
        self.threshold = threshold
        self.max_plots = max_plots

    def run(self, df: pl.DataFrame):
        cat_cols = df.select(cs.string() | cs.categorical()).columns
        if not cat_cols:
            print("\n--- No categorical features found ---")
            return

        print(f"\n--- Categorical Features ({len(cat_cols)}) ---")
        cols = cat_cols[:self.max_plots] if self.max_plots else cat_cols
        for col in cols:
            series = df[col]
            n_unq = series.n_unique()
            print(f"\n'{col}' (unique: {n_unq})")

            # Compute value counts
            vc = series.value_counts()
            # Identify column names
            cols_vc = vc.columns
            value_col = cols_vc[0]
            count_col = cols_vc[1]

            # Sort descending / ascending
            sorted_desc = vc.sort(count_col, descending=True)
            sorted_asc  = vc.sort(count_col, descending=False)

            print("Top 5:")
            print(sorted_desc.head(5))
            if n_unq > 5:
                print("Bottom 5:")
                print(sorted_asc.head(5))

            # Plot if applicable 
            if self.plot and 1 < n_unq <= self.threshold:
                pd_df = df.select(col).to_pandas()
                top_n = sorted_desc.head(min(n_unq, 30))
                order = top_n[value_col].cast(pl.Utf8).to_list()
                self.plotter.categorical_countplot(pd_df, col, order)
            elif n_unq > self.threshold:
                print(f"Skipping plot for '{col}' ({n_unq} > {self.threshold}).")


class EdaPipeline:
    """Orchestrates a sequence of EDA steps."""

    def __init__(self, steps: List[EdaStep]):
        self.steps = steps

    def run(self, df: pl.DataFrame):
        print("\n=== Starting EDA Pipeline ===")
        for step in self.steps:
            step.run(df)
        print("=== EDA Pipeline Complete ===")

class FeatureEngineering:
    """Null‐handling with drop‐threshold and engineered features, logging all changes."""

    def __init__(self, cat_threshold: int = 50, null_pct_threshold: float = 70.0):
        """
        Args:
            cat_threshold: max unique values for mode‐imputation of categoricals.
            null_pct_threshold: if a column has > this % nulls, it's dropped instead of imputed.
        """
        self.cat_threshold = cat_threshold
        self.null_pct_threshold = null_pct_threshold
        self.impute_values: dict[str, float | str] = {}
        self.drop_cols: List[str] = []
        self.engineered_cols = [
            "price_per_person",
            "price_hist_ratio",
            "star_review_interaction",
            "search_weekday",
        ]

    def fit(self, df: pl.DataFrame) -> "FeatureEngineering":
        # Compute null percentages
        nulls = (
            df
            .null_count()
            .melt(variable_name="column", value_name="missing")
            .with_columns((pl.col("missing") / df.height * 100).round(2).alias("pct"))
        )
        # show the full dataframe
        nulls = nulls.with_columns(pl.col("column").cast(pl.Utf8))
        nulls = nulls.sort("missing", descending=True)
        
        for row in nulls.to_dicts():
            col, pct = row["column"], row["pct"]
            if pct > self.null_pct_threshold:
                self.drop_cols.append(col)
            else:
                # numeric median
                if df.schema[col] in (pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                                      pl.Float32, pl.Float64):
                    self.impute_values[col] = df[col].median()
                # small‐cardinality categorical → mode
                elif row := df[col].value_counts().to_dicts():
                    n_unq = df[col].n_unique()
                    if 1 < n_unq <= self.cat_threshold:
                        # find top value by counts
                        value_col, count_col = list(row[0].keys())
                        mode_val = max(row, key=lambda r: r[count_col])[value_col]
                        self.impute_values[col] = mode_val

        # Logging
        print("=== FeatureEngineering.fit ===")
        print(f"Dropping {len(self.drop_cols)} columns > {self.null_pct_threshold}% null:")
        for c in self.drop_cols:
            pct = next(r["pct"] for r in nulls.to_dicts() if r["column"] == c)
            print(f"  • {c!r}: {pct}% null")
        print(f"\nImputing {len(self.impute_values)} columns:")
        for c, v in self.impute_values.items():
            print(f"  • {c!r} → {v!r}")
        print("============================\n")
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        orig_cols = set(df.columns)

        # 1) Drop high‐null columns
        if self.drop_cols:
            df = df.drop(self.drop_cols)

        # 2) Add null‐flags for the rest
        null_flags = [pl.col(c).is_null().alias(f"{c}_is_null")
                      for c in self.impute_values]
        df = df.with_columns(null_flags)

        # 3) Impute
        df = df.fill_null(self.impute_values)

        # 4) Engineered features
        df = df.with_columns([
            # price per person
            (pl.col("price_usd") /
             (pl.col("srch_adults_count") + pl.col("srch_children_count"))
            ).alias("price_per_person"),

            # ratio to historical log‐price
            (pl.col("price_usd") /
             (pl.col("prop_log_historical_price") + 1)
            ).alias("price_hist_ratio"),

            # interaction: stars × review score
            (pl.col("prop_starrating") * pl.col("prop_review_score")
            ).alias("star_review_interaction"),

            # search weekday (0=Mon…6=Sun)
            pl.col("date_time")
              .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
              .dt.weekday()
              .alias("search_weekday"),
        ])

        # Logging
        new_cols = set(df.columns) - orig_cols
        print("=== FeatureEngineering.transform ===")
        if self.drop_cols:
            print(f"Dropped columns ({len(self.drop_cols)}):")
            for c in self.drop_cols:
                print(f"  • {c!r}")
        print(f"Added null‐flag columns ({len(self.impute_values)}):")
        for c in self.impute_values:
            print(f"  • {c + '_is_null'!r}")
        print(f"Added engineered features ({len(self.engineered_cols)}):")
        for c in self.engineered_cols:
            print(f"  • {c!r}")
        print("============================\n")

        return df




# === Modeling Pipeline ===

class BaseModelPipeline(ABC):
    """Defines the skeleton of a modeling workflow."""

    def __init__(self, model):
        self.model = model

    def execute(self, X: pl.DataFrame, y: pl.Series):
        X_train_pd, X_test_pd, y_train_pd, y_test_pd = self._split(X, y)
        self._train(X_train_pd, y_train_pd)
        preds = self._predict(X_test_pd)
        self._evaluate(y_test_pd, preds)

    def _split(self, X: pl.DataFrame, y: pl.Series):
        X_pd = X.to_pandas()
        y_pd = y.to_pandas()
        return train_test_split(X_pd, y_pd, test_size=0.2, random_state=42)

    @abstractmethod
    def _train(self, X_train, y_train):
        pass

    @abstractmethod
    def _predict(self, X_test):
        pass

    def _evaluate(self, y_true, y_pred):
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))


class LogisticRegressionPipeline(BaseModelPipeline):
    def __init__(self, **params):
        super().__init__(LogisticRegression(**params))

    def _train(self, X_train, y_train):
        start = time.time()
        self.model.fit(X_train, y_train)
        print(f"Trained LogisticRegression in {time.time() - start:.2f}s")

    def _predict(self, X_test):
        return self.model.predict(X_test)


class XGBoostPipeline(BaseModelPipeline):
    def __init__(self, **params):
        super().__init__(XGBClassifier(**params))

    def _train(self, X_train, y_train):
        start = time.time()
        self.model.fit(X_train, y_train)
        print(f"Trained XGBClassifier in {time.time() - start:.2f}s")

    def _predict(self, X_test):
        return self.model.predict(X_test)


# ... Similarly, you can add RandomForestPipeline, CatBoostPipeline, etc.


class AvailableModels:
    """Registry of available pipelines."""
    def __init__(self):
        self.registry = {
            "logistic": LogisticRegressionPipeline,
            "xgboost": XGBoostPipeline,
        }

    def get(self, name: str):
        return self.registry.get(name)
