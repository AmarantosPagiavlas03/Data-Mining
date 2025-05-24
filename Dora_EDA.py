# eda_analyzer.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import time
import os
from typing import Optional, List
import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMRanker
from sklearn.model_selection import GroupShuffleSplit 
import lightgbm as lgb, json, os
print(lgb.__version__, lgb.basic._LIB.LGBM_GetLastError())

# Standard progress bar
try:
    from tqdm import tqdm
except ImportError:
    print("Consider installing tqdm (`pip install tqdm`) for progress bars.")
    tqdm = lambda iterable, *args, **kwargs: iterable # Dummy fallback

# --- Configuration ---
DEFAULT_DATA_FILEPATH = "training_set_VU_DM.csv"
DEFAULT_PLOT_DIR = "eda_plots"

class MyModel: # Renamed class for clarity
    def __init__(self):
        self.dataframe: Optional[pl.DataFrame] = None # Use a more descriptive name, hint with Polars DataFrame

    def load_data(self, filepath: str = DEFAULT_DATA_FILEPATH, **kwargs):
        """
        Load data from a CSV file using Polars.

        Args:
            filepath (str): Path to the CSV file.
            **kwargs: Additional arguments for pl.read_csv (e.g., n_rows, columns).
                      Note: Polars uses different argument names than pandas.
                      'nrows' -> 'n_rows'
                      'usecols' -> 'columns'
                      'engine' is less relevant as Polars uses Arrow by default.
        """
        print(f"Loading data from: {filepath} using Polars")
        start_time = time.time()
        try:
            # Map common pandas kwargs to Polars if present
            if 'nrows' in kwargs:
                kwargs['n_rows'] = kwargs.pop('nrows')
            if 'usecols' in kwargs:
                kwargs['columns'] = kwargs.pop('usecols')
            if 'engine' in kwargs:
                print(f"Note: Polars uses Apache Arrow; 'engine' argument ('{kwargs.pop('engine')}') is ignored.")
                
            # Consider adding 'infer_schema_length=10000' or higher for better type inference on large files
            if 'infer_schema_length' not in kwargs:
                 kwargs['infer_schema_length'] = 10000 # Default heuristic
            if 'null_values' not in kwargs:
                kwargs['null_values'] = ["NULL"]
            self.dataframe = pl.read_csv(filepath, **kwargs)
            duration = time.time() - start_time
            print(f"Data loaded: {self.dataframe.shape} (rows, cols) in {duration:.2f} seconds.")

            if self.dataframe.is_empty(): # Use Polars method
                print("Warning: Loaded dataframe is empty.")

            return self.dataframe

        except FileNotFoundError:
            print(f"Error: File not found at {filepath}")
            self.dataframe = None
            return None
        except Exception as e: # Catch Polars specific errors if needed later (e.g., ComputeError)
            print(f"Error loading data: {e}")
            self.dataframe = None
            return None

    def drop_columns(self, columns: List[str]):
        """
        Drop specified columns from the dataframe.

        Args:
            columns (List[str]): List of column names to drop.
        """
        if self.dataframe is None:
            print("Error: No data loaded. Use load_data() first.")
            return
        if self.dataframe.is_empty(): # Use Polars method
            print("Error: Dataframe is empty. Cannot drop columns.")
            return

        print(f"Dropping columns: {columns}")
        # Polars drop returns a new DataFrame, assign it back
        # It implicitly ignores columns that don't exist
        cols_to_drop = [col for col in columns if col in self.dataframe.columns]
        if len(cols_to_drop) < len(columns):
            ignored = set(columns) - set(cols_to_drop)
            print(f"Warning: Columns not found and ignored: {list(ignored)}")
        
        if cols_to_drop: # Only drop if there are valid columns to drop
            self.dataframe = self.dataframe.drop(cols_to_drop)
            print(f"Remaining columns: {list(self.dataframe.columns)}")
            print(f"Dataframe shape after dropping columns: {self.dataframe.shape}")
        else:
            print("No valid columns specified to drop.")
    # --- Helper Methods ---

    def _plot_correlation_matrix(self, num_df: pl.DataFrame, sample_frac: float, save_plots: bool, plot_dir: str):
        """Plots the correlation matrix for the provided numerical Polars DataFrame."""
        print("\nCalculating correlation matrix...")
        start_time = time.time()

        df_corr = num_df
        n_samples = df_corr.height # Initialize n_samples to full height

        if 0.0 < sample_frac < 1.0 and df_corr.height > 2: # Need >1 for corr, >2 is safer for sampling
             # Ensure n_samples is at least 2
            n_samples_calc = max(2, int(df_corr.height * sample_frac))
            if n_samples_calc < df_corr.height:
                 print(f"Sampling {n_samples_calc} rows ({sample_frac*100:.1f}%) for correlation.")
                 # Polars sample uses 'fraction' or 'n', use 'n' for consistency with pandas code here
                 # Polars uses 'seed' instead of 'random_state'
                 df_corr = num_df.sample(n=n_samples_calc, seed=42, shuffle=True) # Use shuffle=True for random sampling
                 n_samples = n_samples_calc # Update n_samples if sampling occurred
            else:
                 print("Note: Sampling fraction resulted in full dataset size, using all rows.")
        elif sample_frac < 1.0:
             print("Note: Sampling skipped for correlation (too few rows or sample_frac <= 0).")

        if df_corr.is_empty() or df_corr.height < 2:
            print("Error: Not enough data points (after potential sampling) to calculate correlation.")
            return

        try:
            # Calculate correlation using Polars .corr()
            # Polars corr() handles non-numeric columns gracefully by default (ignores them)
            # but we already selected numeric columns, so it's fine.
            correlation_matrix_pl = df_corr.corr()
            duration = time.time() - start_time
            print(f"Correlation matrix calculated in {duration:.2f}s.")

            # Convert Polars DataFrame to Pandas DataFrame for Seaborn heatmap
            # Seaborn typically works best with Pandas or Numpy
            correlation_matrix_pd = correlation_matrix_pl.to_pandas()
            print("\nCorrelation Matrix (numerical values):")
            print(correlation_matrix_pd.round(3))  # Round for readability
            # Set index for heatmap labels if needed (Polars corr might not have index)
            if correlation_matrix_pd.index.name is None and correlation_matrix_pd.shape[0] == len(df_corr.columns):
                 correlation_matrix_pd.index = df_corr.columns # Assign column names as index
                 # Also ensure columns are set correctly after potential index reset
                 correlation_matrix_pd.columns = df_corr.columns


            plt.figure(figsize=(min(15, correlation_matrix_pd.shape[1]*0.6+2), min(12, correlation_matrix_pd.shape[0]*0.6+1)))
            sns.heatmap(correlation_matrix_pd, annot=False, cmap='coolwarm', linewidths=.2) # Slightly thinner lines
            plt.xticks(fontsize=26, rotation=45, ha='right')
            plt.yticks(fontsize=26)
            title = 'Correlation Matrix'
            if sample_frac < 1.0 and n_samples < num_df.height: # Check if sampling actually happened
                title += f' (Sampled {sample_frac*100:.1f}%)'
            plt.title(title)
            plt.tight_layout()
            self._save_or_show_plot(plt, save_plots, plot_dir, "correlation_matrix.png")

        except Exception as e:
            print(f"Error plotting correlation matrix: {e}")
            plt.close() # Ensure plot is closed on error

    def _plot_numerical_distributions(self, df: pl.DataFrame, num_cols: List[str], max_plots: Optional[int], save_plots: bool, plot_dir: str):
        """Plots histogram/KDE and Q-Q plot for numerical columns using Polars Series."""
        cols_to_plot = num_cols
        desc = "Numerical Distributions"
        if max_plots is not None and len(num_cols) > max_plots:
            print(f"\nPlotting distributions for first {max_plots} of {len(num_cols)} numerical features...")
            cols_to_plot = num_cols[:max_plots]
        else:
             print(f"\nPlotting distributions for {len(cols_to_plot)} selected numerical features...")


        for col in tqdm(cols_to_plot, desc=desc):
            if col not in df.columns: continue # Safety check

            try:
                plt.figure(figsize=(12, 5))

                # Select the column as a Polars Series and drop nulls
                # Use .drop_nulls() instead of .dropna()
                col_series = df[col].drop_nulls()

                # Plot 1: Histogram / KDE
                plt.subplot(1, 2, 1)
                if not col_series.is_empty(): # Use Polars is_empty()
                    # Convert Polars Series to NumPy array for Seaborn/Matplotlib
                    col_data_np = col_series.to_numpy()
                    # Simple bin calculation (use n_unique from Polars series)
                    n_unq = col_series.n_unique()
                    bins = min(50, max(10, n_unq)) if n_unq > 1 else 10
                    sns.histplot(col_data_np, kde=True, bins=bins)
                    plt.title(f'Distribution of {col}')
                else:
                    plt.title(f'Distribution of {col} (All Null)')
                    plt.text(0.5, 0.5, 'All values are Null', ha='center', va='center', transform=plt.gca().transAxes)

                # Plot 2: Q-Q Plot
                plt.subplot(1, 2, 2)
                plot_title = f'Q-Q Plot of {col}'
                # Use Polars n_unique()
                if not col_series.is_empty() and col_series.n_unique() > 1:
                    # Sample for large datasets for performance using Polars sample
                    qq_series = col_series
                    if col_series.len() > 5000:
                        qq_series = col_series.sample(n=min(col_series.len(), 5000), seed=42, shuffle=True)
                        plot_title += ' (Sampled)'

                    # Convert Polars Series to NumPy array for SciPy stats.probplot
                    qq_data_np = qq_series.to_numpy()
                    stats.probplot(qq_data_np, dist="norm", plot=plt)
                    plt.title(plot_title)
                elif not col_series.is_empty(): # Constant value case
                    plt.title(plot_title + ' (Constant)')
                    plt.text(0.5, 0.5, 'Constant value, cannot plot Q-Q', ha='center', va='center', transform=plt.gca().transAxes)
                else: # All Null case
                    plt.title(plot_title + ' (All Null)')
                    plt.text(0.5, 0.5, 'All values are Null', ha='center', va='center', transform=plt.gca().transAxes)

                plt.tight_layout()
                self._save_or_show_plot(plt, save_plots, plot_dir, f"dist_{col}.png")

            except Exception as e:
                print(f"Error plotting distribution for {col}: {e}")
                plt.close()

    def _analyze_categorical_features(self, df: pl.DataFrame, cat_cols: List[str], max_plots: Optional[int], threshold: int, plot_dist: bool, save_plots: bool, plot_dir: str):
        """Analyzes and optionally plots distributions for categorical features using Polars."""
        cols_to_analyze = cat_cols
        desc = "Categorical Analysis"
        if max_plots is not None and len(cat_cols) > max_plots:
            print(f"\nAnalyzing first {max_plots} of {len(cat_cols)} categorical features...")
            cols_to_analyze = cat_cols[:max_plots]
        else:
            print(f"\nAnalyzing {len(cols_to_analyze)} selected categorical features...")

        for col in tqdm(cols_to_analyze, desc=desc):
            if col not in df.columns: continue # Safety check

            try:
                # Use Polars n_unique()
                # Ensure column exists before accessing
                col_series = df[col]
                n_unique = col_series.n_unique()
                print(f"\n'{col}' (Unique values: {n_unique})")

                # Value Counts using Polars value_counts() -> returns DataFrame
                # Sort by count implicitly
                value_counts_df = col_series.value_counts()

                print("Value Counts (Top 5):")
                print(value_counts_df.head(5)) # Print head of DataFrame
                if n_unique > 5:
                    print("Value Counts (Bottom 5):")
                    # Need to sort ascending to get bottom N by count
                    print(value_counts_df.sort("counts", descending=False).head(5))

                # Plotting
                if plot_dist:
                    if 1 < n_unique <= threshold:
                        plt.figure(figsize=(min(12, max(6, n_unique*0.4)), max(5, min(8, n_unique*0.25 + 2))))
                        
                        n_bars_to_show = min(n_unique, 30)
                        # Get top N categories from the value_counts DataFrame
                        # The column name in value_counts_df is the original column name 'col'
                        top_n_cats_series = value_counts_df.head(n_bars_to_show)[col]

                        # --- Data Conversion for Seaborn ---
                        # Convert the relevant column portion to Pandas Series for Seaborn
                        # Also handle potential categorical type - convert to string for broad compatibility
                        plot_data_pd = df.select(pl.col(col).cast(pl.Utf8).alias(col)).to_pandas() # Use cast for safety
                        
                        # Get the list of top N categories as strings
                        top_n_cats_list = top_n_cats_series.cast(pl.Utf8).to_list()

                        # Filter the pandas series to only include top N categories for plotting order
                        plot_data_filtered_pd = plot_data_pd[plot_data_pd[col].isin(top_n_cats_list)]

                        # Use horizontal bars for > 10 categories usually looks better
                        if n_unique > 10:
                             # Pass the filtered pandas Series and the order list
                            sns.countplot(y=plot_data_filtered_pd[col], order=top_n_cats_list, palette="viridis")
                            plt.xlabel("Count")
                            plt.ylabel(col) # Y label is column name
                        else:
                            # Pass the filtered pandas Series and the order list
                            sns.countplot(x=plot_data_filtered_pd[col], order=top_n_cats_list, palette="viridis")
                            plt.xlabel(col) # X label is column name
                            plt.ylabel("Count")
                            plt.xticks(rotation=30, ha='right') # Rotate labels slightly

                        plot_title = f'Distribution of {col}'
                        if n_unique > n_bars_to_show:
                             plot_title += f' (Top {n_bars_to_show})'
                        plt.title(plot_title)
                        plt.tight_layout()
                        self._save_or_show_plot(plt, save_plots, plot_dir, f"cat_{col}.png")

                    elif n_unique > threshold:
                        print(f"  Skipping plot: Too many unique values ({n_unique} > {threshold}).")
                    else: # n_unique <= 1
                        print(f"  Skipping plot: Only one unique value or all Null.")

            except Exception as e:
                # Catch potential errors during Polars operations or plotting
                print(f"Error analyzing categorical feature {col}: {e}")
                plt.close()

    def _save_or_show_plot(self, plt_obj, save, directory, filename):
        """Saves or shows the current plot and closes it."""
        if save:
            os.makedirs(directory, exist_ok=True) # Simpler way to create dir

            # Basic filename sanitization (replace non-alphanumeric/dot/dash/underscore)
            safe_filename = "".join(c if c.isalnum() or c in ('_', '.', '-') else '_' for c in filename)
            # Avoid overly long filenames if column names are huge
            safe_filename = safe_filename[:200] + ".png" if len(safe_filename) > 200 else safe_filename
            filepath = os.path.join(directory, safe_filename)
            try:
                plt_obj.savefig(filepath, bbox_inches='tight', dpi=150)
                # print(f"  Plot saved: {filepath}") # Optional: uncomment for verbose output
            except Exception as e:
                print(f"  Error saving plot {filepath}: {e}")
            finally:
                plt_obj.close() # Always close after saving attempt
        else:
            try:
                plt_obj.show()
            except Exception as e:
                 print(f" Error showing plot: {e}")
            finally:
                 plt_obj.close() # Close after showing in non-saving mode

    def get_highly_correlated_columns(self, threshold: float = 0.7, sample_frac: float = 1.0):
        """
        Returns pairs of columns with absolute correlation above the given threshold.

        Args:
            threshold (float): Minimum absolute correlation to include (default 0.7).
            sample_frac (float): Fraction of data to sample for correlation.

        Returns:
            List[Tuple[str, str, float]]: List of (col1, col2, correlation) sorted by descending correlation.
        """
        if self.dataframe is None or self.dataframe.is_empty():
            return []

        num_df = self.dataframe.select(pl.selectors.numeric())

        if num_df.shape[1] < 2:
            return []

        df_corr = num_df
        if 0.0 < sample_frac < 1.0 and df_corr.height > 2:
            df_corr = num_df.sample(fraction=sample_frac, seed=42, shuffle=True)

        # Compute correlation matrix
        corr_matrix = df_corr.corr().to_pandas()
        corr_matrix.index = df_corr.columns
        corr_matrix.columns = df_corr.columns

        # Extract upper triangle pairs
        correlated_pairs = []
        for i, col1 in enumerate(corr_matrix.columns):
            for j, col2 in enumerate(corr_matrix.columns):
                if j > i:
                    corr_value = corr_matrix.loc[col1, col2]
                    if abs(corr_value) >= threshold:
                        correlated_pairs.append((col1, col2, corr_value))

        # Sort by absolute correlation descending
        correlated_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        return correlated_pairs

    def get_columns_with_high_missing(self, threshold: float = 0.8):
        """
        Returns columns with more than the given fraction of missing (NaN) values.

        Args:
            threshold (float): Fraction (0–1), e.g., 0.7 means >70% missing.

        Returns:
            List[str]: List of column names with high missingness.
        """
        if self.dataframe is None or self.dataframe.is_empty():
            return []

        total_rows = self.dataframe.height
        null_counts = self.dataframe.null_count().to_dict(as_series=False)
        high_missing_cols = []
        for col, count_list in null_counts.items():
            missing_fraction = count_list[0] / total_rows
            if missing_fraction > threshold:
                high_missing_cols.append(col)

        print(f"Columns with more than 80% missing values: {high_missing_cols}")
        return high_missing_cols

    def summarize_numerical_columns(self):
        """
        Prints the range, median, and mean for each numerical column.
        """
        if self.dataframe is None or self.dataframe.is_empty():
            print("Dataframe is empty or not loaded.")
            return

        num_cols = self.dataframe.select(pl.selectors.numeric()).columns

        if not num_cols:
            print("No numerical columns found.")
            return

        summary = []

        for col in num_cols:
            col_series = self.dataframe[col].drop_nulls()
            if col_series.is_empty():
                summary.append((col, None, None, None))
                continue

            min_val = col_series.min()
            max_val = col_series.max()
            median_val = col_series.median()
            mean_val = col_series.mean()

            summary.append((col, f"{min_val}–{max_val}", median_val, mean_val))

        print("\nNumerical Column Summary:")
        for col, col_range, median_val, mean_val in summary:
            print(f"Column: {col}")
            if col_range is None:
                print("  (All values are null)")
            else:
                print(f"  Range: {col_range}")
                print(f"  Median: {median_val}")
                print(f"  Mean: {mean_val}")

    def get_non_numerical_columns(self):
        """
        Returns a list of non-numerical (string, categorical, etc.) column names.
        """
        if self.dataframe is None or self.dataframe.is_empty():
            print("Dataframe is empty or not loaded.")
            return []

        non_numeric_cols = self.dataframe.select(~pl.selectors.numeric()).columns
        print("Non-numerical columns:", non_numeric_cols)
        return non_numeric_cols

    def get_columns_with_missing_values(self):
        """
        Prints all columns that have at least one missing (null) value.
        """
        if self.dataframe is None or self.dataframe.is_empty():
            print("Dataframe is empty or not loaded.")
            return []

        null_counts = self.dataframe.null_count().to_dict(as_series=False)

        cols_with_missing = [col for col, count_list in null_counts.items() if count_list[0] > 0]

        if cols_with_missing:
            print("Columns with missing values:")
            for col in cols_with_missing:
                print(f"  {col}: {null_counts[col][0]} missing")
        else:
            print("No columns have missing values.")

        return cols_with_missing


class FeatureEngineer:
    """
    Automatically imputes:
    - Mode for binary/small categorical numeric columns with missing values.
    - Median for continuous numeric columns with missing values.

    Target engineering (e.g., adding 'interaction_target') is now a separate method.
    """

    def __init__(self, df: pl.DataFrame):
        """
        Args:
            df (pl.DataFrame): The input dataframe.
        """
        self.df = df
        self.impute_values = {}
        self.binary_or_categorical_cols = []
        self.continuous_numeric_cols = []

    def transform(self) -> pl.DataFrame:
        print("\n=== FeatureEngineering (Auto-Imputation): Start ===")

        cols_with_missing = self._get_columns_with_missing()
        print(f"Found {len(cols_with_missing)} columns with missing values: {cols_with_missing}")

        for col in cols_with_missing:
            unique_vals = self.df[col].drop_nulls().unique().to_list()
            n_unique = len(unique_vals)
            dtype = self.df.schema[col]
            is_numeric = dtype in (
                pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                pl.Float32, pl.Float64
            )

            if is_numeric and n_unique in [2, 3]:
                # Binary or small categorical numeric (e.g., -1, 0, +1)
                mode_val = self._get_mode(col)
                self.df = self.df.with_columns(pl.col(col).fill_null(mode_val))
                self.impute_values[col] = mode_val
                self.binary_or_categorical_cols.append(col)
                print(f"Imputed {col} (binary/categorical numeric) with mode: {mode_val}")
            elif is_numeric:
                # Continuous numeric column
                median_val = self.df[col].median()
                self.df = self.df.with_columns(pl.col(col).fill_null(median_val))
                self.impute_values[col] = median_val
                self.continuous_numeric_cols.append(col)
                print(f"Imputed {col} (continuous numeric) with median: {median_val}")
            else:
                print(f"Skipped {col}: Not numeric (dtype={dtype}).")

        print("\n=== FeatureEngineering: Complete ===")
        print(f"Binary/categorical numeric columns imputed: {self.binary_or_categorical_cols}")
        print(f"Continuous numeric columns imputed: {self.continuous_numeric_cols}")

        return self.df
    
    # ------------------------------------------------------------
    # Search-level relative features
    # ------------------------------------------------------------
    def add_search_relative_features(self) -> pl.DataFrame:
        """
        Adds:
        - ft_price_rank  :  price rank (0=cheapest) within each srch_id, scaled 0-1
        - ft_price_rel   :  (price_usd – mean_price) / std_price   (z-score style)
        - ft_review_norm :  prop_review_score divided by max in search
        """
        # safety checks – skip gracefully if a column is missing
        required = {"srch_id", "price_usd", "prop_review_score"}
        missing  = required - set(self.df.columns)
        if missing:
            print(f"[add_search_relative_features] skipped – missing {missing}")
            return self.df

        # price rank (0..1, lower is cheaper)
        self.df = (
            self.df
            .with_columns(
                pl.col("price_usd")
                  .rank("dense")                         # 1,2,3,…
                  .over("srch_id")
                  .alias("tmp_rank")
            )
            .with_columns(
                (pl.col("tmp_rank") - 1)                 # 0-based
                / (pl.col("tmp_rank").max().over("srch_id") - 1)
                .fill_nan(0)                             # single-option searches
                .alias("ft_price_rank")
            )
            .drop("tmp_rank")
        )

        # price z-score within search
        self.df = self.df.with_columns(
            (
                (pl.col("price_usd") - pl.col("price_usd").mean().over("srch_id"))
                / pl.col("price_usd").std().over("srch_id")
            ).fill_nan(0).alias("ft_price_rel")
        )

        # review score normalised 0-1 within search
        self.df = self.df.with_columns(
            (
                pl.col("prop_review_score")
                / pl.col("prop_review_score").max().over("srch_id")
            ).fill_nan(0).alias("ft_review_norm")
        )

        print("✅ search-relative features added: "
              "ft_price_rank, ft_price_rel, ft_review_norm")
        return self.df


    def add_same_country_feature(self) -> pl.DataFrame:
        self.df = self.df.with_columns(
            (pl.col("visitor_location_country_id") == pl.col("prop_country_id"))
            .cast(pl.Int8)
            .alias("ft_same_country")
        )

        return self.df

    def add_avg_guest_count(self, how: str = "left") -> pl.DataFrame:
 
        if not isinstance(self.df, pl.DataFrame):
            self.df = pl.from_pandas(self.df)
 
        agg = (
            self.df
            .group_by("srch_id")
            .agg(
                (
                    pl.col("srch_adults_count") +
                    pl.col("srch_children_count")
                )
                .mean()
                .alias("ft_avg_guest_count")
            )
        ) 
        return self.df

    def add_interaction_target(self) -> pl.DataFrame:
        """
        Adds the 'interaction_target' column if 'booking_bool' and 'click_bool' are present.
        """
        print("\n=== FeatureEngineering (Target Engineering): Start ===")
        if 'booking_bool' in self.df.columns and 'click_bool' in self.df.columns:
            print("✅ Adding interaction_target column (combining booking_bool and click_bool)...")
            self.df = self.df.with_columns(
                pl.when(pl.col("booking_book")==1)
                .then(5)
                .when(pl.col("click_bool")==1)
                .then(1)
                .otherwise(0)
                .alias("interaction_target")
            )
            print("✅ interaction_target added!")
        else:
            print("⚠️ Skipping target engineering: booking_bool or click_bool not found.")
        return self.df
    
    # ------------------------------------------------------------
    # Extra hand-crafted features (price, stars, comps, length-of-stay)
    # ------------------------------------------------------------
    def add_extra_features(self) -> pl.DataFrame:
        """
        Adds 7 commonly useful features.  All computations are leakage-safe.
        """
        df = self.df  # shorthand

        # ---------- log(price) -----------------------------------
        if "price_usd" in df.columns:
            df = df.with_columns(
                pl.col("price_usd").log1p().alias("ft_log_price")
            )

        # ---------- price per person -----------------------------
        need_pp = {"price_usd", "srch_adults_count", "srch_children_count"}
        if need_pp <= set(df.columns):
            print("Adding ft_price_per_person (price per person)...")
            denom = (
                pl.col("srch_adults_count") + pl.col("srch_children_count")
            ).clip(lower_bound=1)          # guarantees ≥ 1

            df = df.with_columns(
                (pl.col("price_usd") / denom).alias("ft_price_per_person")
            )
 

        # ---------- star-rating difference ----------------------
        need_star = {"prop_starrating", "visitor_hist_starrating"}
        if need_star <= set(df.columns):
            print("Adding ft_star_diff (star rating difference)...")
            df = df.with_columns(
                (
                    pl.col("prop_starrating") - pl.col("visitor_hist_starrating")
                ).fill_null(0).alias("ft_star_diff")
            )

        # ---------- location score normalised within search -----
        if {"srch_id", "prop_location_score1"} <= set(df.columns):
            print("Adding ft_loc_score_norm (location score normalised)...")
            df = df.with_columns(
                (
                    pl.col("prop_location_score1")
                    / pl.col("prop_location_score1").max().over("srch_id")
                ).fill_nan(0).alias("ft_loc_score_norm")
            )

        # ---------- competitor cheaper count --------------------
        comp_rate_cols = [c for c in df.columns if c.startswith("comp") and c.endswith("_rate")]
        if comp_rate_cols:
            print("Adding ft_comp_cheaper (competitor cheaper count)...")
            df = df.with_columns(
                (
                    sum((pl.col(c) < 0).cast(pl.Int8) for c in comp_rate_cols)
                ).alias("ft_comp_cheaper")
            )

        # ---------- price percentile (0..1) ---------------------
        if {"srch_id", "price_usd"} <= set(df.columns):
            print("Adding ft_price_percentile (price percentile)...")
            df = (
                df
                .with_columns(
                    pl.col("price_usd")
                      .rank("dense")
                      .over("srch_id")
                      .alias("tmp_rank_p")
                )
                .with_columns(
                    (pl.col("tmp_rank_p") - 1)
                    / (pl.col("tmp_rank_p").max().over("srch_id") - 1)
                    .fill_nan(0)
                    .alias("ft_price_percentile")
                )
                .drop("tmp_rank_p")
            )

        self.df = df
        print("✅ extra features added")
        return self.df

    # ------------------------------------------------------------
    # 1.  Temporal features from date_time
    # ------------------------------------------------------------
    def add_temporal_features(self) -> pl.DataFrame:
        """
        Expects column 'date_time' in canonical Expedia format
        e.g. '2013-01-02 08:07:28'.
        Adds:
        - ft_month           : 1-12
        - ft_dow             : 0-6  (Mon=0)
        - ft_is_weekend      : 1 if Sat/Sun, else 0
        - ft_request_hour    : 0-23
        """

        if "date_time" not in self.df.columns:
            print("[add_temporal_features] skipped – 'date_time' missing")
            return self.df

        dt = pl.col("date_time").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")

        self.df = self.df.with_columns([
            dt.dt.month().alias("ft_month"),
            dt.dt.weekday().alias("ft_dow"),
            (dt.dt.weekday().is_in([5, 6])).cast(pl.Int8).alias("ft_is_weekend"),
            dt.dt.hour().alias("ft_request_hour"),
        ])

        print("✅ temporal features added: "
            "ft_month, ft_dow, ft_is_weekend, ft_request_hour")
        return self.df


    # ------------------------------------------------------------
    # 2.  Price vs historical baselines
    # ------------------------------------------------------------
    def add_price_history_features(self) -> pl.DataFrame:
        """
        Needs:
        - price_usd
        - visitor_hist_adr_usd      (user's avg daily rate)  – may be null
        - prop_log_historical_price (property's avg price)
        Adds:
        - ft_price_vs_user_hist     : price - user_hist
        - ft_price_vs_prop_hist     : price - prop_hist
        - ft_price_pct_over_hist    : (price / prop_hist) - 1
        """

        required_cols = {"price_usd",
                        "visitor_hist_adr_usd",
                        "prop_log_historical_price"}
        missing = required_cols - set(self.df.columns)
        if missing:
            print(f"[add_price_history_features] skipped – missing {missing}")
            return self.df

        self.df = self.df.with_columns([
            (pl.col("price_usd") - pl.col("visitor_hist_adr_usd"))
                .fill_null(0).alias("ft_price_vs_user_hist"),

            (pl.col("price_usd") - pl.col("prop_log_historical_price"))
                .alias("ft_price_vs_prop_hist"),

            ((pl.col("price_usd") / pl.col("prop_log_historical_price")) - 1)
                .alias("ft_price_pct_over_hist"),
        ])

        print("✅ price-history features added: "
            "ft_price_vs_user_hist, ft_price_vs_prop_hist, "
            "ft_price_pct_over_hist")
        return self.df

 
    def evaluate_before_and_after(
        self,
        before_cols: List[str],
        after_cols:  List[str],
        *,
        mode: str            = "fast",   # "fast"  or "full"
        sample_frac: float   = 0.30,     # only used in fast-mode
        target_col: str      = "interaction_target",
        group_col:  str      = "srch_id",
        random_state: int    = 42,
    ) -> dict:
        """
        Quickly decide whether a new feature set helps.

        mode="fast"  ➜  CPU LightGBM ranker, 64 trees, optional row-sample
        mode="full"  ➜  Your 1 500-tree GPU LambdaMART (default)

        Returns
        -------
        {"before": {"ndcg@5": float}, "after": {"ndcg@5": float}}
        """

        # ---- cheap validations ---------------------------------------
        for name, cols in (("before", before_cols), ("after", after_cols)):
            missing = [c for c in cols if c not in self.df.columns]
            if missing:
                raise ValueError(f"{name} columns missing: {missing}")
        for col in (target_col, group_col):
            if col not in self.df.columns:
                raise ValueError(f"Column '{col}' not found in dataframe")

        # ---- optional row sampling for very fast turnaround ----------
        df_eval = (
            self.df.sample(fraction=sample_frac, seed=random_state)
            if mode == "fast" and 0 < sample_frac < 1.0
            else self.df
        )

        y       = df_eval[target_col].to_pandas()
        groups  = df_eval[group_col].to_pandas()

        # keep the split identical for before/after
        from sklearn.model_selection import GroupShuffleSplit
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
        tr_idx, va_idx = next(gss.split(X=np.zeros(len(y)), y=y, groups=groups))

        def _to_group_sizes(sr):
            return sr.value_counts(sort=False).sort_index().values

        # ---- parameter presets ---------------------------------------
        if mode == "fast":
            lgb_params = dict(
                objective      = "lambdarank",
                metric         = "ndcg",
                eval_at        = [5],
                n_estimators   = 64,
                learning_rate  = 0.2,
                num_leaves     = 31,
                max_depth      = 4,
                subsample      = 0.8,
                colsample_bytree = 0.8,
                device_type    = "gpu",      # feather-weight
                max_bin        = 63,
                random_state   = random_state,
                n_jobs         = -1,
            )
        else:  # full GPU block (same as your HotelRanker defaults)
            lgb_params = dict(
                objective      = "lambdarank",
                metric         = "ndcg",
                eval_at        = [5],
                n_estimators   = 1500,
                learning_rate  = 0.03,
                num_leaves     = 255,
                subsample      = 0.8,
                colsample_bytree = 0.8,
                device_type    = "gpu",
                max_bin        = 63,
                random_state   = random_state,
                n_jobs         = -1,
            )

        from lightgbm import LGBMRanker, log_evaluation

        def _score(feature_cols, tag):
            X = df_eval.select(feature_cols).to_pandas()
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

            g_tr = _to_group_sizes(groups.iloc[tr_idx])
            g_va = _to_group_sizes(groups.iloc[va_idx])

            model = LGBMRanker(**lgb_params)
            model.fit(
                X_tr, y_tr,
                group=g_tr,
                eval_set=[(X_va, y_va)],
                eval_group=[g_va],
                callbacks=[log_evaluation(0)],        # silence output
            )
            ndcg5 = model.best_score_["valid_0"]["ndcg@5"]
            print(f"{tag:<7s} | NDCG@5 = {ndcg5:.4f}  ({mode}-mode)")
            return ndcg5

        ndcg_before = _score(before_cols, "before")
        ndcg_after  = _score(after_cols,  "after")

        diff = ndcg_after - ndcg_before
        print(f"\nΔNDCG@5 = {diff:+.4f}  →  {'IMPROVED' if diff>0 else 'WORSE'}")

        return {"before": {"ndcg@5": ndcg_before},
                "after":  {"ndcg@5": ndcg_after},
                "delta":  diff}

    def _get_columns_with_missing(self) -> list:
        null_counts = self.df.null_count().to_dict(as_series=False)
        cols_with_missing = [col for col, count_list in null_counts.items() if count_list[0] > 0]
        return cols_with_missing

    def _get_mode(self, col: str):
        vc = self.df[col].drop_nulls().value_counts()
        if vc.is_empty():
            return None

        # Dynamically get column names
        colnames = vc.columns
        value_col = colnames[0]
        count_col = colnames[1]

        mode_val = vc.sort(count_col, descending=True)[value_col][0]
        return mode_val


class HotelRanker:
    def __init__(
        self,
        df: pl.DataFrame,
        target_col: str = "interaction_target",
        exclude_cols: Optional[List[str]] = None,
        lgbm_params: Optional[dict] = None,        # <- new
    ):
        self.df = df
        self.target_col = target_col
        self.exclude_cols = exclude_cols or []

        # --- LightGBM ranker --------------------------------------------------
        default_params = dict(
            objective      = "lambdarank",
            metric         = "ndcg",
            eval_at        = [1, 3, 5],
            n_estimators   = 2000,        # more trees, smaller lr on GPU
            learning_rate  = 0.03,
            num_leaves     = 255,         # power of two works well with GPU
            max_depth      = -1,
            subsample      = 0.8,
            colsample_bytree = 0.8,
            # ---------- GPU switches ----------
            device_type    = "gpu",       # <-- activates GPU learner
            gpu_platform_id= 0,           # change if multiple OpenCL platforms
            gpu_device_id  = 0,           # pick your GPU
            # Smaller histograms speed up GPU even more
            max_bin        = 63,          # 2⁶-1; try 127 if accuracy drops
            random_state   = 42,
            n_jobs         = -1,
)
        if lgbm_params:
            default_params.update(lgbm_params)

        self.model = LGBMRanker(**default_params)  # <-- replaces RandomForest

    def _prepare_data(self, return_groups: bool = False):
        all_cols = self.df.columns
        exclude = set([self.target_col, "srch_id", "prop_id"] + self.exclude_cols)
        feature_cols = [c for c in all_cols if c not in exclude]

        X = self.df.select(feature_cols).to_pandas()
        y = self.df[self.target_col].to_pandas()

        if return_groups:
            groups = self.df["srch_id"].to_pandas()    # 1-row-per-hotel
            return X, y, groups
        return X, y

    def train(self):
        # Keep whole searches together in the split
        X, y, groups = self._prepare_data(return_groups=True)

        gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
        train_idx, val_idx = next(gss.split(X, y, groups))

        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val,   y_val   = X.iloc[val_idx],   y.iloc[val_idx]

        grp_train = groups.iloc[train_idx]
        grp_val   = groups.iloc[val_idx]

        # LightGBM wants "number of rows in each group"  ➜  np.array([...])
        def _to_group_sizes(series):
            return series.value_counts(sort=False).sort_index().values

        group_train_sizes = _to_group_sizes(grp_train)
        group_val_sizes   = _to_group_sizes(grp_val)

        print("Training LightGBM LambdaMART ranker …")
        self.model.fit(
            X_train,
            y_train,
            group=group_train_sizes,
            eval_set=[(X_val, y_val)],
            eval_group=[group_val_sizes],
        )
        best_ndcg5 = self.model.best_score_["valid_0"]["ndcg@5"]
        print(f"Best NDCG@5 on validation: {best_ndcg5:.4f}")


    def predict(self, df_test: pl.DataFrame) -> pl.DataFrame:
        all_cols = df_test.columns
        exclude  = set(["srch_id", "prop_id"] + self.exclude_cols)
        feature_cols = [c for c in all_cols if c not in exclude]

        X_test = df_test.select(feature_cols).to_pandas()
        scores = self.model.predict(X_test)        # 1-D relevance scores

        return pl.DataFrame(
            {
                "srch_id": df_test["srch_id"],
                "prop_id": df_test["prop_id"],
                "score":   scores,
            }
        )


    def export_ranking(self, predictions: pl.DataFrame, filename: str = "submission.csv"):
        """
        Exports the predictions in Kaggle format (SearchId, PropertyId), sorted by score.
        """
        print(f"Exporting predictions to {filename}...")

        # Rank properties per search_id by score descending
        submission = (
            predictions
            .sort(['srch_id', 'score'], descending=[False, True])
            .select(['srch_id', 'prop_id'])
            .rename({'srch_id': 'SearchId', 'prop_id': 'PropertyId'})
        )

        # Save to CSV
        submission.write_csv(filename)
        print(f"✅ Saved submission file: {filename}")