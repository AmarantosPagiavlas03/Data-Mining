# eda_analyzer.py

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import time
import os
from typing import Optional, List
import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

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
    
    def add_avg_location_score(self, how="left") -> pl.DataFrame:
        if not isinstance(self.df, pl.DataFrame):
            self.df = pl.DataFrame(self.df)
        agg = (
            self.df
            .group_by("prop_id")                             
            .agg(
                (pl.col("prop_location_score1") +
                 pl.col("prop_location_score2"))
                .mean()
                .alias("avg_location_score")
            )
        )
        self.df = (
            self.df
            .join(agg, on="prop_id", how=how)
            .drop(["prop_location_score1", "prop_location_score2"])
        )
        return self.df

    def add_same_country_feature(self) -> pl.DataFrame:
        self.df = self.df.with_columns(
            (pl.col("visitor_location_country_id") == pl.col("prop_country_id"))
            .cast(pl.Int8)
            .alias("same_country")
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
                .alias("avg_guest_count")
            )
        ) 
        self.df = (
            self.df
            .join(agg, on="srch_id", how=how)
            .drop(["srch_adults_count", "srch_children_count"])
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
                (
                        (pl.col("booking_bool") * 2) +
                        ((pl.col("click_bool") == 1) & (pl.col("booking_bool") == 0)).cast(pl.Int8)
                ).alias("interaction_target")
            )
            print("✅ interaction_target added!")
        else:
            print("⚠️ Skipping target engineering: booking_bool or click_bool not found.")
        return self.df

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
    def __init__(self, df: pl.DataFrame, target_col: str = "interaction_target", exclude_cols: Optional[List[str]] = None):
        self.df = df
        self.target_col = target_col
        self.exclude_cols = exclude_cols if exclude_cols else []
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def _prepare_data(self):
        """
        Splits into X and y, drops non-feature columns (like search_id etc.)
        """
        print("Preparing data...")

        all_cols = self.df.columns

        # Define columns to exclude: target + ID cols + user exclusions
        exclude = set([self.target_col, 'srch_id', 'prop_id'] + self.exclude_cols)
        feature_cols = [col for col in all_cols if col not in exclude]

        print(f"Using {len(feature_cols)} feature columns: {feature_cols[:5]}... (and more)")

        X = self.df.select(feature_cols).to_pandas()
        y = self.df[self.target_col].to_pandas()

        return X, y

    def train(self):
        """
        Trains the model.
        """
        X, y = self._prepare_data()
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        print("Training model...")
        self.model.fit(X_train, y_train)
        score = self.model.score(X_val, y_val)
        print(f"Validation Accuracy: {score:.4f}")

    def predict(self, df_test: pl.DataFrame) -> pl.DataFrame:
        """
        Predicts scores on the test data and returns Polars DataFrame with srch_id, prop_id, score.
        """
        print("Predicting scores for ranking...")

        all_cols = df_test.columns
        exclude = set(['srch_id', 'prop_id'] + self.exclude_cols)
        feature_cols = [col for col in all_cols if col not in exclude]

        X_test = df_test.select(feature_cols).to_pandas()
        scores = self.model.predict_proba(X_test)[:, 1]  # Probability of booking = 1

        # Return Polars DataFrame for ranking
        result = pl.DataFrame({
            'srch_id': df_test['srch_id'],
            'prop_id': df_test['prop_id'],
            'score': scores
        })

        return result

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