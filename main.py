# eda_analyzer.py

import polars as pl # Changed import
import polars.selectors as cs # Import selectors for easier dtype selection
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import time
import os
from typing import Optional, List, Union # Added Union for type hints

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


    def run_eda(self,
            columns: Optional[List[str]] = None, # Renamed for clarity
            plot_distributions: bool = True,
            plot_correlation: bool = True,
            max_num_plots: Optional[int] = 10,
            max_cat_plots: Optional[int] = 10,
            corr_sample_frac: float = 1.0,
            cat_plot_threshold: int = 50,
            save_plots: bool = False,
            plot_dir: str = DEFAULT_PLOT_DIR):
        """
        Perform Exploratory Data Analysis on loaded data using Polars.

        Args:
            columns (List[str], optional): Specific columns to analyze. Defaults to all.
            plot_distributions (bool): Generate distribution plots.
            plot_correlation (bool): Generate correlation heatmap (for numerical columns).
            max_num_plots (int, optional): Max numerical distribution plots if analyzing all columns. None for no limit.
            max_cat_plots (int, optional): Max categorical count plots if analyzing all columns. None for no limit.
            corr_sample_frac (float): Fraction of data for correlation calculation (0.0 to 1.0).
            cat_plot_threshold (int): Max unique values for categorical plots.
            save_plots (bool): Save plots to disk instead of displaying.
            plot_dir (str): Directory for saved plots.
        """
        start_time = time.time()
        if self.dataframe is None:
            print("Error: No data loaded. Use load_data() first.")
            return
        if self.dataframe.is_empty(): # Use Polars method
            print("Error: Dataframe is empty. Cannot perform EDA.")
            return

        # --- Determine Columns to Analyze ---
        if columns:
            # Validate provided columns
            invalid_cols = [col for col in columns if col not in self.dataframe.columns]
            if invalid_cols:
                print(f"Error: Invalid columns specified: {invalid_cols}")
                print(f"Available columns: {list(self.dataframe.columns)}")
                return
            # Select columns using Polars select
            try:
                df_analyze = self.dataframe.select(columns)
            except pl.ColumnNotFoundError as e:
                 print(f"Error selecting columns: {e}") # Should be caught above, but safety
                 return
            analysis_desc = f"specified columns ({len(columns)})"
            # If specific columns are given, analyze all of them, ignore limits
            num_limit = None
            cat_limit = None
        else:
            df_analyze = self.dataframe # Work on the whole dataframe (it's a reference)
            analysis_desc = "full dataset"
            num_limit = max_num_plots
            cat_limit = max_cat_plots

        print(f"\n=== Starting EDA for {analysis_desc} ===")

        # --- Basic Info ---
        print("\n--- Basic Information ---")
        print(f"Shape: {df_analyze.shape}")
        print("Columns:", list(df_analyze.columns))
        # Display schema for dtype info
        print("\nSchema (Data Types):")
        # Use Polars schema pretty printing if possible, fallback to simple dict
        try:
             # Polars >= 0.20 uses __repr__ for nice printing
             print(df_analyze.schema)
        except AttributeError:
             # Older Polars or fallback
             print(dict(zip(df_analyze.columns, df_analyze.dtypes)))

        print("\nFirst 5 rows:")
        # Polars head() returns a DataFrame, print it directly
        # No easy HTML conversion built-in like pandas, just print
        print(df_analyze.head())


        # --- Missing Values ---
        print("\n--- Missing Values ---")
        # Use Polars null_count() which returns a DataFrame
        missing_info_df = df_analyze.null_count()

        # If null_count returns all zeros, it might be a single row df. Need to melt.
        if missing_info_df.shape[0] == 1:
             missing_info_df = missing_info_df.melt(variable_name="column", value_name="Missing Count")
        else:
             # Should not happen with standard null_count, but handle defensively
             print("Unexpected format for null_count output.")
             missing_info_df = pl.DataFrame({"column": [], "Missing Count": []}, schema={"column": pl.Utf8, "Missing Count": pl.UInt32})


        # Filter out columns with no missing values
        missing_info_df = missing_info_df.filter(pl.col("Missing Count") > 0)

        if not missing_info_df.is_empty():
            # Calculate percentages and sort
            total_rows = df_analyze.height
            missing_info_df = missing_info_df.with_columns(
                (pl.col("Missing Count") / total_rows * 100).round(2).alias("Missing (%)")
            ).sort("Missing Count", descending=True)

            print(missing_info_df) # Print the Polars DataFrame
            total_missing = missing_info_df["Missing Count"].sum()
            print(f"Total missing values in selection: {total_missing}")
        else:
            print("No missing values found in the selected columns.")

        # --- Identify Column Types in Selection ---
        # Use Polars selectors or dtype checks
        numerical_cols = df_analyze.select(cs.numeric()).columns
        # Polars treats strings as Utf8, also check for Categorical
        categorical_cols = df_analyze.select(cs.string() | cs.categorical()).columns
        # Note: cs.string() is equivalent to pl.col(pl.Utf8)

        # --- Numerical Analysis ---
        if numerical_cols:
            print(f"\n--- Numerical Features ({len(numerical_cols)}) ---")
            print("Summary Statistics:")
            # Polars describe() returns a DataFrame, print it
            print(df_analyze.select(numerical_cols).describe())

            # Correlation
            if plot_correlation:
                if len(numerical_cols) > 1:
                     # Pass the actual numerical data subset to the helper
                    self._plot_correlation_matrix(df_analyze.select(numerical_cols), corr_sample_frac, save_plots, plot_dir)
                elif len(numerical_cols) == 1:
                    print("Skipping correlation: Only one numerical column selected.")
                # No else needed if len is 0, already handled by outer 'if numerical_cols'

            # Distributions
            if plot_distributions:
                 # Pass the Polars dataframe and the list of numerical cols
                 # The helper will handle potential Series conversions for plotting libraries
                self._plot_numerical_distributions(df_analyze, numerical_cols, num_limit, save_plots, plot_dir)
        elif not df_analyze.is_empty(): # Only mention if there was data to analyze
             print("\n--- No numerical features found in selection ---")

        # --- Categorical Analysis ---
        if categorical_cols:
            print(f"\n--- Categorical Features ({len(categorical_cols)}) ---")
            self._analyze_categorical_features(df_analyze, categorical_cols, cat_limit, cat_plot_threshold, plot_distributions, save_plots, plot_dir)
        elif not df_analyze.is_empty():
            print("\n--- No categorical features found in selection ---")

        duration = time.time() - start_time
        print(f"\n=== EDA completed in {duration:.2f} seconds ===")

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
            # Set index for heatmap labels if needed (Polars corr might not have index)
            if correlation_matrix_pd.index.name is None and correlation_matrix_pd.shape[0] == len(df_corr.columns):
                 correlation_matrix_pd.index = df_corr.columns # Assign column names as index
                 # Also ensure columns are set correctly after potential index reset
                 correlation_matrix_pd.columns = df_corr.columns


            plt.figure(figsize=(min(15, correlation_matrix_pd.shape[1]*0.6+2), min(12, correlation_matrix_pd.shape[0]*0.6+1)))
            sns.heatmap(correlation_matrix_pd, annot=False, cmap='coolwarm', linewidths=.2) # Slightly thinner lines
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
 