# eda_analyzer.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import time
import os
from typing import Optional, List

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
        self.dataframe = None # Use a more descriptive name

    def load_data(self, filepath: str = DEFAULT_DATA_FILEPATH, **kwargs):
        """
        Load data from a CSV file.

        Args:
            filepath (str): Path to the CSV file.
            **kwargs: Additional arguments for pd.read_csv (e.g., nrows, usecols).
        """
        print(f"Loading data from: {filepath}")
        start_time = time.time()
        try:
            # Simplified engine selection - use default unless pyarrow explicitly requested or beneficial
            engine = kwargs.pop('engine', 'c') # Default to 'c'
            if engine == 'pyarrow':
                try:
                    import pyarrow
                    print("Using 'pyarrow' engine.")
                except ImportError:
                    print("Warning: 'pyarrow' engine requested but pyarrow not installed. Falling back to 'c'.")
                    engine = 'c'

            self.dataframe = pd.read_csv(filepath, engine=engine, **kwargs)
            duration = time.time() - start_time
            print(f"Data loaded: {self.dataframe.shape} (rows, cols) in {duration:.2f} seconds.")

            if self.dataframe.empty:
                print("Warning: Loaded dataframe is empty.")

            return self.dataframe

        except FileNotFoundError:
            print(f"Error: File not found at {filepath}")
            self.dataframe = None
            return None
        except Exception as e:
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
        if self.dataframe.empty:
            print("Error: Dataframe is empty. Cannot drop columns.")
            return

        print(f"Dropping columns: {columns}")
        self.dataframe.drop(columns=columns, inplace=True, errors='ignore')
        print(f"Remaining columns: {list(self.dataframe.columns)}")
        print(f"Dataframe shape after dropping columns: {self.dataframe.shape}")

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
        Perform Exploratory Data Analysis on loaded data.

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
        if self.dataframe.empty:
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
            df_analyze = self.dataframe[columns] # Work on a view/copy of selected columns
            analysis_desc = f"specified columns ({len(columns)})"
            # If specific columns are given, analyze all of them, ignore limits
            num_limit = None
            cat_limit = None
        else:
            df_analyze = self.dataframe # Work on the whole dataframe
            analysis_desc = "full dataset"
            num_limit = max_num_plots
            cat_limit = max_cat_plots

        print(f"\n=== Starting EDA for {analysis_desc} ===")

        # --- Basic Info ---
        print("\n--- Basic Information ---")
        print(f"Shape: {df_analyze.shape}")
        print("Columns:", list(df_analyze.columns))
        print("\nFirst 5 rows:")
        try:
             from IPython.display import display, HTML
             display(HTML(df_analyze.head().to_html()))
        except ImportError:
             print(df_analyze.head())

        # --- Missing Values ---
        print("\n--- Missing Values ---")
        missing_info = df_analyze.isnull().sum()
        missing_info = missing_info[missing_info > 0].sort_values(ascending=False) # Only show columns with missing
        if not missing_info.empty:
            missing_perc = (missing_info / len(df_analyze)) * 100
            missing_df = pd.DataFrame({'Missing Count': missing_info, 'Missing (%)': missing_perc.round(2)})
            print(missing_df)
            print(f"Total missing values in selection: {missing_info.sum()}")
        else:
            print("No missing values found in the selected columns.")

        # --- Identify Column Types in Selection ---
        numerical_cols = df_analyze.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df_analyze.select_dtypes(include=['object', 'category']).columns.tolist()

        # --- Numerical Analysis ---
        if numerical_cols:
            print(f"\n--- Numerical Features ({len(numerical_cols)}) ---")
            print("Summary Statistics:")
            print(df_analyze[numerical_cols].describe().to_string())

            # Correlation
            if plot_correlation:
                if len(numerical_cols) > 1:
                     # Pass the actual numerical data subset to the helper
                    self._plot_correlation_matrix(df_analyze[numerical_cols], corr_sample_frac, save_plots, plot_dir)
                elif len(numerical_cols) == 1:
                    print("Skipping correlation: Only one numerical column selected.")
                # No else needed if len is 0, already handled by outer 'if numerical_cols'

            # Distributions
            if plot_distributions:
                 # Pass the original dataframe (or selection) and the list of cols
                self._plot_numerical_distributions(df_analyze, numerical_cols, num_limit, save_plots, plot_dir)
        elif not df_analyze.empty: # Only mention if there was data to analyze
             print("\n--- No numerical features found in selection ---")

        # --- Categorical Analysis ---
        if categorical_cols:
            print(f"\n--- Categorical Features ({len(categorical_cols)}) ---")
            self._analyze_categorical_features(df_analyze, categorical_cols, cat_limit, cat_plot_threshold, plot_distributions, save_plots, plot_dir)
        elif not df_analyze.empty:
            print("\n--- No categorical features found in selection ---")

        duration = time.time() - start_time
        print(f"\n=== EDA completed in {duration:.2f} seconds ===")

    # --- Helper Methods ---

    def _plot_correlation_matrix(self, num_df, sample_frac, save_plots, plot_dir):
        """Plots the correlation matrix for the provided numerical DataFrame."""
        print("\nCalculating correlation matrix...")
        start_time = time.time()

        df_corr = num_df
        # Simplified sampling logic
        if 0.0 < sample_frac < 1.0 and len(num_df) > 2: # Need >1 for corr, >2 is safer for sampling
            n_samples = max(2, int(len(num_df) * sample_frac))
            print(f"Sampling {n_samples} rows ({sample_frac*100:.1f}%) for correlation.")
            df_corr = num_df.sample(n=n_samples, random_state=42)
        elif sample_frac < 1.0:
             print("Note: Sampling skipped for correlation (too few rows or sample_frac <= 0).")


        try:
            # Calculate correlation only on the (potentially sampled) data
            correlation_matrix = df_corr.corr()
            duration = time.time() - start_time
            print(f"Correlation matrix calculated in {duration:.2f}s.")

            plt.figure(figsize=(min(15, len(correlation_matrix)*0.6+2), min(12, len(correlation_matrix)*0.6+1))) # Adjusted sizing
            sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=.2) # Slightly thinner lines
            title = 'Correlation Matrix'
            if sample_frac < 1.0 and df_corr.shape[0] < num_df.shape[0]: # Check if sampling actually happened
                title += f' (Sampled {sample_frac*100:.1f}%)'
            plt.title(title)
            plt.tight_layout()
            self._save_or_show_plot(plt, save_plots, plot_dir, "correlation_matrix.png")

        except Exception as e:
            print(f"Error plotting correlation matrix: {e}")
            plt.close() # Ensure plot is closed on error

    def _plot_numerical_distributions(self, df, num_cols, max_plots, save_plots, plot_dir):
        """Plots histogram/KDE and Q-Q plot for numerical columns."""
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
                col_data = df[col].dropna()

                # Plot 1: Histogram / KDE
                plt.subplot(1, 2, 1)
                if not col_data.empty:
                    # Simple bin calculation
                    bins = min(50, max(10, len(col_data.unique()))) if col_data.nunique() > 1 else 10
                    sns.histplot(col_data, kde=True, bins=bins)
                    plt.title(f'Distribution of {col}')
                else:
                    plt.title(f'Distribution of {col} (All NaN)')
                    plt.text(0.5, 0.5, 'All values are NaN', ha='center', va='center', transform=plt.gca().transAxes)

                # Plot 2: Q-Q Plot
                plt.subplot(1, 2, 2)
                plot_title = f'Q-Q Plot of {col}'
                if not col_data.empty and col_data.nunique() > 1:
                    # Sample for large datasets for performance
                    qq_data = col_data.sample(min(len(col_data), 5000), random_state=42) if len(col_data) > 5000 else col_data
                    if len(qq_data) < len(col_data):
                         plot_title += ' (Sampled)'
                    stats.probplot(qq_data, dist="norm", plot=plt)
                    plt.title(plot_title)
                elif not col_data.empty: # Constant value case
                    plt.title(plot_title + ' (Constant)')
                    plt.text(0.5, 0.5, 'Constant value, cannot plot Q-Q', ha='center', va='center', transform=plt.gca().transAxes)
                else: # All NaN case
                    plt.title(plot_title + ' (All NaN)')
                    plt.text(0.5, 0.5, 'All values are NaN', ha='center', va='center', transform=plt.gca().transAxes)

                plt.tight_layout()
                self._save_or_show_plot(plt, save_plots, plot_dir, f"dist_{col}.png")

            except Exception as e:
                print(f"Error plotting distribution for {col}: {e}")
                plt.close()


    def _analyze_categorical_features(self, df, cat_cols, max_plots, threshold, plot_dist, save_plots, plot_dir):
        """Analyzes and optionally plots distributions for categorical features."""
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
                n_unique = df[col].nunique()
                print(f"\n'{col}' (Unique values: {n_unique})")

                # Value Counts (more concise display)
                value_counts = df[col].value_counts()
                print("Value Counts (Top 5):")
                print(value_counts.head().to_string())
                if n_unique > 5:
                    print("Value Counts (Bottom 5):")
                    print(value_counts.tail().to_string())
                # No need to print all if between 5 and 10, top/bottom covers it

                # Plotting
                if plot_dist:
                    if 1 < n_unique <= threshold:
                        plt.figure(figsize=(min(12, max(6, n_unique*0.4)), max(5, min(8, n_unique*0.25 + 2)))) # Adjusted fig size
                        # Limit bars shown for readability, even if under threshold
                        n_bars_to_show = min(n_unique, 30)
                        top_n_cats = value_counts.index[:n_bars_to_show]

                        # Use horizontal bars for > 10 categories usually looks better
                        if n_unique > 10:
                            sns.countplot(y=df[col].astype(str), order=top_n_cats, palette="viridis") # Use y for horizontal
                            plt.xlabel("Count")
                            plt.ylabel(col) # Y label is column name
                        else:
                            sns.countplot(x=df[col].astype(str), order=top_n_cats, palette="viridis") # Use x for vertical
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
                        print(f"  Skipping plot: Only one unique value or all NaN.")

            except Exception as e:
                print(f"Error analyzing categorical feature {col}: {e}")
                plt.close()


    def _save_or_show_plot(self, plt_obj, save, directory, filename):
        """Saves or shows the current plot and closes it."""
        if save:
            # Ensure directory exists
            os.makedirs(directory, exist_ok=True) # Simpler way to create dir

            # Basic filename sanitization (replace non-alphanumeric/dot/dash/underscore)
            safe_filename = "".join(c if c.isalnum() or c in ('_', '.', '-') else '_' for c in filename)
            filepath = os.path.join(directory, safe_filename)
            try:
                plt_obj.savefig(filepath, bbox_inches='tight', dpi=150)
                # print(f"  Plot saved: {filepath}") # Optional: uncomment for verbose output
            except Exception as e:
                print(f"  Error saving plot {filepath}: {e}")
            finally:
                plt_obj.close() # Always close after saving attempt
        else:
            plt_obj.show()
            plt_obj.close() # Close after showing in non-saving mode

 
