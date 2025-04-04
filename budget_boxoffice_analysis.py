# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import re # Regular expressions for cleaning
# Import formatter for clearer axis labels with large numbers
from matplotlib.ticker import FuncFormatter

# --- Configuration ---
# !! IMPORTANT: YOU MUST UPDATE these column names based on YOUR CSV file !!
FILENAME = 'imdb_top250_movies.csv'
NAME_COL = 'name'           # CHANGE if your column name is different
YEAR_COL = 'year'           # CHANGE if different
RATING_COL = 'rating'       # CHANGE if different
BUDGET_COL = 'budget'       # CHANGE if different
BOX_OFFICE_COL = 'box_office' # CHANGE if different

# --- Helper Function for Cleaning Currency Columns ---
def clean_currency(value):
    """Attempts to clean currency strings ($ ,) and convert to float."""
    if pd.isna(value):
        return np.nan
    s_value = str(value).strip()
    s_value = s_value.replace('$', '').replace(',', '')
    # Add more specific cleaning rules here if needed (e.g., handle ' million', ' B', etc.)
    # Example: Convert '1.5M' to 1500000 (more robust cleaning might be needed)
    # if 'M' in s_value or 'm' in s_value:
    #     s_value = s_value.replace('M','').replace('m','')
    #     try: return float(s_value) * 1_000_000
    #     except ValueError: return np.nan
    # if 'B' in s_value or 'b' in s_value:
    #     ... similar logic for Billions ...
    try:
        return float(s_value)
    except ValueError:
        return np.nan

# --- Helper Function for Formatting Large Numbers ---
def format_millions(x, pos):
    'The two args are the value and tick position'
    # Format as millions with one decimal place
    if x >= 1e6:
        return f'{x*1e-6:1.1f}M'
    elif x >= 1e3:
         return f'{x*1e-3:1.0f}K' # Show thousands if less than a million
    else:
        return f'{x:1.0f}'

million_formatter = FuncFormatter(format_millions)


# --- 1. Load Data ---
# (Loading code remains the same as before)
print(f"Attempting to load dataset: {FILENAME}")
try:
    df = pd.read_csv(FILENAME)
    print(f"Dataset loaded successfully. Initial shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: File '{FILENAME}' not found.")
    print("Ensure the CSV is in the same directory and filenames/column names in the script are correct.")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred loading CSV: {e}")
    sys.exit(1)

print("\n--- Initial Data Overview ---")
print("First 5 rows (relevant columns):")
relevant_cols_for_display = [NAME_COL, YEAR_COL, RATING_COL, BUDGET_COL, BOX_OFFICE_COL]
display_cols = [col for col in relevant_cols_for_display if col in df.columns]
print(df[display_cols].head())
print("\nDataset Info:")
df.info()


# --- 2. Data Cleaning & Preparation ---
# (Cleaning code remains largely the same as before)
print("\n--- Data Cleaning ---")
essential_financial = [BUDGET_COL, BOX_OFFICE_COL]
missing_financial_data = False # Flag if financial data is missing/unusable
if BUDGET_COL and BUDGET_COL in df.columns:
    df['Budget_Clean'] = df[BUDGET_COL].apply(clean_currency)
    print(f"- Cleaned '{BUDGET_COL}' into 'Budget_Clean'. Original non-NA: {df[BUDGET_COL].notna().sum()}, Cleaned non-NA: {df['Budget_Clean'].notna().sum()}")
else:
    print(f"Warning: Budget column '{BUDGET_COL}' not found or configured as None.")
    BUDGET_COL = None # Mark as unavailable if not found
    missing_financial_data = True

if BOX_OFFICE_COL and BOX_OFFICE_COL in df.columns:
    df['BoxOffice_Clean'] = df[BOX_OFFICE_COL].apply(clean_currency)
    print(f"- Cleaned '{BOX_OFFICE_COL}' into 'BoxOffice_Clean'. Original non-NA: {df[BOX_OFFICE_COL].notna().sum()}, Cleaned non-NA: {df['BoxOffice_Clean'].notna().sum()}")
else:
    print(f"Warning: Box office column '{BOX_OFFICE_COL}' not found or configured as None.")
    BOX_OFFICE_COL = None # Mark as unavailable
    missing_financial_data = True

print("\nHandling missing cleaned financial data...")
initial_rows = len(df)
required_cleaned_cols = []
if BUDGET_COL: required_cleaned_cols.append('Budget_Clean')
if BOX_OFFICE_COL: required_cleaned_cols.append('BoxOffice_Clean')

if len(required_cleaned_cols) > 0:
    df_financial = df.dropna(subset=required_cleaned_cols).copy()
    rows_dropped = initial_rows - len(df_financial)
    print(f"Dropped {rows_dropped} rows missing essential cleaned financial data ({', '.join(required_cleaned_cols)}).")
    print(f"Shape for financial analysis: {df_financial.shape}")

    # Convert other columns if they exist
    print("\nEnsuring other columns are numeric...")
    if YEAR_COL in df_financial.columns:
        df_financial[YEAR_COL] = pd.to_numeric(df_financial[YEAR_COL], errors='coerce').fillna(0).astype(int)
    if RATING_COL in df_financial.columns:
        df_financial[RATING_COL] = pd.to_numeric(df_financial[RATING_COL], errors='coerce')

    # Remove duplicates
    if NAME_COL in df_financial.columns and YEAR_COL in df_financial.columns:
        initial_rows_fin = len(df_financial)
        df_financial = df_financial.drop_duplicates(subset=[NAME_COL, YEAR_COL], keep='first')
        rows_dropped_dup = initial_rows_fin - len(df_financial)
        if rows_dropped_dup > 0: print(f"Dropped {rows_dropped_dup} duplicate entries.")

else:
    print("Cannot proceed with financial analysis due to missing budget/box office columns or data.")
    df_financial = pd.DataFrame() # Ensure it's an empty DataFrame


# --- 3. Analysis & Visualization ---
# Only proceed if BOTH budget and box office columns were found, cleaned, and resulted in some data
if not df_financial.empty and BUDGET_COL and BOX_OFFICE_COL:
    print("\n--- Analyzing Budget vs. Box Office ---")
    sns.set_style("darkgrid") # Try darkgrid for potentially better contrast

    # A. Distributions of Budget and Box Office (Log Scale)
    plt.figure(figsize=(14, 6)) # Slightly wider figure

    plt.subplot(1, 2, 1)
    sns.histplot(df_financial['Budget_Clean'], log_scale=True, kde=False, bins=25, color='skyblue')
    plt.title('Distribution of Movie Budgets\n(IMDb Top 250)', fontsize=14)
    # Log scale axes are best labeled by powers of 10 (matplotlib does this automatically)
    plt.xlabel('Budget (USD - Log Scale)', fontsize=12)
    plt.ylabel('Number of Movies', fontsize=12)
    plt.grid(True, which="both", ls="--", linewidth=0.5) # Ensure gridlines

    plt.subplot(1, 2, 2)
    sns.histplot(df_financial['BoxOffice_Clean'], log_scale=True, kde=False, bins=25, color='lightcoral')
    plt.title('Distribution of Box Office Gross Revenue\n(IMDb Top 250)', fontsize=14)
    plt.xlabel('Box Office Gross (USD - Log Scale)', fontsize=12)
    plt.ylabel('Number of Movies', fontsize=12)
    plt.grid(True, which="both", ls="--", linewidth=0.5)

    plt.tight_layout(pad=3.0) # Add padding between subplots
    plt.show()

    # B. Scatter Plot: Budget vs. Box Office (Log-Log Scale)
    plt.figure(figsize=(10, 8)) # Make scatter plot larger
    scatter = sns.scatterplot(
        x='Budget_Clean',
        y='BoxOffice_Clean',
        data=df_financial,
        alpha=0.7, # Slightly less transparency
        s=60, # Slightly larger points
        hue=RATING_COL if RATING_COL in df_financial.columns else None, # Color by rating if available
        palette='viridis', # Choose a color palette
        legend='brief' if RATING_COL in df_financial.columns else False
    )
    plt.title('Budget vs. Box Office Gross for IMDb Top 250 Movies\n(Logarithmic Scale)', fontsize=16)
    plt.xlabel('Production Budget (USD - Log Scale)', fontsize=12)
    plt.ylabel('Worldwide Box Office Gross (USD - Log Scale)', fontsize=12)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which="both", ls=":", linewidth=0.6, color='gray') # Lighter grid

    # Add a y=x reference line (more carefully plotted for log-log)
    # Find common range based on plot limits after scaling
    xlims = plt.xlim()
    ylims = plt.ylim()
    common_min = max(xlims[0], ylims[0]) # Use the larger minimum limit
    common_max = min(xlims[1], ylims[1]) # Use the smaller maximum limit
    # Plot line only within the common visible range to avoid extending too far
    if common_min < common_max:
         plt.plot([common_min, common_max], [common_min, common_max], color='red', linestyle='--', linewidth=1.5, label='Break Even (BoxOffice = Budget)')
         plt.legend(fontsize=10)

    plt.show()

    # C. Correlation Analysis
    # (Calculation remains the same)
    correlation = df_financial['Budget_Clean'].corr(df_financial['BoxOffice_Clean'])
    print(f"\nCorrelation between Budget and Box Office: {correlation:.2f}")

    if RATING_COL in df_financial.columns:
        corr_matrix_df = df_financial[['Budget_Clean', 'BoxOffice_Clean', RATING_COL]].corr()
        print("\nCorrelation Matrix:")
        print(corr_matrix_df)

        plt.figure(figsize=(7, 5)) # Adjusted size
        sns.heatmap(corr_matrix_df, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, annot_kws={"size": 12})
        plt.title('Correlation Matrix\n(Budget, Box Office, Rating)', fontsize=14)
        plt.xticks(rotation=45, ha='right') # Rotate labels slightly
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

    # D. Return on Investment (ROI) Proxy Calculation
    roi_df = df_financial[df_financial['Budget_Clean'] > 0].copy()
    if not roi_df.empty:
        roi_df['ROI_Percentage'] = ((roi_df['BoxOffice_Clean'] - roi_df['Budget_Clean']) / roi_df['Budget_Clean']) * 100

        print(f"\nCalculated ROI for {len(roi_df)} movies with non-zero budget.")
        print("\nROI Summary Statistics (%):")
        print(roi_df['ROI_Percentage'].describe())

        # Visualize ROI Distribution (Histogram)
        plt.figure(figsize=(10, 6))
        # Cap extreme outliers for better visualization (e.g., show ROI between -100% and 1000%)
        roi_to_plot = roi_df['ROI_Percentage'].clip(-100, 1000)
        sns.histplot(roi_to_plot, bins=40, kde=False, color='green')
        plt.title('Distribution of Estimated ROI (%)\n(IMDb Top 250, Capped at -100% to 1000%)', fontsize=14)
        plt.xlabel('Return on Investment (%)', fontsize=12)
        plt.ylabel('Number of Movies', fontsize=12)
        plt.grid(axis='y', linestyle='--', linewidth=0.5)
        plt.show()

        # Top 5 Movies by ROI
        print("\nTop 5 Movies by Estimated ROI (%):")
        # Display relevant columns clearly formatted
        print(roi_df.nlargest(5, 'ROI_Percentage')[[NAME_COL, 'ROI_Percentage', 'Budget_Clean', 'BoxOffice_Clean']].round(1))
        # Bottom 5 Movies by ROI
        print("\nBottom 5 Movies by Estimated ROI (%):")
        print(roi_df.nsmallest(5, 'ROI_Percentage')[[NAME_COL, 'ROI_Percentage', 'Budget_Clean', 'BoxOffice_Clean']].round(1))
    else:
        print("\nCould not calculate ROI (no movies with non-zero budget).")

    # E. (Optional) Scatter Plot: Rating vs. Box Office (Log Scale Y)
    if RATING_COL in df_financial.columns:
        plt.figure(figsize=(10, 7))
        sns.scatterplot(x=RATING_COL, y='BoxOffice_Clean', data=df_financial, alpha=0.7, s=50)
        plt.title(f'IMDb Rating vs. Box Office Gross\n(IMDb Top 250)', fontsize=16)
        plt.xlabel(f'IMDb Rating ({df_financial[RATING_COL].min():.1f}-{df_financial[RATING_COL].max():.1f})', fontsize=12)
        plt.ylabel('Worldwide Box Office Gross (USD - Log Scale)', fontsize=12)
        plt.yscale('log')
        plt.grid(True, which="both", ls=":", linewidth=0.6, color='gray')
        plt.show()

elif not df_financial.empty:
    print("\n--- Limited Financial Analysis ---")
    print("Only one financial column was available or cleaned successfully.")
    single_col_clean = 'Budget_Clean' if BUDGET_COL else 'BoxOffice_Clean'
    single_col_orig = BUDGET_COL if BUDGET_COL else BOX_OFFICE_COL
    plt.figure(figsize=(8, 5))
    sns.histplot(df_financial[single_col_clean], log_scale=True, kde=False, bins=20)
    plt.title(f'Distribution of {single_col_orig} (Log Scale)', fontsize=14)
    plt.xlabel(f'{single_col_orig} (USD - Log Scale)', fontsize=12)
    plt.ylabel('Number of Movies', fontsize=12)
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.show()

else:
    print("\n--- Financial Analysis Skipped ---")
    print("No usable financial data could be processed.")

# --- 4. Interpretation ---
# (Interpretation guide remains the same, guiding the user to look at the newly formatted plots)
print("\n--- Interpretation Guide ---")
if not df_financial.empty and BUDGET_COL and BOX_OFFICE_COL:
    print("1. Distributions:")
    print("   - Review the histograms for Budget and Box Office (Log Scale). Note the range and where most movies fall.")
    print("2. Budget vs. Box Office Scatter Plot (Log-Log):")
    print("   - Observe the overall trend. Does higher budget generally associate with higher gross?")
    print("   - How spread out are the points? Does the hue (rating) show any pattern?")
    print("   - Note movies above/below the red 'Break Even' line.")
    print("3. Correlation Value:")
    print(f"   - Consider the calculated correlation ({correlation:.2f}) - how strong is the *linear* link on the log-log scale?")
    print("   - Check the heatmap: How does Rating correlate with the financial figures?")
    print("4. ROI Distribution:")
    print("   - Look at the ROI Histogram (%). Where is the peak? How many films lost money (ROI < 0)? How many had very high returns?")
    print("   - Refer to the summary stats and Top/Bottom 5 lists for specific examples.")
    print("5. Rating vs. Box Office Scatter Plot (Optional):")
    print("   - Among these top-rated films, does a slightly higher rating strongly predict box office, or are other factors more dominant? Look at the vertical spread.")
else:
     print("Financial analysis was limited or skipped.")

print("\n--- Conclusion ---")
# (Conclusion section remains the same, summarizing findings based on the improved plots)
if not df_financial.empty and BUDGET_COL and BOX_OFFICE_COL:
    print("This analysis explored the financial aspects of IMDb's Top 250 movies.")
    print(f"A positive correlation ({correlation:.2f}) was found between budget and box office gross (log scale), suggesting some association.")
    print("ROI analysis revealed [Summarize e.g., 'a wide distribution, with most films profitable but some significant outliers...'].")
    print("(Optional) The rating vs. box office plot showed [Summarize e.g., 'a weak positive relationship...'].")
else:
    print("Financial analysis was not fully completed due to missing/unusable data.")

print("\nImportant Considerations:")
print("- Data Quality: Accuracy depends on source & cleaning.")
print("- Inflation: Figures are not inflation-adjusted.")
print("- Correlation vs. Causation: Correlation doesn't prove cause.")
print("- Top 250 Bias: Only represents elite films.")
print("\nAnalysis Complete.")