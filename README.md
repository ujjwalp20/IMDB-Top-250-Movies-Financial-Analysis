# IMDb Top 250: Budget vs. Box Office Analysis

## Objective
Analyze the relationship between production budget and box office gross for IMDb Top 250 movies using Python (Pandas, Matplotlib, Seaborn). Includes data cleaning, correlation analysis, and ROI calculation.

## Dataset
*   Requires an IMDb Top 250 dataset (CSV format), typically found on Kaggle.
    *   **Example Source Link:** `https://www.kaggle.com/datasets/rajugc/imdb-top-250-movies-dataset`
*   Save the dataset as `imdb_top250_movies.csv` in the project directory.
*   **Key Columns Needed (Default Names):** `name`, `year`, `rating`, `budget`, `box_office`.
*   **!! IMPORTANT !!:** **Edit the `--- Configuration ---` section in `budget_boxoffice_analysis.py`** to match the column names in your specific CSV file. Review the `clean_currency` function for your data's format (e.g., '$', ',', 'M').

## Libraries Used
*   Python 3
*   Pandas, NumPy, Matplotlib, Seaborn

## How to Run
1.  Install libraries: `pip install pandas numpy matplotlib seaborn`
2.  Download the dataset CSV and save it as `imdb_top250_movies.csv`.
3.  **Configure column names** inside `budget_boxoffice_analysis.py`.
4.  Run the script: `python budget_boxoffice_analysis.py`

## Key Findings (Example)
*   A positive correlation exists between budget and box office gross (log-log scale) within the Top 250. (`<-- EDIT THIS: Optionally add your correlation value -->`)
*   Return on Investment (ROI) varies widely among these top films.
*   The direct link between small rating variations (within the Top 250) and box office might be weaker than the budget-box office link.

## Limitations
*   Relies on estimated financial data quality and cleaning assumptions.
*   Does not adjust for inflation.
*   Correlation does not imply causation.
*   Analysis limited to the elite Top 250 films.
