import os
import pandas as pd
import seaborn as sns
from typing import List, Optional, Union, Tuple, Dict
import matplotlib.pyplot as plt


__all__ = [
    "get_files_by_keywords",
    "summarize_columns_from_sources",
    "group_columns_by_type",
    "plot_and_save_correlations",
    "build_job_text"
]


def get_files_by_keywords(folder: str,
                      keywords: Optional[List[str]] = None,
                      extensions: Optional[List[str]] = None,
                      logic: str = "and"
                      ) -> List[str]:
    if not os.path.isdir(folder):
        print(f"Folder {folder} does not exist.")
        return []

    if extensions is None:
        extensions = ['.csv']

    extensions = [ext.lower() for ext in extensions]

    if keywords:
        keywords_lower = [kw.lower() for kw in keywords]
    else:
        keywords_lower = []

    matching_files = []
    for file in os.listdir(folder):
        file_lower = file.lower()

        has_valid_ext = any(file_lower.endswith(ext) for ext in extensions)
        if not has_valid_ext:
            continue

        if not keywords_lower:
            matching_files.append(file)
            continue

        if logic == "and":
            if all(kw in file_lower for kw in keywords_lower):
                matching_files.append(file)
        elif logic == "or":
            if any(kw in file_lower for kw in keywords_lower):
                matching_files.append(file)

    if matching_files:
        print(f"Found {len(matching_files)} matching file(s).")
    else:
        print(f"No files found matching criteria in '{folder}'.")

    return matching_files



#def table_data_table_columns(df: pd.DataFrame
       #                      ) -> pd.DataFrame:


def infer_column_type(series: pd.Series) -> str:
    """Infers human-friendly type from a pandas Series."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        unique_ratio = series.nunique() / len(series)
        if unique_ratio < 0.05 and series.nunique() < 20:
            return "categorical"
        return "numeric"
    if pd.api.types.is_object_dtype(series):
        # Try to detect numeric stored as text
        if pd.to_numeric(series, errors='coerce').notna().all():
            return "numeric (text)"
        # Try boolean-like text
        lower_vals = series.astype(str).str.lower()
        if set(lower_vals.unique()).issubset({'true', 'false', '1', '0', 'yes', 'no', 'nan', ''}):
            return "boolean (text)"
        # Low cardinality → categorical
        unique_ratio = series.nunique() / len(series.dropna())
        if unique_ratio < 0.05 and series.nunique() < 50:
            return "categorical"
        return "text"
    return "mixed/unknown"

def group_columns_by_type(summary_df: pd.DataFrame, max_per_line: int = 5) -> dict:
    """
    Takes the summary DataFrame from summarize_columns_from_sources
    and returns a dictionary grouping column names by inferred_type.

    Example output:
    {
        'numeric': ['salary', 'years_experience', 'id'],
        'categorical': ['role', 'location', 'department'],
        'text': ['description', 'company_name'],
        'boolean': ['remote_ok', 'full_time'],
        'datetime': ['posted_date']
    }
    """
    if summary_df.empty:
        print("No summary data to group.")
        return {}

    # Group by inferred_type and collect column names
    grouped = summary_df.groupby('inferred_type')['column_name'].agg(list).to_dict()


    for key in grouped:
        grouped[key].sort()

    print("Columns grouped by type:")
    for col_type, cols in grouped.items():
        print(f"  {col_type}:")
        # Print in chunks of max_per_line
        for i in range(0, len(cols), max_per_line):
            chunk = cols[i:i + max_per_line]
            print(f"    {', '.join(chunk)}")

    return grouped




def summarize_columns_from_sources(sources: List[Union[str, pd.DataFrame]],
                           folder: str
                                 ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[str]]]:
    dataframes = []

    for source in sources:
        if isinstance(source, str):
            # It's a CSV filename
            file_path = os.path.join(folder, source)
            try:
                df = pd.read_csv(file_path)
                df['_source_file'] = source  # Use actual filename
                print(f"Loaded {source}: {df.shape[0]} rows, {df.shape[1]} columns")
                dataframes.append(df)
            except Exception as e:
                print(f"Error loading {source}: {e}")

        elif isinstance(source, tuple) and len(source) == 2 and isinstance(source[1], pd.DataFrame):
            # It's (custom_label, DataFrame)
            label, df = source
            df = df.copy()  # Avoid modifying original
            df['_source_file'] = label  # Use your custom name
            print(f"Using pre-loaded DataFrame '{label}': {df.shape[0]} rows, {df.shape[1]} columns")
            dataframes.append(df)

        elif isinstance(source, pd.DataFrame):
            # Fallback: generic name if just DataFrame passed
            df = source.copy()
            df['_source_file'] = "preloaded_dataframe"
            print(f"Using unnamed pre-loaded DataFrame: {df.shape[0]} rows, {df.shape[1]} columns")
            dataframes.append(df)

        else:
            print(f"Unsupported source type: {type(source)}")

    if not dataframes:
        print("No valid data sources provided.")
        return pd.DataFrame(), pd.DataFrame(), {}

    combined_df = pd.concat(dataframes, ignore_index=True, sort=False)
    print(f"\nCombined total: {combined_df.shape[0]} rows, {combined_df.shape[1]} columns\n")

    summary_data = []

    for col in combined_df.columns:
        series = combined_df[col]
        missing_pct = series.isnull().mean() * 100
        summary_data.append({
            "column_name": col,
            "inferred_type": infer_column_type(series),
            "pandas_dtype": str(series.dtype),
            "unique_values": series.nunique(),
            "missing_pct": round(missing_pct, 2)
        })

    summary_df = pd.DataFrame(summary_data)

    # Sort for readability
    type_order = {"numeric": 0, "categorical": 1, "text": 2, "boolean": 3, "datetime": 4, "mixed/unknown": 5}
    summary_df["sort"] = summary_df["inferred_type"].map(type_order).fillna(6)
    summary_df = summary_df.sort_values("sort").drop("sort", axis=1).reset_index(drop=True)

    print("Column Summary Table:")
    print(summary_df.to_string(index=False))

    type_groups = group_columns_by_type(summary_df,max_per_line=5)  # uses your existing function

    return combined_df, summary_df, type_groups

def add_pay_range_features(df):
    df = df.copy()
    df["pay_range"] = df["max_amount"] - df["min_amount"]
    return df


def plot_and_save_correlations(df, numeric_vars, output_path):
    corr = df[numeric_vars].corr()

    plt.figure(figsize=(10, 10))
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()



def build_job_text(df, cols: List[str]):
    return(
        df.loc[:,cols]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.replace("-{2,}"," ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


import re



def anonymize_pii(df: pd.DataFrame) -> pd.DataFrame:

    df_clean = df.copy()

    # Pattern for likely PII column names (case-insensitive)
    pii_patterns = [
        'email', 'phone', 'linkedin', 'profile', 'recruiter', 'hiring_manager',
        'contact', 'poster', 'name.*id', 'user.*id', 'applicant', 'scraper'
    ]git add .

    # Compile regex for matching
    pii_regex = re.compile('|'.join(pii_patterns), re.IGNORECASE)

    # Find columns to drop
    cols_to_drop = [col for col in df_clean.columns if pii_regex.search(col)]

    if cols_to_drop:
        print(f"Dropping potential PII columns: {cols_to_drop}")
        df_clean = df_clean.drop(columns=cols_to_drop)
    else:
        print("No potential PII columns detected.")

    # Redact any stray email-like strings in text columns
    email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')

    text_cols = df_clean.select_dtypes(include=['object','string']).columns
    for col in text_cols:
        if df_clean[col].str.contains(email_pattern, regex=True, na=False).any():
            print(f"Redacting emails in column: {col}")
            df_clean[col] = df_clean[col].str.replace(email_pattern, '[EMAIL REDACTED]', regex=True)

    return df_clean