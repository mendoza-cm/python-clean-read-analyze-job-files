import data_management as dm
import ml_text_search as ml



def main():
    job_files = dm.get_files_by_keywords(
        folder="../Data/",
        keywords=["job"])

    combined_df, summary_df, type_groups = dm.summarize_columns_from_sources(
        job_files,
        folder="../Data/")

    # Keep a pristine copy (optional but safe)
    # original_df = combined_df.copy()

    # print(f"Original shape: {original_df.shape}")

    clean_df = combined_df.copy()

    # Anonymize a separate clean version
    # clean_df = dm.anonymize_pii(combined_df)  # or pass original_df.copy() if you prefer
    # print(f"Shape after anonymization: {clean_df.shape}")

    # Use clean_df for text search (PII-safe)
    job_text = dm.build_job_text(
        clean_df,
        ["description", "programming_languages", "additional_benefits"]
    )

    top_scores = ml.calc_tfidf_cosine_similarity(clean_df,
                                                 job_text,
                                                 "statistical analysis",
                                                 10,
                                                 ["title"])
    print(top_scores)
    print(job_text.shape)
    print(clean_df.columns)

    # For numeric/correlation analysis, decide which version you want:
    # Option 1: Use original_df if you want all columns (e.g., if a numeric col was mistakenly flagged as PII)
    # Option 2: Use clean_df to stay consistent and safe

    # Example using original_df (since PII columns like 'emails' aren't numeric anyway)

    exclude_from_numeric = {'_jobspy_index', 'id', 'job_id', 'listing_id', 'index', 'maximum_pay'}
    # numeric_cols = [
    #    col for col in type_groups.get('numeric', [])
    #    if col not in exclude_from_numeric and original_df[col].nunique() > 1
    # ]

    numeric_cols = [
        col for col in type_groups.get('numeric', [])
        if col not in exclude_from_numeric and clean_df[col].nunique() > 1
    ]
    print(f"Clean numeric columns for regression: {numeric_cols}")

    # If you want correlations on clean data:
    # dm.plot_and_save_correlations(clean_df, numeric_cols, ...)

    # Or on original:
    # dm.plot_and_save_correlations(original_df, numeric_cols, ...)

    # If you need the clean_df for other analysis (correlations, etc.), use it from now on
    # e.g., dm.plot_and_save_correlations(clean_df, numeric_cols, ...)


    categorical_cols = type_groups.get('categorical', [])

    # Safe numeric columns — exclude IDs and indexes


    print(f"Clean numeric columns for regression: {numeric_cols}")

    # dm.plot_and_save_correlations(
    #     combined_df,
    #     numeric_cols,
    #     output_path="./Results/plot_and_save_test.png"
    # )








if __name__ == '__main__':
    main()


