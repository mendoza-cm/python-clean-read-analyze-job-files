import data_management as dm
import ml_text_search as ml

def main():
    job_files = dm.get_files_by_keywords(
        folder = "../Data/",
        keywords=["job"])

    combined_df, summary_df, type_groups = dm.summarize_columns_from_sources(
        job_files,
        folder = "../Data/")


    print("type_groups keys:", type_groups.keys())
    print(combined_df['additional_benefits'].head())



    categorical_cols = type_groups.get('categorical', [])

    # Safe numeric columns — exclude IDs and indexes
    exclude_from_numeric = {'_jobspy_index', 'id', 'job_id', 'listing_id', 'index', 'maximum_pay'}

    numeric_cols = [
        col for col in type_groups.get('numeric', [])
        if col not in exclude_from_numeric and combined_df[col].nunique() > 1
    ]

    print(f"Clean numeric columns for regression: {numeric_cols}")

    dm.plot_and_save_correlations(
        combined_df,
        numeric_cols,
        output_path="./Results/plot_and_save_test.png"
    )

    job_text = dm.build_job_text(
        combined_df,
        ["description", "programming_languages", "additional_benefits"]
    )

    print(job_text.shape)

    clean_df = dm.anonymize_pii(combined_df)






if __name__ == '__main__':
    main()


