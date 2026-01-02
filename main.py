import dataManagement as dm
def main():
    job_files = dm.get_files_by_keywords(folder = "../Data/",keywords=["job"])

    combined_df, summary_df, type_groups = dm.summarize_columns_from_sources(job_files, folder = "../Data/")


    print("type_groups keys:", type_groups.keys())
    print(combined_df['additional_benefits'].head())



    categorical_cols = type_groups.get('categorical', [])

    # Safe numeric columns — exclude IDs and indexes
    exclude_from_numeric = {'_jobspy_index', 'id', 'job_id', 'listing_id', 'index', 'maximum_pay'}  # add any others you spot

    numeric_cols = [
        col for col in type_groups.get('numeric', [])
        if col not in exclude_from_numeric and combined_df[col].nunique() > 1  # at least some variation
    ]

    print(f"Clean numeric columns for regression: {numeric_cols}")
    #dm.plot_and_save_correlations(combined_df, numeric_cols,output_path = "../Results/plot_and_save_test.png")

    my_combined_columns = dm.build_job_text(combined_df, ["description", "programming_languages", "additional_benefits"])

    print(my_combined_columns.shape)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
