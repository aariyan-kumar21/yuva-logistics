"""
main_pipeline.py
Runs all five steps of the Week 2 logistics data preprocessing pipeline in order.
"""

from importlib import import_module

step1 = import_module("01_simulate_collection")
step2 = import_module("02_data_profiling")
step3 = import_module("03_data_cleaning_imputation")
step4 = import_module("04_outlier_treatment")
step5 = import_module("05_feature_scaling")


def main():
    print("Step 1: Generating raw logistics data...")
    step1.create_raw_dataset()

    print("Step 2: Running initial data quality audit...")
    step2.run_data_audit()

    print("Step 3: Cleaning text, deduplicating, and imputing missing values...")
    step3.clean_logistics_data()

    print("Step 4: Flagging and capping outliers...")
    step4.handle_outliers()

    print("Step 5: Scaling numeric features for ML...")
    step5.scale_features()

    print("\nAll pipeline steps finished! Check the 'data/' directory for the CSV outputs.")


if __name__ == "__main__":
    main()
