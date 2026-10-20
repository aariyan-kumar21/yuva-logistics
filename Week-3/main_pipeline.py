"""
main_pipeline.py
Runs the complete Week 3 Logistics EDA and Visualization workflow:
1. Generates the 1,800-row 2025 operational dataset
2. Computes descriptive statistics, skewness, and correlation matrices
3. Produces all 5 visualization PNGs inside charts/
"""

from importlib import import_module

step1 = import_module("01_simulate_logistics_data")
step2 = import_module("02_exploratory_data_analysis")
step3 = import_module("03_generate_visualizations")


def main():
    print("Step 1: Generating Week 3 logistics dataset...")
    step1.generate_eda_dataset()

    print("\nStep 2: Running Exploratory Data Analysis (EDA)...")
    step2.perform_eda()

    print("\nStep 3: Generating visualizations in charts/...")
    step3.build_all_visualizations()

    print("\nWeek 3 pipeline complete!")


if __name__ == "__main__":
    main()
