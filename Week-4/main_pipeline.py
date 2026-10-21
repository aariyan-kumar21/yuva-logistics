"""
main_pipeline.py
Executes the complete Week 4 Predictive Modeling and Optimization workflow:
1. Generates the 2,000-shipment modeling dataset
2. Trains, cross-validates, and tunes predictive regression models
3. Runs the prescriptive dock & carrier optimization engine on the test set
4. Generates all 4 diagnostic and optimization charts in charts/
"""

from importlib import import_module

s1 = import_module("01_simulate_modeling_data")
s2 = import_module("02_train_evaluate_models")
s3 = import_module("03_logistics_optimization")
s4 = import_module("04_generate_model_charts")


def main():
    print("Step 1: Simulating Week 4 logistics modeling dataset...")
    s1.generate_modeling_dataset()

    print("\nStep 2: Training, cross-validating, and tuning predictive models...")
    s2.train_and_evaluate_models()

    print("\nStep 3: Running model-guided prescriptive optimization...")
    s3.run_operations_optimization()

    print("\nStep 4: Generating evaluation and optimization charts...")
    s4.generate_all_charts()

    print("\nWeek 4 pipeline finished! All CSV outputs and PNG charts are saved.")


if __name__ == "__main__":
    main()
