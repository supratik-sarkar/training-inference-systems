
import argparse
import mlflow
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="default")
    parser.add_argument("--run_name", default="dev")
    args = parser.parse_args()

    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_param("example_param", 42)
        # Simulated metric logging for template
        for step in range(5):
            mlflow.log_metric("loss", 1.0/(step+1), step=step)
            time.sleep(0.1)

if __name__ == "__main__":
    main()
