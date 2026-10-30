
from pathlib import Path
import subprocess
import sys
import json
from datetime import datetime, timezone

from airflow.sdk import DAG
from airflow.providers.standard.operators.python import PythonOperator

import airflow_config as c


PROJECT_ROOT = Path(c.PROJECT_ROOT)
SRC_DIR = PROJECT_ROOT / "src"


def ensure_directories():
    for folder in [
        Path(c.DATA_DIR),
        Path(c.MODELS_DIR),
        Path(c.WEIGHTS_DIR),
        Path(c.ARTIFACTS_DIR),
        Path(c.RUNS_DIR),
    ]:
        folder.mkdir(parents=True, exist_ok=True)

    print("Project directories are ready.")


def validate_dataset():
    raw_dir = Path(c.DATA_DIR) / "raw"

    if not raw_dir.exists():
        raise FileNotFoundError(f"Missing dataset: {raw_dir}")

    if not any(raw_dir.iterdir()):
        raise ValueError(f"Dataset directory is empty: {raw_dir}")

    print(f"Dataset found: {raw_dir}")


def verify_splits():
    required_dirs = [
        Path(c.DATA_DIR) / "processed",
    ]

    for directory in required_dirs:
        if not directory.exists():
            raise FileNotFoundError(
                f"Missing processed dataset directory: {directory}"
            )

    print("Dataset split directory exists.")


def prepare_classification():
    run_script("prepare_classification.py")


def prepare_detection():
    run_script("prepare_detection.py")


def train_classifier():
    run_script("train_classification.py")


def train_detector():
    run_script("train_detection.py")


def evaluate_classifier():
    run_script("classification_metrics.py")


def evaluate_detector():
    run_script("detection_metrics.py")


def run_script(script_name):
    script_path = SRC_DIR / script_name

    if not script_path.exists():
        raise FileNotFoundError(
            f"Script not found: {script_path}"
        )

    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PROJECT_ROOT),
        check=True,
    )


def publish_summary():
    summary = {
        "pipeline": c.DAG_ID,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "classification_models": c.CLASSIFICATION_MODELS,
        "regularization_configs": c.REGULARIZATION_CONFIGS,
        "detection_model": c.YOLO_MODEL,
    }

    summary_path = Path(c.ARTIFACTS_DIR) / "run_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print(f"Run summary saved: {summary_path}")


with DAG(
    dag_id=c.DAG_ID,
    start_date=c.START_DATE,
    schedule=c.SCHEDULE,
    catchup=c.CATCHUP,
    max_active_runs=c.MAX_ACTIVE_RUNS,
    default_args=c.DEFAULT_ARGS,
    tags=c.TAGS,
) as dag:

    setup_directories = PythonOperator(
        task_id="setup_directories",
        python_callable=ensure_directories,
    )

    validate_dataset_task = PythonOperator(
        task_id="validate_dataset",
        python_callable=validate_dataset,
    )

    verify_splits_task = PythonOperator(
        task_id="verify_splits",
        python_callable=verify_splits,
    )

    prepare_classification_task = PythonOperator(
        task_id="prepare_classification",
        python_callable=prepare_classification,
    )

    train_classifier_task = PythonOperator(
        task_id="train_classifier",
        python_callable=train_classifier,
    )

    evaluate_classifier_task = PythonOperator(
        task_id="evaluate_classifier",
        python_callable=evaluate_classifier,
    )

    prepare_detection_task = PythonOperator(
        task_id="prepare_detection",
        python_callable=prepare_detection,
    )

    train_detector_task = PythonOperator(
        task_id="train_detector",
        python_callable=train_detector,
    )

    evaluate_detector_task = PythonOperator(
        task_id="evaluate_detector",
        python_callable=evaluate_detector,
    )

    publish_summary_task = PythonOperator(
        task_id="publish_summary",
        python_callable=publish_summary,
    )

    setup_directories >> validate_dataset_task >> verify_splits_task

    verify_splits_task >> [
        prepare_classification_task,
        prepare_detection_task,
    ]

    prepare_classification_task >> train_classifier_task
    train_classifier_task >> evaluate_classifier_task

    prepare_detection_task >> train_detector_task
    train_detector_task >> evaluate_detector_task

    [
        evaluate_classifier_task,
        evaluate_detector_task,
    ] >> publish_summary_task
