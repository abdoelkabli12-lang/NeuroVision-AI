from datetime import datetime
from datetime import timedelta

DAG_ID="neurovision_pipeline"

START_DATE=datetime(year=2026, month=10, day=30)

SCHEDULE=None

CATCHUP=False

MAX_ACTIVE_RUNS=1

DEFAULT_VIEW='graph'

TAGS=["MRI", "ML"]

DEFAULT_ARGS = {
    "owner": "neurovision",
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
    "depends_on_past": False
}

PROJECT_ROOT = "/opt/airflow/project"

DATA_DIR = f"{PROJECT_ROOT}/data"
MODELS_DIR = f"{PROJECT_ROOT}/models"
WEIGHTS_DIR = f"{PROJECT_ROOT}/weights"
ARTIFACTS_DIR = f"{PROJECT_ROOT}/artifacts"
RUNS_DIR = f"{PROJECT_ROOT}/runs"

SEED = 32

CLASSIFICATION_MODELS = [
    "resnet50",
    "resnet101",
    "mobilenetv2"
]

REGULARIZATION_CONFIGS = [
    "L1",
    "L2"
]

BATCH_SIZE = 32
IMAGE_SIZE = (150, 150)

CLASSIFICATION_EPOCHS = 30
DETECTION_EPOCHS = 30

YOLO_MODEL = "yolo11n.pt"