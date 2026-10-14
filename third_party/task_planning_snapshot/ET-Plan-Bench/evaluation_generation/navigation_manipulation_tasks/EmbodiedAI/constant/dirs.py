import os


PATH_TO_REPO = os.path.dirname(os.path.dirname(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(PATH_TO_REPO)))
REPO_NAME = PATH_TO_REPO.split("/")[-1]

LLAVA_DIR = os.path.join(BASE_DIR, "llama_sft")

UNITY_EXEC_FILE = os.path.join(
    BASE_DIR,
    "simulator",
    "virtualhome",
    "virtualhome",
    "simulation",
    "unity_simulator_2.3.0",
    "linux_exec.v2.3.0.x86_64",
)

LLAVA_MODEL_BASE_DIR = os.path.join(LLAVA_DIR, "liuhaotian", "llava-v1.5-7b")
LLAVA_MODEL_PATH = os.path.join(
    LLAVA_DIR,
    "checkpoints",
    "llava-v1.5-7b-task-lora-simple_navigation_and_manipulation_QA",
)
LLAVA_IMAGE_PATH = os.path.join(LLAVA_DIR, "empty_img.jpg")

LOG_DIR = os.path.join(PATH_TO_REPO, "Log")
LOG_UNITY_DIR = os.path.join(LOG_DIR, "unity")

OUTPUT_DIR = os.path.join(BASE_DIR, "Output", "nav_mani")
OUTPUT_DATA_DIR = os.path.join(OUTPUT_DIR, "data")
OUTPUT_TASK_DIR = os.path.join(OUTPUT_DIR, "task")

TASK_INFO_DIR = os.path.join(PATH_TO_REPO, "task_gen", "info")

GENERATED_TASK_DIR = os.path.join(OUTPUT_TASK_DIR, "generated_tasks")
GENERATED_TASK_SOURCE_DIR = os.path.join(BASE_DIR, "generated_tasks")
GENERATED_TASK_FILTERED_DIR = os.path.join(GENERATED_TASK_DIR, "filtered")
GENERATED_TASK_ABSTRACT_DIR = os.path.join(GENERATED_TASK_DIR, "abstract")

GENERATED_DATA_DIR = os.path.join(OUTPUT_DATA_DIR, "generated_data")
GENERATED_DATA_GROUND_TRUTH_DIR = os.path.join(GENERATED_DATA_DIR, "ground_truth")
GENERATED_DATA_VISIBLE_DIR = os.path.join(GENERATED_DATA_DIR, "visible")
GENERATED_DATA_VISIBLE_LLAVA_DIR = os.path.join(GENERATED_DATA_DIR, "visible_llava")
GENERATED_DATA_DEBUG_DIR = os.path.join(GENERATED_DATA_DIR, "debug")
