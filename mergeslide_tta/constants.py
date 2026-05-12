"""
Shared constants used across training, merging, and evaluation.
All task-specific configurations are defined here once.
"""

# Total number of sequential tasks (TCGA cohorts)
NUM_TASKS: int = 6

# Number of classes per task
NUM_CLASSES: list[int] = [2, 3, 2, 2, 2, 2]

# Total number of classes across all tasks
TOTAL_CLASSES: int = sum(NUM_CLASSES)  # = 13

# Embedding dimension from TITAN vision encoder
EMBED_DIM: int = 768

# Patch size used during WSI preprocessing
WSI_PATCH_PIXELS: int = 256

TITAN_PS_ARG: int = 1024

# Number of patches randomly sampled per slide during training/inference
K_PATCHES: int = 400

# Maps task_id → [start_class_idx, end_class_idx] in the global 13-class classifier
# Used to slice prompt prototype embeddings
TASK_CLASS_RANGES: dict[int, list[int]] = {
    0: [0, 1],    # BRCA: IDC(0), ILC(1)
    1: [2, 4],    # RCC:  CCRCC(2), PRCC(3), CHRCC(4)
    2: [5, 6],    # NSCLC: LUAD(5), LUSC(6)
    3: [7, 8],    # ESCA: class0(7), class1(8)
    4: [9, 10],   # TGCT: class0(9), class1(10)
    5: [11, 12],  # CESC: class0(11), class1(12)
}

# Maps task_id → maps local class idx → global class idx
# Used in Class-IL evaluation to convert per-task predictions to global space
TASK_TO_GLOBAL_CLASS: dict[int, dict[int, int]] = {
    0: {0: 0,  1: 1},
    1: {0: 2,  1: 3,  2: 4},
    2: {0: 5,  1: 6},
    3: {0: 7,  1: 8},
    4: {0: 9,  1: 10},
    5: {0: 11, 1: 12},
}

# Task names for logging
TASK_NAMES: list[str] = ["BRCA", "RCC", "NSCLC", "ESCA", "TGCT", "CESC"]