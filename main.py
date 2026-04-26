from pipeline import run_pipeline


# =========================
# 🔧 CONFIGURATION
# =========================

# Choose experiment mode
# Options:
# "baseline"
# "temporal"
# "hybrid_svm"
# "hybrid_elm"
# "skeleton"
# "multimodal"
# "hardware_test"

EXPERIMENT = "baseline"

# Choose dataset
# Options: "ucf101", "hmdb51"
DATASET = "ucf101"


# =========================
# 🚀 RUN
# =========================

if __name__ == "__main__":
    run_pipeline(mode=EXPERIMENT, dataset=DATASET)