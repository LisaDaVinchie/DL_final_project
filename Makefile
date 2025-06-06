BASE_DIR := $(shell pwd)
PYTHON := $(shell which python3)

DATA_DIR := $(BASE_DIR)/data
SRC_DIR := $(BASE_DIR)/src
FIG_DIR := $(BASE_DIR)/figs
TEST_DIR := $(BASE_DIR)/tests


DATASETS_DIR := $(DATA_DIR)/datasets
DATASET_BASENAME := dataset
DATASET_FILE_EXT := .pt

RESULTS_DIR := $(DATA_DIR)/results
RESULT_BASENAME := result
RESULT_FILE_EXT := .txt

WEIGHTS_DIR := $(DATA_DIR)/weights
WEIGHTS_BASENAME := weights
WEIGHTS_FILE_EXT := .pt

MASKS_DIR := $(DATA_DIR)/masks
MASKS_BASENAME := mask
MASKS_FILE_EXT := .pt

CURRENT_IDX_DATASET := $(shell find "$(DATASETS_DIR)" -type f -name "$(DATASET_BASENAME)_*$(DATASET_FILE_EXT)" | \
    sed 's|.*_\([0-9]*\)\$(DATASET_FILE_EXT)|\1|' | \
    sort -n | tail -1)
NEXT_IDX_DATASET = $(shell echo $$(($(CURRENT_IDX_DATASET) + 1)))

CURRENT_IDX_RESULT := $(shell find "$(RESULTS_DIR)" -type f -name "$(RESULT_BASENAME)_*$(RESULT_FILE_EXT)" | \
	sed 's|.*_\([0-9]*\)\$(RESULT_FILE_EXT)|\1|' | \
	sort -n | tail -1)
NEXT_IDX_RESULT = $(shell echo $$(($(CURRENT_IDX_RESULT) + 1)))

CURRENT_IDX_MASK := $(shell find "$(MASKS_DIR)" -type f -name "$(MASKS_BASENAME)_*$(MASKS_FILE_EXT)" | \
	sed 's|.*_\([0-9]*\)\$(MASKS_FILE_EXT)|\1|' | \
	sort -n | tail -1)
NEXT_IDX_MASK = $(shell echo $$(($(CURRENT_IDX_MASK) + 1)))

CURRENT_DATASET_PATH := $(DATASETS_DIR)/$(DATASET_BASENAME)_$(CURRENT_IDX_DATASET)$(DATASET_FILE_EXT)
NEXT_DATASET_PATH := $(DATASETS_DIR)/$(DATASET_BASENAME)_$(NEXT_IDX_DATASET)$(DATASET_FILE_EXT)

CURRENT_RESULT_PATH := $(RESULTS_DIR)/$(RESULT_BASENAME)_$(CURRENT_IDX_RESULT)$(RESULT_FILE_EXT)
NEXT_RESULT_PATH := $(RESULTS_DIR)/$(RESULT_BASENAME)_$(NEXT_IDX_RESULT)$(RESULT_FILE_EXT)

CURRENT_WEIGHTS_PATH := $(WEIGHTS_DIR)/$(WEIGHTS_BASENAME)_$(CURRENT_IDX_RESULT)$(WEIGHTS_FILE_EXT)
NEXT_WEIGHTS_PATH := $(WEIGHTS_DIR)/$(WEIGHTS_BASENAME)_$(NEXT_IDX_RESULT)$(WEIGHTS_FILE_EXT)

CURRENT_MASK_PATH := $(MASKS_DIR)/$(MASKS_BASENAME)_$(CURRENT_IDX_MASK)$(MASKS_FILE_EXT)
NEXT_MASK_PATH := $(MASKS_DIR)/$(MASKS_BASENAME)_$(NEXT_IDX_MASK)$(MASKS_FILE_EXT)

PATHS_FILE := $(SRC_DIR)/paths.json
PARAMS_FILE := $(SRC_DIR)/params.json

.PHONY: config train test

config:
	@echo "Storing paths to json..."
	@echo "{" > $(PATHS_FILE)
	@echo "	\"current_dataset_path\": \"$(CURRENT_DATASET_PATH)\"," >> $(PATHS_FILE)
	@echo "	\"next_dataset_path\": \"$(NEXT_DATASET_PATH)\"," >> $(PATHS_FILE)
	@echo "	\"current_result_path\": \"$(CURRENT_RESULT_PATH)\"," >> $(PATHS_FILE)
	@echo "	\"next_result_path\": \"$(NEXT_RESULT_PATH)\"," >> $(PATHS_FILE)
	@echo "	\"current_weights_path\": \"$(CURRENT_WEIGHTS_PATH)\"," >> $(PATHS_FILE)
	@echo "	\"next_weights_path\": \"$(NEXT_WEIGHTS_PATH)\"," >> $(PATHS_FILE)
	@echo "	\"current_mask_path\": \"$(CURRENT_MASK_PATH)\"," >> $(PATHS_FILE)
	@echo "	\"next_mask_path\": \"$(NEXT_MASK_PATH)\"" >> $(PATHS_FILE)
	@echo "}" >> $(PATHS_FILE)

mask: config
	@echo "Generating masks..."
	$(PYTHON) $(SRC_DIR)/generate_masks.py --paths $(PATHS_FILE) --params $(PARAMS_FILE)

train: config
	$(PYTHON) $(SRC_DIR)/train.py --paths $(PATHS_FILE) --params $(PARAMS_FILE)

test:
	$(PYTHON) -m unittest discover -s $(TEST_DIR) -p "*.py"