RM = rm -f

VENV_DIR := .venv

MAKEFLAGS += --no-print-directory

# --- colours ---
RESET := \033[0m
BOLD  := \033[1m
RED   := \033[31m
GRN   := \033[32m
YEL   := \033[33m
BLU   := \033[34m

all: run

help:
	@printf "$(YEL)make install$(RESET)			# install dependencies via uv\n"
	@printf "$(YEL)make / make all / make run$(RESET)	# run program\n\n"
	@printf "$(YEL)make debug$(RESET)		# run program in pdb\n"
	@printf "$(YEL)make debug-print$(RESET)	# run program and show debug prints\n\n"
# 	@printf "$(YEL)make test$(RESET)		# run program with custom input and output files\n\n"
	@printf "$(YEL)make lint$(RESET)		# run flake8 and mypy\n"
	@printf "$(YEL)make lint-strict$(RESET)	# run flake8 and mypy in a strict mode\n\n"
	@printf "$(YEL)make clean$(RESET)		# remove caches and temps\n"
	@printf "$(YEL)make fclean$(RESET)		# remove caches and temps + venv\n"
	@printf "$(YEL)make re$(RESET)			# clean + run\n\n"

install:
	@printf "Installing dependencies via uv...\n"
	@uv sync
	@printf "$(GRN)Dependencies installed successfully.$(RESET)\n"

FUNC_DEFS_FILE = data/input/functions_definition.json
FUNC_CALLS_FILE = data/input/function_calling_tests.json
OUTPUT_FILE = data/output/function_calls.json

DEBUG_PRINT :=

run:
	-@uv run python -m src \
		--functions_definition $(FUNC_DEFS_FILE) \
		--input $(FUNC_CALLS_FILE) \
		--output $(OUTPUT_FILE) \
		$(DEBUG_PRINT)

debug-print:
	-@$(MAKE) run DEBUG_PRINT="--debug"

define banner
	printf "\n$(YEL)%s$(RESET)\n" "================================================================================"; \
	printf "$(BOLD)$(YEL)==> $(1)$(RESET)"; \
	printf "\n$(YEL)%s$(RESET)\n" "================================================================================";
endef

# test:
# 	@$(call banner,CUSTOM INPUT - BASIC)
# 	-@$(MAKE) run \
# 		FUNC_DEFS_FILE=data/input/functions_definition_custom.json \
# 		FUNC_CALLS_FILE=data/input/function_calling_tests_custom.json \
# 		OUTPUT_FILE=data/output/function_calls_custom.json
# 	@$(call banner,CUSTOM INPUT - SPEECHLIKE PROMPTS)
# 	-@$(MAKE) run \
# 		FUNC_DEFS_FILE=data/input/functions_definition_custom.json \
# 		FUNC_CALLS_FILE=data/input/function_calling_tests_speech.json \
# 		OUTPUT_FILE=data/output/function_calls_speech.json
# 	@$(call banner,CUSTOM INPUT - NONSENSE PROMPTS)
# 	-@$(MAKE) run \
# 		FUNC_DEFS_FILE=data/input/functions_definition_custom.json \
# 		FUNC_CALLS_FILE=data/input/function_calling_tests_no_call.json \
# 		OUTPUT_FILE=data/output/function_calls_no_call.json

debug:
	@uv run python -m pdb -m src

clean:
	@find . -type d \( -name "__pycache__" -o -name ".mypy_cache" \) -exec rm -r {} +
	$(RM) $(OUTPUT_FILE) $(LINT_LOGFILE)
	@printf "$(YEL)Caches successfully deleted.$(RESET)\n"

LINT_LOGFILE="lint.log"
SRC = . # src

.lint-base: install
	@overall=0; \
	echo "" > $(LINT_LOGFILE); \
	printf "Running flake8... "; \
	printf "FLAKE8:\n" > $(LINT_LOGFILE); \
	uv run flake8 $(SRC) >> $(LINT_LOGFILE) 2>&1 && \
		printf "$(GRN)[OK]$(RESET)\n" || { printf "$(RED)[KO]$(RESET)\n"; overall=1; }; \
	printf "Running mypy... "; \
	printf "\nMYPY:\n" >> $(LINT_LOGFILE); \
	uv run mypy $(SRC) $(MYPY_FLAGS) >> $(LINT_LOGFILE) 2>&1 && \
		printf "$(GRN)[OK]$(RESET)\n" || { printf "$(RED)[KO]$(RESET)\n"; overall=1; }; \
	if [ $$overall -eq 0 ]; then \
		$(RM) $(LINT_LOGFILE); \
		printf "✅ $(GRN)Lint passed!$(RESET)\n"; \
	else \
		printf "❌ $(RED)Lint failed!$(RESET)\n"; \
		printf " - check $(BLU)$(LINT_LOGFILE)$(RESET) for details\n"; \
	fi

lint: MYPY_FLAGS := --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
lint: .lint-base

lint-strict: MYPY_FLAGS := --strict
lint-strict: .lint-base

re: clean all

fclean: clean
	@$(RM) -r $(VENV_DIR)
	@printf "$(YEL)Virtual env $(VENV_DIR) successfully deleted.$(RESET)\n"

.PHONY: all clean install run debug lint lint-strict re fclean help debug-print test
