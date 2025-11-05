🛡️ **Concentrator: Unified Hashcat Rule Processor (v1.4.1)**

Concentrator is a powerful, parallelized Python script designed to analyze existing Hashcat rule sets and generate new, highly optimized rules. It supports three core modes: Frequency Extraction, Statistical (Markov) Extraction, and Validated Combinatorial/Markov Rule Generation.

It is built with speed and accuracy in mind, featuring multiprocessing, robust Hashcat syntax validation, and optional external cleanup integration (e.g., for cleanup-rules.bin).

✨ **Features**

1. Recursive File Search: Analyzes rule files in specified directories and subdirectories.

2. Parallel Processing: Uses multiprocessing to speed up rule file analysis.

3. Hashcat Syntax Validation: Ensures all generated rules (Combinatorial and Markov) are syntactically correct (i.e., operators have the right number of arguments).

**Extraction Modes:**

1. Frequency: Extracts the most frequently occurring unique rules.

2. Statistical (Markov): Extracts existing rules sorted by their statistical probability (Markov Log-Probability), prioritizing common, effective rule chains.

**Generation Modes:**

1. Combinatorial Generation: Generates new rules from the most frequent operators, up to a target count, with full syntax validation.

2. Statistical (Markov) Generation: Generates new rules by traversing a statistical Markov model built from the input rules, creating statistically probable and syntactically valid chains.

3. External Cleanup Integration: Supports running an external rule cleanup tool (like Hashcat's cleanup-rules.bin) on the generated output files for post-processing.

🚀 **Usage**

Prerequisites

You need Python 3.x. No external non-standard libraries are strictly required.

- Clone the repository (or just download the script)

```git clone https://github.com/A113L/Concentrator.git```

```cd Concentrator```

**Basic Analysis & Frequency Extraction**

Analyze rule files in the current directory and its subfolders, then extract the top 50,000 rules sorted by raw frequency:


```python3 concentrator_v1.4.1.py . -ob top_50k_freq.rule -t 50000```

**Statistical Extraction**

Extract the top 10,000 existing rules, but sort them by their Markov sequence probability (statistical weight):

```python3 concentrator_v1.4.1.py ./path/to/rules/ -s -t 10000 -ob top_10k_stat.rule```

**Validated Markov Rule Generation**

Generate 200,000 new, statistically probable rules of length 1 to 5, and save them to a derived file name.

```python3 concentrator_v1.4.1.py ./path/to/rules/ --generate_markov_rules -n 200000 -ml 1 5 -ob base_rules.rule```
- Output will be saved to: base_rules_markov.rule

**Validated Combinatorial Generation with Cleanup**

Generate up to 1 million syntactically valid combinatorial rules of length 1 to 3, and then pipe the output through an external cleanup utility.


```python3 concentrator_v1.4.1.py . --generate_combo -n 1000000 -l 1 3 -cb /usr/local/bin/cleanup-rules.bin -ca 2 -gc generated_combos.rule```

- The final cleaned file will be renamed like: generated_combos_CLEANED_2_[COUNT].rule

**Concentrator command line arguments**

```
usage: concentrator_v1.4.2.py [-h] [-ob OUTPUT_BASE_NAME] [-e] [-t TOP_RULES] [-s] [-g] [-n COMBO_TARGET] [-l COMBO_LENGTH [COMBO_LENGTH ...]] [-gm]
                              [-gt GENERATE_TARGET] [-ml MARKOV_LENGTH [MARKOV_LENGTH ...]] [-m MAX_LENGTH] [--temp-dir TEMP_DIR] [--in-memory]
                              [-cb CLEANUP_BIN] [-ca CLEANUP_ARG]
                              paths [paths ...]

Extracts top N rules, generates VALID combinatorial/Markov rules. Requires exactly one mode (-e, -g, or -gm). Supports recursive folder search.

positional arguments:
  paths                 Paths to rule files or directories to analyze. If a directory is provided, it will be searched recursively.

options:
  -h, --help            show this help message and exit
  -ob OUTPUT_BASE_NAME, --output_base_name OUTPUT_BASE_NAME
                        The base name for the output file. The script will append a suffix based on the mode (e.g., "_extracted.txt", "_combo.txt",
                        "_markov.txt").
  -e, --extract_rules   Enables rule extraction and sorting from input files. Uses -t for count.
  -t TOP_RULES, --top_rules TOP_RULES
                        The number of top existing rules to extract and save (used with -e).
  -s, --statistical_sort
                        Sorts EXTRACTED rules by Markov sequence probability instead of raw frequency (used with -e).
  -g, --generate_combo  Enables generating combinatorial rules. Uses -n for target count.
  -n COMBO_TARGET, --combo_target COMBO_TARGET
                        The approximate number of rules to generate in combinatorial mode (used with -g).
  -l COMBO_LENGTH [COMBO_LENGTH ...], --combo_length COMBO_LENGTH [COMBO_LENGTH ...]
                        The range of rule chain lengths for combinatorial mode (e.g., 1 3) (used with -g).
  -gm, --generate_markov_rules
                        Enables generating statistically probable Markov rules. Uses -gt for target count.
  -gt GENERATE_TARGET, --generate_target GENERATE_TARGET
                        The target number of rules to generate in Markov mode (used with -gm).
  -ml MARKOV_LENGTH [MARKOV_LENGTH ...], --markov_length MARKOV_LENGTH [MARKOV_LENGTH ...]
                        The range of rule chain lengths for Markov mode (e.g., 1 5) (used with -gm). Defaults to [1, 3].
  -m MAX_LENGTH, --max_length MAX_LENGTH
                        The maximum length for rules to be extracted/considered in analysis. Default is 31.
  --temp-dir TEMP_DIR   Optional: Specify a directory for temporary files.
  --in-memory           Process all rules entirely in RAM.
  -cb CLEANUP_BIN, --cleanup-bin CLEANUP_BIN
                        Optional: Path to the external cleanup binary (e.g., ./cleanup-rules.bin). If provided, it will run after rule generation.
  -ca CLEANUP_ARG, --cleanup-arg CLEANUP_ARG
                        Argument to pass to the cleanup binary (e.g., "2" for hashcat's cleanup-rules.bin).

```
https://hcrulestools.pages.dev/concentrator.static_workflow
