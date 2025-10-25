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


```python3 concentrator_v1.4.1.py . -o top_50k_freq.rule -t 50000```

**Statistical Extraction**
Extract the top 10,000 existing rules, but sort them by their Markov sequence probability (statistical weight):

```python3 concentrator_v1.4.1.py ./path/to/rules/ -s -t 10000 -o top_10k_stat.rule```

**Validated Markov Rule Generation**
Generate 200,000 new, statistically probable rules of length 1 to 5, and save them to a derived file name.

```python3 concentrator_v1.4.1.py ./path/to/rules/ --generate_markov_rules -n 200000 -ml 1 5 -o base_rules.rule```
- Output will be saved to: base_rules_markov.rule

**Validated Combinatorial Generation with Cleanup**
Generate up to 1 million syntactically valid combinatorial rules of length 1 to 3, and then pipe the output through an external cleanup utility.


```python3 concentrator_v1.4.1.py . --generate_combo -n 1000000 -l 1 3 -cb /usr/local/bin/cleanup-rules.bin -ca 2 -gc generated_combos.rule```
- The final cleaned file will be renamed like: generated_combos_CLEANED_2_[COUNT].rule

**Concentrator command line arguments**

```paths	(Positionals)	Required. Paths to rule files or directories to analyze (recursive search is enabled).
-t	--top_rules	eg. 10000	- number of top rules to extract (Frequency or Statistical mode).
-m	--max_length	eg. 31	- maximum length for rules to be extracted.
-o	--output_file	eg. optimized_top_rules.txt	- output file for extracted rules.
-s	--statistical_sort	- false	Sorts extracted rules by Markov probability instead of raw frequency.
-g	--generate_combo - enables combinatorial rule generation.
-gc	--combo_output_file eg.	generated_combos_validated.txt	- output file for combinatorial rules.
-n	--combo_target	eg. 100000	- the approximate number of rules to generate in combination/Markov mode.
-l	--combo_length - length range for combinatorial generation (e.g., 1 3).
-gm	--generate_markov_rules	-	enables statistical (Markov) rule generation.
-ml	--markov_length	-	length range for Markov generation. Defaults to --combo_length if not set.
--in-memory	-	process all rules entirely in RAM (useful for small datasets, risky for large ones).
-cb	--cleanup-bin	-	path to an external cleanup utility (e.g., cleanup-rules.bin).
-ca	--cleanup-arg	2	argument to pass to the external cleanup binary.```

https://roptimization.pages.dev
