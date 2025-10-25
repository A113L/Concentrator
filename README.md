# Concentrator
**Unified Hashcat Rule Processor**

[![concentrator.jpg](https://i.postimg.cc/jdfV4j7D/concentrator.jpg)](https://postimg.cc/JGRd1MK8)

The script is a Unified Hashcat Rule Processor designed for advanced analysis and optimization of password cracking rule sets. It operates in two distinct, sequential phases: Analysis and Extraction/Generation.


**Analysis Phase (Prerequisite)**

The script first performs a comprehensive analysis of the input rule files, filtering rules by maximum length (default: 31 characters) and removing comments. This phase generates two primary data sets:

*Full Rule Counts:* A frequency map of every unique rule.

*Operator Counts:* A count of every individual Hashcat operator (e.g., l, c, sX, ^1) found.


**Extraction & Generation Phases (Modes)**

The script supports three modes for outputting optimized rules:

A. *Frequency Sort (Default)*
It extracts and saves the top N unique rules based solely on their raw occurrence count in the input files.

B. *Statistical Sort (-s)*
It prioritizes rules by calculating a Markov Log-Probability Weight. A bigram/trigram Markov model is built from the observed operator sequences. Rules exhibiting statistically stronger, high-frequency operator transitions are ranked and extracted.

C. *Added the flag -gm/--generate_markov_rules* for statistical rule generation (1.4.0)
Allows generating new, statistically probable rule chains based on the model built from input files.

D. *Combinatorial Generation (-g)*
This optional mode generates a separate, comprehensive rule file by:
Determining the minimum number of most frequently used operators required to generate approximately N rules.
Using itertools.product to generate all possible combinations of these selected operators within a user-defined length range (e.g., 1 to 3 operators).
This results in a dense, new rule set optimized for coverage based on observed operator usage.

```
usage: concentrator_v1.4.1.py [-h] [-t TOP_RULES] [-o OUTPUT_FILE] [-m MAX_LENGTH] [-s] [-g] [-gc COMBO_OUTPUT_FILE] [-n COMBO_TARGET]
                              [-l COMBO_LENGTH [COMBO_LENGTH ...]] [-gm] [-ml MARKOV_LENGTH [MARKOV_LENGTH ...]] [--temp-dir TEMP_DIR] [--in-memory]
                              [-cb CLEANUP_BIN] [-ca CLEANUP_ARG]
                              paths [paths ...]

Extracts top N rules sorted by raw frequency, statistical probability, or generates VALID combinatorial/Markov rules, with optional post-processing cleanup.
Supports recursive folder search.

positional arguments:
  paths                 Paths to rule files or directories to analyze. If a directory is provided, it will be searched recursively.

options:
  -h, --help            show this help message and exit
  -t TOP_RULES, --top_rules TOP_RULES
                        The number of top existing rules to extract and save.
  -o OUTPUT_FILE, --output_file OUTPUT_FILE
                        The name of the output file for extracted rules (also used as base for Markov output).
  -m MAX_LENGTH, --max_length MAX_LENGTH
                        The maximum length for rules to be extracted. Default is 31.
  -s, --statistical_sort
                        Sorts EXTRACTED rules by Markov sequence probability instead of raw frequency.
  -g, --generate_combo  Enables generating a separate file with combinatorial rules from top operators.
  -gc COMBO_OUTPUT_FILE, --combo_output_file COMBO_OUTPUT_FILE
                        The name of the output file for generated combinatorial rules.
  -n COMBO_TARGET, --combo_target COMBO_TARGET
                        The approximate number of rules to generate in combinatorial mode.
  -l COMBO_LENGTH [COMBO_LENGTH ...], --combo_length COMBO_LENGTH [COMBO_LENGTH ...]
                        The range of rule chain lengths for combinatorial mode (e.g., 1 3).
  -gm, --generate_markov_rules
                        Enables generating statistically probable rules by traversing the Markov model.
  -ml MARKOV_LENGTH [MARKOV_LENGTH ...], --markov_length MARKOV_LENGTH [MARKOV_LENGTH ...]
                        The range of rule chain lengths for Markov mode (e.g., 1 5). Defaults to --combo_length if not set.
  --temp-dir TEMP_DIR   Optional: Specify a directory for temporary files.
  --in-memory           Process all rules entirely in RAM.
  -cb CLEANUP_BIN, --cleanup-bin CLEANUP_BIN
                        Optional: Path to the external cleanup binary (e.g., ./cleanup-rules.bin). If provided, it will run after rule generation.
  -ca CLEANUP_ARG, --cleanup-arg CLEANUP_ARG
                        Argument to pass to the cleanup binary (e.g., "2" for hashcat's cleanup-rules.bin).

