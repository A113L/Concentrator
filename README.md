# Concentrator
Unified Hashcat Rule Processor

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

C. *Combinatorial Generation (-g)*
This optional mode generates a separate, comprehensive rule file by:
Determining the minimum number of most frequently used operators required to generate approximately N rules.
Using itertools.product to generate all possible combinations of these selected operators within a user-defined length range (e.g., 1 to 3 operators).
This results in a dense, new rule set optimized for coverage based on observed operator usage.

```usage: concentrator_v1.1.0.py [-h] [-t TOP_RULES] [-o OUTPUT_FILE] [-m MAX_LENGTH] [-s] [-g] [-n COMBO_TARGET] [-l COMBO_LENGTH [COMBO_LENGTH ...]]
                              [-gc COMBO_OUTPUT_FILE] [--temp-dir TEMP_DIR] [--in-memory]
                              filepaths [filepaths ...]

positional arguments:
  filepaths             Paths to the rule files to analyze.

options:
  -h, --help            show this help message and exit
  -t TOP_RULES, --top_rules TOP_RULES
                        The number of top existing rules to extract and save (for Frequency/Statistical modes).
  -o OUTPUT_FILE, --output_file OUTPUT_FILE
                        The name of the output file for extracted rules.
  -m MAX_LENGTH, --max_length MAX_LENGTH
                        The maximum length (in characters/bytes) for rules to be extracted. Default is 31.
  -s, --statistical_sort
                        Sorts rules by Markov sequence probability (statistical strength) instead of raw frequency.
  -g, --generate_combo  Enables generating a separate file with combinatorial rules from top operators.
  -n COMBO_TARGET, --combo_target COMBO_TARGET
                        The approximate number of rules to generate in combinatorial mode.
  -l COMBO_LENGTH [COMBO_LENGTH ...], --combo_length COMBO_LENGTH [COMBO_LENGTH ...]
                        The range of rule chain lengths for combinatorial mode (e.g., 1 3).
  -gc COMBO_OUTPUT_FILE, --combo_output_file COMBO_OUTPUT_FILE
                        The name of the output file for generated rules.
  --temp-dir TEMP_DIR   Optional: Specify a directory for temporary files created during parallel processing (e.g., a fast SSD path). Ignored if --in-memory
                        is used.
  --in-memory           Process all rules entirely in RAM, skipping temporary file creation and disk I/O for rule collection.


```
