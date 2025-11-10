**Concentrator v1.4.2**

Unified Hashcat Rule Processor with validated combinatorial generation and statistical (Markov) rule generation.

**Features**

- 3 Processing Modes: Extract, Generate Combinatorial, Generate Markov

- Parallel Processing: Multi-core file analysis

- Syntax Validation: Ensures generated rules are valid Hashcat syntax

- Statistical Modeling: Markov chain probability for rule sequences

- External Cleanup: Optional integration with Hashcat's cleanup tools

**Examples:**

```
# Extract top 10k rules by frequency
python concentrator.py -e -t 10000 rules/ -ob my_rules

# Extract top 5k rules by statistical weight
python concentrator.py -e -s -t 5000 rules/ -ob statistical_rules
```

```
# Generate ~100k rules using top operators (lengths 1-3)
python concentrator.py -g -n 100000 -l 1 3 rules/ -ob combo_rules
```

```
# Generate 10k statistically probable rules
python concentrator.py -gm -gt 10000 -ml 1 5 rules/ -ob markov_rules
```

```
# Generate rules and run Hashcat's cleanup-rules.bin
python concentrator.py -gm -gt 5000 rules/ -ob clean_rules -cb ./cleanup-rules.bin -ca 2
```


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
https://hcrt.pages.dev/concentrator.static_workflow

**USAGE MINIMIZER RECOMMENDATIONS**

To optimize your generated rules and reduce file size:

* Hashcat's rulefilter:
   ./minimizer_cl.py rulesPath's 
   Filters rules based on various criteria

These tool can significantly reduce rule file size while
maintaining or even improving cracking effectiveness.

https://github.com/A113L/minimizer
