# Concentrator v1.1.0
# Description: Unified Hashcat Rule Processor with Combinatorial Generation.
# Extracts rules based on either raw frequency or Markov sequence probability.

import sys
import re
import argparse
from collections import defaultdict
import math
import itertools 
import multiprocessing
import os 
import tempfile 

# --- Settings Required for Analysis ---
ALL_RULE_CHARS = set("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ:,.lu.#()=%!?|~+*-^$sStTiIoOcCrRyYzZeEfFxXdDpPbBqQ`[]><@&vV")
OPERATORS_REQUIRING_ARGS = {
        's': 2, 'S': 2, 't': 2, 'T': 2, 'i': 2, 'I': 2, 'o': 2, 'O': 2, 'c': 2, 'C': 2, 'r': 2, 'R': 2, 'y': 2, 'Y': 2, 'z': 2, 'Z': 2, 'e': 2, 'E': 2,
        'f': 1, 'F': 1, 'x': 1, 'X': 1, 'd': 1, 'D': 1, 'p': 1, 'P': 1, 'b': 1, 'B': 1, 'q': 1, 'Q': 1, '`': 1, '[': 1, ']': 1, '>': 1, '<': 1, '@': 1, '&': 1,
        'v': 3, 'V': 3,}
SIMPLE_OPERATORS = [
        ':', ',', 'l', 'u', '.', '#', '(', ')', '=', '%', '!', '?', '|', '~', '+', '*', '-', '^', '$']
ALL_OPERATORS = list(OPERATORS_REQUIRING_ARGS.keys()) + SIMPLE_OPERATORS
for i in range(10):
        ALL_OPERATORS.append(f'${i}')
        ALL_OPERATORS.append(f'^{i}')
# Compile Regex Pattern ONCE
REGEX_OPERATORS = [re.escape(op) for op in ALL_OPERATORS]
COMPILED_REGEX = re.compile('|'.join(filter(None, REGEX_OPERATORS)))

# Global variables to store CLI settings (required for multiprocessing workers)
_TEMP_DIR_PATH = None
_IN_MEMORY_MODE = False

def set_global_flags(temp_dir_path, in_memory_mode):
    """Sets the global flags required by worker processes."""
    global _TEMP_DIR_PATH
    global _IN_MEMORY_MODE
    _IN_MEMORY_MODE = in_memory_mode

    if temp_dir_path and not in_memory_mode:
        _TEMP_DIR_PATH = temp_dir_path
        if not os.path.isdir(_TEMP_DIR_PATH):
            try:
                os.makedirs(_TEMP_DIR_PATH, exist_ok=True)
                print(f"Using temporary directory: {_TEMP_DIR_PATH}")
            except OSError as e:
                print(f"Warning: Could not create temporary directory at {temp_dir_path}. Falling back to default system temp directory. Error: {e}", flush=True)
                _TEMP_DIR_PATH = None
    elif in_memory_mode:
         print("In-Memory Mode activated. Temporary files will be skipped.")

# --- Core Worker Function for File Analysis (Runs in a separate process) ---
def process_single_file(filepath, max_rule_length):
    """
    Processes a single rule file. Depending on the global flag _IN_MEMORY_MODE,
    it either writes rules to a temporary file or returns them directly in memory.
    """
    operator_counts = defaultdict(int)
    full_rule_counts = defaultdict(int)
    clean_rules_list = []
    temp_rule_filepath = None
    
    # Check the global flag (inherited from the main process)
    global _IN_MEMORY_MODE, _TEMP_DIR_PATH

    try:
        with open(filepath, 'r', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or len(line) > max_rule_length:
                    continue
                                                
                clean_line = ''.join(c for c in line if c in ALL_RULE_CHARS)
                if not clean_line: continue
                                                
                # 1. Frequency Count
                full_rule_counts[clean_line] += 1
                                
                # 2. Store rule (always collect in memory first)
                clean_rules_list.append(clean_line)
                                                
                # 3. Operator Count
                for match in COMPILED_REGEX.finditer(clean_line):
                    op = match.group(0)
                    if len(op) > 1 and op[0] in ('$', '^') and op[1].isdigit():
                        operator_counts[op[0]] += 1
                    else:
                        operator_counts[op] += 1
        
        if not _IN_MEMORY_MODE:
            # --- FILE MODE: Write collected rules to a temporary file ---
            temp_rule_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding='utf-8', dir=_TEMP_DIR_PATH)
            temp_rule_filepath = temp_rule_file.name
            for rule in clean_rules_list:
                temp_rule_file.write(rule + '\n')
            temp_rule_file.close() 
            print(f"File analysis complete: {filepath}. Temp rules saved to {temp_rule_filepath}", flush=True)
            # In file mode, return the path and an empty list for total_all_clean_rules
            return operator_counts, full_rule_counts, [], temp_rule_filepath
        else:
            # --- IN-MEMORY MODE: Return the list of rules directly ---
            print(f"File analysis complete: {filepath}. Rules returned in memory.", flush=True)
            # In memory mode, return the list of rules and None for temp_filepath
            return operator_counts, full_rule_counts, clean_rules_list, None
            
    except Exception as e:
        print(f"An error occurred while processing {filepath}: {e}", flush=True)
        if temp_rule_filepath and os.path.exists(temp_rule_filepath):
            os.unlink(temp_rule_filepath) # Clean up the failed temp file
        return defaultdict(int), defaultdict(int), [], None

def analyze_rule_files_parallel(filepaths, max_rule_length):
    """
    Parallel file analysis using multiprocessing.Pool. Handles both file-based 
    and in-memory aggregation of rules based on the global flag.
    """
    total_operator_counts = defaultdict(int)
    total_full_rule_counts = defaultdict(int) 
    
    temp_files_to_merge = []
    total_all_clean_rules = [] # List to hold all rules (either merged from files or directly from workers)
    
    global _IN_MEMORY_MODE # Get the mode from global setting
    
    num_processes = min(os.cpu_count() or 1, len(filepaths))
    tasks = [(filepath, max_rule_length) for filepath in filepaths]
    
    print(f"Starting parallel analysis of {len(filepaths)} files using {num_processes} processes...")
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Results contain: (op_counts, rule_counts_worker, clean_rules_list, temp_filepath)
        results = pool.starmap(process_single_file, tasks)
        
    # Aggregate results from all processes in the main thread
    for op_counts, rule_counts_worker, clean_rules_worker, temp_filepath in results:
        # 1. Aggregate Operator Counts
        for op, count in op_counts.items():
            total_operator_counts[op] += count
            
        # 2. Aggregate Rule Counts (Unique rules)
        for rule, count in rule_counts_worker.items():
            total_full_rule_counts[rule] += count
            
        # 3. Store rules/paths based on mode
        if _IN_MEMORY_MODE:
            # In-Memory Mode: Rules are already in the list returned by the worker
            total_all_clean_rules.extend(clean_rules_worker)
        else:
            # File Mode: Store temporary file path for later merging
            if temp_filepath:
                temp_files_to_merge.append(temp_filepath)
            
    # 4. Final Rule Aggregation (if in File Mode)
    if not _IN_MEMORY_MODE:
        print("\nMerging temporary rule files into memory for Markov processing...")
        for temp_filepath in temp_files_to_merge:
            try:
                with open(temp_filepath, 'r', encoding='utf-8') as f:
                    # Appending the list of rules from the temp file
                    total_all_clean_rules.extend([line.strip() for line in f])
                os.unlink(temp_filepath) # Delete file after use
            except Exception as e:
                print(f"Error merging temp file {temp_filepath}: {e}")
            
    print(f"Total rules loaded into memory: {len(total_all_clean_rules)}")
    
    sorted_op_counts = sorted(total_operator_counts.items(), key=lambda item: item[1], reverse=True)
    return sorted_op_counts, total_full_rule_counts, total_all_clean_rules

# --- Markov and Extraction Functions ---
def get_markov_weighted_rules(unique_rules):
    """
    Builds the Markov model and calculates the log-probability weight for each unique rule.
    Used for the Statistical Sort mode.
    """
    print("\n--- Calculating Statistical Weight (Markov Sequence Probability) ---")
    markov_model_counts = defaultdict(lambda: defaultdict(int))
    START_CHAR = '^'         
    weighted_rules = []
    
    # 1. Build the Markov Model (Trigrams and Bigrams)
    for rule in unique_rules.keys():
        markov_model_counts[START_CHAR][rule[0]] += 1
        for i in range(len(rule) - 1):
            markov_model_counts[rule[i]][rule[i+1]] += 1
        for i in range(len(rule) - 2):
            prefix = rule[i:i+2]
            suffix = rule[i+2]
            markov_model_counts[prefix][suffix] += 1
            
    total_transitions = {char: sum(counts.values()) for char, counts in markov_model_counts.items()}
    
    # 2. Calculate Log-Probability Weight for each rule
    for rule in unique_rules.keys():
        log_probability_sum = 0.0
        current_prefix = START_CHAR
        next_char = rule[0]
        
        if next_char in markov_model_counts[current_prefix]:
            probability = markov_model_counts[current_prefix][next_char] / total_transitions[current_prefix]
            log_probability_sum += math.log(probability)
        else:
            continue
            
        for i in range(len(rule) - 1):
            if i >= 1:
                current_prefix = rule[i-1:i+1] # Trigram prefix
            else:
                current_prefix = rule[i] # Bigram prefix
            next_char = rule[i+1]
            
            if next_char in markov_model_counts[current_prefix]:
                probability = markov_model_counts[current_prefix][next_char] / total_transitions[current_prefix]
                log_probability_sum += math.log(probability)
            else:
                log_probability_sum = -float('inf') 
                break
                
        if log_probability_sum > -float('inf'):
            weighted_rules.append((rule, log_probability_sum))
            
    sorted_weighted_rules = sorted(weighted_rules, key=lambda item: item[1], reverse=True)
    return sorted_weighted_rules

# --- Combinatorial Generation Functions (Parallelized) ---
def find_min_operators_for_target(sorted_operators, target_rules, min_len, max_len):
    """
    Finds the minimum number of top operators needed to generate the target number of rules.
    """
    current_rule_count = 0
    num_operators = 0
    while current_rule_count < target_rules and num_operators < len(sorted_operators):
        num_operators += 1
        top_ops = [op for op, count in sorted_operators[:num_operators]]
        current_rule_count = 0
        for length in range(min_len, max_len + 1):
            current_rule_count += (len(top_ops) ** length)
            
    return [op for op, count in sorted_operators[:num_operators]]

def generate_rules_for_length(args):
    """Worker function to generate rules for a single length (L)."""
    top_operators, length = args
    generated_rules = set()
    for combo in itertools.product(top_operators, repeat=length):
        new_rule = ''.join(combo)
        generated_rules.add(new_rule)
    return generated_rules

def generate_rules_parallel(top_operators, min_len, max_len):
    """
    Generates all rules in parallel based on a list of operators and a length range.
    """
    all_lengths = list(range(min_len, max_len + 1))
    tasks = [(top_operators, length) for length in all_lengths]
    
    num_processes = min(os.cpu_count() or 1, len(all_lengths))
    print(f"Generating new rules of length {min_len} to {max_len} using {len(top_operators)} operators across {num_processes} processes...")
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(generate_rules_for_length, tasks)
        
    # Aggregate results from all processes (sets are merged)
    generated_rules = set().union(*results)
    return generated_rules

# --- Utility Functions (Modified for combinatorial warning) ---
def save_rules_to_file(rules_data, filename, mode):
    """
    Saves the rules to a file. Handles both extracted data (list of tuples) 
    and generated data (set of strings).
    """
    if mode == 'frequency':
        rules_to_save = [r[0] for r in rules_data]
        header = "# Rules extracted and sorted by RAW FREQUENCY (most occurrences).\n"
    elif mode == 'statistical':
        rules_to_save = [r[0] for r in rules_data]
        header = "# Rules extracted and sorted by STATISTICAL WEIGHT (Markov Log-Probability).\n"
    elif mode == 'combo':
        rules_to_save = sorted(list(rules_data)) # Ensure it is a sorted list of strings
        # Warning about potentially invalid rules:
        header = "# Rules generated combinatorially from top used operators. WARNING: This set may contain a large number of syntactically INVALID rules (e.g., missing arguments) that Hashcat will skip.\n"
    else:
        rules_to_save = sorted(list(rules_data))
        header = f"# Rules saved: {len(rules_to_save)} total.\n"
                
    print(f"\nSaving {len(rules_to_save)} rules to file '{filename}'...")
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(header)
        for rule in rules_to_save:
            f.write(f"{rule}\n")
                    
    print("Done.")
    sys.stdout.flush()

# --- Main Execution Logic ---
if __name__ == '__main__':
    # Fix for multiprocessing on Windows
    multiprocessing.freeze_support() 
    
    parser = argparse.ArgumentParser(description='Extracts top N rules sorted by raw frequency (default) or Markov sequence probability (-s), or generates combinatorial rules (-g). NOW WITH MULTIPROCESSING AND COMPATIBILITY FIX.')
    parser.add_argument('filepaths', nargs='+', help='Paths to the rule files to analyze.')
    parser.add_argument('-t', '--top_rules', type=int, default=10000, help='The number of top existing rules to extract and save (for Frequency/Statistical modes).')
    parser.add_argument('-o', '--output_file', type=str, default='optimized_top_rules.txt', help='The name of the output file for extracted rules.')
    parser.add_argument('-m', '--max_length', type=int, default=31, help='The maximum length (in characters/bytes) for rules to be extracted. Default is 31.')
    parser.add_argument('-s', '--statistical_sort', action='store_true', help='Sorts rules by Markov sequence probability (statistical strength) instead of raw frequency.')
    parser.add_argument('-g', '--generate_combo', action='store_true', help='Enables generating a separate file with combinatorial rules from top operators.')
    parser.add_argument('-n', '--combo_target', type=int, default=100000, help='The approximate number of rules to generate in combinatorial mode.')
    parser.add_argument('-l', '--combo_length', nargs='+', type=int, default=[1, 3], help='The range of rule chain lengths for combinatorial mode (e.g., 1 3).')
    parser.add_argument('-gc', '--combo_output_file', type=str, default='generated_combos.txt', help='The name of the output file for generated rules.')
    
    # --- New CLI Flags ---
    parser.add_argument('--temp-dir', type=str, default=None, 
                        help='Optional: Specify a directory for temporary files created during parallel processing (e.g., a fast SSD path). Ignored if --in-memory is used.')
    parser.add_argument('--in-memory', action='store_true', 
                        help='Process all rules entirely in RAM, skipping temporary file creation and disk I/O for rule collection.')
    # -----------------------
            
    args = parser.parse_args()
    
    # Set the global flags (will be inherited by child processes)
    set_global_flags(args.temp_dir, args.in_memory)
        
    if args.generate_combo:
        if len(args.combo_length) not in [1, 2] or (len(args.combo_length) == 2 and args.combo_length[0] > args.combo_length[1]):
            print("Error: Invalid chain length range for combinatorial mode. Use 'L' or 'L_min L_max'.")
            sys.exit(1)
        min_len = args.combo_length[0]
        max_len = args.combo_length[-1]

    # --- 1. Parallel Rule File Analysis (Compatible Mode) ---
    print("--- 1. Starting Parallel Rule File Analysis (Compatible Mode) ---")
    
    sorted_op_counts, full_rule_counts, all_clean_rules = analyze_rule_files_parallel(args.filepaths, args.max_length) 
    
    if not sorted_op_counts:
        print("No operators found in files. Exiting.")
        sys.exit(1)

    # --- 2. Determine Sorting Method and Extract Top N ---
    if args.statistical_sort:
        mode = 'statistical'
        print("\n--- Mode: Statistical Sort (Markov Weight) ---")
        sorted_rule_data = get_markov_weighted_rules(full_rule_counts)
    else:
        mode = 'frequency'
        print("\n--- Mode: Frequency Sort (Raw Count) ---")
        sorted_rule_data = sorted(full_rule_counts.items(), key=lambda item: item[1], reverse=True)
        
    top_rules_data = sorted_rule_data[:args.top_rules]

    # --- 3. Display analysis results ---
    print(f"\n--- 2. Analysis Results (Mode: {mode.upper()}) ---")
    print("Most frequently used operators (TOP 10):")
    for op, count in sorted_op_counts[:10]:
        print(f"  '{op}': {count} times")
        
    print(f"\nTop {min(10, len(top_rules_data))} Extracted Rules:")
    if mode == 'frequency':
        for rule, count in top_rules_data[:10]:
            print(f"  '{rule}': Count {count}")
    else:
        for rule, weight in top_rules_data[:10]:
            print(f"  '{rule}': Weight {weight:.4f}")
            
    print(f"\nExtracted {len(top_rules_data)} top unique rules (max length: {args.max_length} characters).")

    # --- 4. Save the extracted results ---
    save_rules_to_file(top_rules_data, args.output_file, mode)

    # --- 5. Parallel Combinatorial Generation Logic ---
    if args.generate_combo:
        print("\n" + "="*50)
        print("--- 3. Starting PARALLEL Combinatorial Rule Generation ---")
        print("="*50)
        
        min_operators_needed = find_min_operators_for_target(
            sorted_op_counts, 
            args.combo_target, 
            min_len, 
            max_len
        )
                
        print(f"Using {len(min_operators_needed)} most frequent operators to target ~{args.combo_target} rules (Length {min_len}-{max_len}).")
        
        generated_rules_set = generate_rules_parallel(min_operators_needed, min_len, max_len)
        
        save_rules_to_file(generated_rules_set, args.combo_output_file, 'combo')
        
    print("\nComplete! The script has finished. Check the output file(s) for the optimized rules.")
    sys.stdout.flush()
