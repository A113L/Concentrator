# Concentrator v1.4.1
# Description: Unified Hashcat Rule Processor with Validated Combinatorial Generation
# and Statistical (Markov) Rule Generation. Supports recursive file search and external cleanup

import sys
import re
import argparse
from collections import defaultdict
import math
import itertools
import multiprocessing
import os
import tempfile
import subprocess
import random # Needed for Markov generation

# --- Hashcat Rule Syntax Definitions ---
ALL_RULE_CHARS = set("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ:,.lu.#()=%!?|~+*-^$sStTiIoOcCrRyYzZeEfFxXdDpPbBqQ`[]><@&vV")
OPERATORS_REQUIRING_ARGS = {
    's': 2, 'S': 2, 't': 2, 'T': 2, 'i': 2, 'I': 2, 'o': 2, 'O': 2, 'c': 2, 'C': 2, 'r': 2, 'R': 2, 'y': 2, 'Y': 2, 'z': 2, 'Z': 2, 'e': 2, 'E': 2,
    'f': 1, 'F': 1, 'x': 1, 'X': 1, 'd': 1, 'D': 1, 'p': 1, 'P': 1, 'b': 1, 'B': 1, 'q': 1, 'Q': 1, '`': 1, '[': 1, ']': 1, '>': 1, '<': 1, '@': 1, '&': 1,
    'v': 3, 'V': 3,
}
SIMPLE_OPERATORS = [
    ':', ',', 'l', 'u', '.', '#', '(', ')', '=', '%', '!', '?', '|', '~', '+', '*', '-', '^', '$']

ALL_OPERATORS = list(OPERATORS_REQUIRING_ARGS.keys()) + SIMPLE_OPERATORS
for i in range(10):
    ALL_OPERATORS.append(f'${i}')
    ALL_OPERATORS.append(f'^{i}')

REGEX_OPERATORS = [re.escape(op) for op in ALL_OPERATORS if op not in ('$', '^')] # Exclude $ and ^ as standalone single chars in this list
REGEX_OPERATORS += [r'\$[0-9]', r'\^[0-9]', r'\:', r'\,', r'\.', r'\#', r'\(', r'\)', r'\=', r'\%', r'\!', r'\?', r'\|', r'\~', r'\+', r'\*', r'\-', r'\^', r'\$']
COMPILED_REGEX = re.compile('|'.join(filter(None, sorted(list(set(REGEX_OPERATORS)), key=len, reverse=True)))) # Sort by length for correct matching

_TEMP_DIR_PATH = None
_IN_MEMORY_MODE = False
_OP_REQS = OPERATORS_REQUIRING_ARGS
_VALID_CHARS = ALL_RULE_CHARS

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

# --- Hashcat Syntax Validation Function ---
def is_valid_hashcat_rule(rule: str, op_reqs: dict, valid_chars: set) -> bool:
    """
    Checks if a generated rule string has valid Hashcat syntax, specifically 
    ensuring operators have the correct number of arguments.
    """
    i = 0
    while i < len(rule):
        op = rule[i]
        
        if op in ('$', '^') and i + 1 < len(rule) and rule[i+1].isdigit():
            # Handle positional operators $X and ^X
            i += 2 
            continue
        
        if op not in op_reqs:
            if op not in valid_chars:
                return False
            i += 1
            continue
            
        required_args = op_reqs.get(op, 0)
        
        if i + 1 + required_args > len(rule):
            return False
            
        args_segment = rule[i+1 : i + 1 + required_args]
        if not all(arg in valid_chars for arg in args_segment):
            return False
            
        i += 1 + required_args 
            
    return True

# --- Core Worker Function for File Analysis ---
def process_single_file(filepath, max_rule_length):
    """Processes a single rule file and counts frequencies."""
    operator_counts = defaultdict(int)
    full_rule_counts = defaultdict(int)
    clean_rules_list = []
    temp_rule_filepath = None
    
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
                    
                # 2. Store rule
                clean_rules_list.append(clean_line)
                    
                # 3. Operator Count
                # Use the pre-compiled regex for operator counting
                i = 0
                while i < len(clean_line):
                    match = COMPILED_REGEX.match(clean_line, i)
                    if match:
                        op = match.group(0)
                        if op in ('$', '^') and i + 1 < len(clean_line) and clean_line[i+1].isdigit():
                            # Positional operators are counted by their symbol ($ or ^)
                            operator_counts[op] += 1
                            i += 2
                        elif op in OPERATORS_REQUIRING_ARGS.keys():
                            # Operator with arguments
                            operator_counts[op] += 1
                            i += 1 + OPERATORS_REQUIRING_ARGS[op]
                        else:
                            # Simple operator
                            operator_counts[op] += 1
                            i += len(op)
                    else:
                        # Character is an argument or not an operator
                        i += 1
            
        if not _IN_MEMORY_MODE:
            # --- FILE MODE: Write collected rules to a temporary file ---
            temp_rule_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding='utf-8', dir=_TEMP_DIR_PATH)
            temp_rule_filepath = temp_rule_file.name
            for rule in clean_rules_list:
                temp_rule_file.write(rule + '\n')
            temp_rule_file.close()  
            print(f"File analysis complete: {filepath}. Temp rules saved to {temp_rule_filepath}", flush=True)
            return operator_counts, full_rule_counts, [], temp_rule_filepath
        else:
            # --- IN-MEMORY MODE: Return the list of rules directly ---
            print(f"File analysis complete: {filepath}. Rules returned in memory.", flush=True)
            return operator_counts, full_rule_counts, clean_rules_list, None
            
    except Exception as e:
        print(f"An error occurred while processing {filepath}: {e}", flush=True)
        if temp_rule_filepath and os.path.exists(temp_rule_filepath):
            os.unlink(temp_rule_filepath)  
        return defaultdict(int), defaultdict(int), [], None

def analyze_rule_files_parallel(filepaths, max_rule_length):
    """Parallel file analysis using multiprocessing.Pool."""
    total_operator_counts = defaultdict(int)
    total_full_rule_counts = defaultdict(int) 
    
    temp_files_to_merge = []
    total_all_clean_rules = []
    
    global _IN_MEMORY_MODE
    
    existing_filepaths = [fp for fp in filepaths if os.path.exists(fp) and os.path.isfile(fp)]
    
    if not existing_filepaths:
        print("Warning: No valid rule files found to process.")
        return defaultdict(int), defaultdict(int), [], None

    num_processes = min(os.cpu_count() or 1, len(existing_filepaths))
    tasks = [(filepath, max_rule_length) for filepath in existing_filepaths]
    
    print(f"Starting parallel analysis of {len(existing_filepaths)} files using {num_processes} processes...")
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(process_single_file, tasks)
        
    for op_counts, rule_counts_worker, clean_rules_worker, temp_filepath in results:
        for op, count in op_counts.items():
            total_operator_counts[op] += count
            
        for rule, count in rule_counts_worker.items():
            total_full_rule_counts[rule] += count
            
        if _IN_MEMORY_MODE:
            total_all_clean_rules.extend(clean_rules_worker)
        else:
            if temp_filepath:
                temp_files_to_merge.append(temp_filepath)
            
    if not _IN_MEMORY_MODE:
        print("\nMerging temporary rule files into memory for Markov processing...")
        for temp_filepath in temp_files_to_merge:
            try:
                with open(temp_filepath, 'r', encoding='utf-8') as f:
                    total_all_clean_rules.extend([line.strip() for line in f])
                os.unlink(temp_filepath)
            except Exception as e:
                print(f"Error merging temp file {temp_filepath}: {e}")
            
    print(f"Total rules loaded into memory: {len(total_all_clean_rules)}")
    
    sorted_op_counts = sorted(total_operator_counts.items(), key=lambda item: item[1], reverse=True)
    return sorted_op_counts, total_full_rule_counts, total_all_clean_rules

# --- Markov and Extraction Functions ---
def get_markov_model(unique_rules):
    """Builds the Markov model (counts) and transition probabilities."""
    print("\n--- Building Markov Sequence Probability Model ---")
    markov_model_counts = defaultdict(lambda: defaultdict(int))
    START_CHAR = '^'             
    
    # 1. Build the Markov Model (Bigrams and Trigrams)
    for rule in unique_rules.keys():
        # Bigram from start
        markov_model_counts[START_CHAR][rule[0]] += 1
        
        # Bigrams O(i) -> O(i+1)
        for i in range(len(rule) - 1):
            markov_model_counts[rule[i]][rule[i+1]] += 1
            
        # Trigrams O(i-1)O(i) -> O(i+1)
        for i in range(len(rule) - 2):
            prefix = rule[i:i+2]
            suffix = rule[i+2]
            markov_model_counts[prefix][suffix] += 1
            
    total_transitions = {char: sum(counts.values()) for char, counts in markov_model_counts.items()}
    
    # 2. Calculate Probabilities
    markov_probabilities = defaultdict(lambda: defaultdict(float))
    for prefix, next_counts in markov_model_counts.items():
        total = total_transitions[prefix]
        for next_op, count in next_counts.items():
            markov_probabilities[prefix][next_op] = count / total
            
    return markov_probabilities, total_transitions

def get_markov_weighted_rules(unique_rules, markov_probabilities, total_transitions):
    """Calculates the log-probability weight for each unique rule based on the model."""
    weighted_rules = []

    # 3. Calculate Log-Probability Weight for each rule
    for rule in unique_rules.keys():
        log_probability_sum = 0.0
        
        # P(O1 | Start)
        current_prefix = '^'
        next_char = rule[0]
        if next_char in markov_probabilities[current_prefix]:
            probability = markov_probabilities[current_prefix][next_char]
            log_probability_sum += math.log(probability)
        else:
            continue # Skip rules not starting with a known sequence
            
        # P(Oi | O_i-1) or P(Oi | O_i-2 O_i-1)
        for i in range(len(rule) - 1):
            # Try Trigram (O_i-1 O_i -> O_i+1) first
            if i >= 1:
                current_prefix = rule[i-1:i+1] # Trigram prefix
                next_char = rule[i+1]
                if current_prefix in markov_probabilities and next_char in markov_probabilities[current_prefix]:
                    probability = markov_probabilities[current_prefix][next_char]
                    log_probability_sum += math.log(probability)
                    continue # Trigram found, move to next step
            
            # Fallback to Bigram (O_i -> O_i+1)
            current_prefix = rule[i]  
            next_char = rule[i+1]
            if current_prefix in markov_probabilities and next_char in markov_probabilities[current_prefix]:
                probability = markov_probabilities[current_prefix][next_char]
                log_probability_sum += math.log(probability)
            else:
                log_probability_sum = -float('inf')  
                break
            
        if log_probability_sum > -float('inf'):
            weighted_rules.append((rule, log_probability_sum))
            
    sorted_weighted_rules = sorted(weighted_rules, key=lambda item: item[1], reverse=True)
    return sorted_weighted_rules

# --- UPDATED: Markov Rule Generation Logic ---
def generate_rules_from_markov_model(markov_probabilities, target_rules, min_len, max_len):
    """
    Generates new rules by traversing the Markov model, prioritizing high-probability transitions.
    The generation stops attempting new rules once 'target_rules' unique, valid rules are found.
    """
    print(f"\n--- Generating Rules via Markov Model Traversal ({min_len}-{max_len} Operators) ---")
    generated_rules = set()
    START_CHAR = '^'
    
    def get_next_operator(current_prefix):
        """Returns the next operator based on probability distribution (weighted random choice)."""
        if current_prefix not in markov_probabilities:
            return None
        
        choices = list(markov_probabilities[current_prefix].keys())
        weights = list(markov_probabilities[current_prefix].values())
        
        if not choices:
            return None
        
        # Weighted random choice (simulating the probability)
        return random.choices(choices, weights=weights, k=1)[0]
    
    # Use a maximum number of attempts (e.g., 5 times the target) to prevent infinite loops
    generation_attempts = target_rules * 5 
    
    for attempt in range(generation_attempts):
        # Integration of -t flag: Stop once the target number of unique rules is reached
        if len(generated_rules) >= target_rules:
            break

        # Start rule with the most probable starting operator (or weighted random)
        current_rule = get_next_operator(START_CHAR)
        if not current_rule: continue
        
        # Traverse until max_len
        while len(current_rule) < max_len:
            last_op = current_rule[-1]
            last_two_ops = current_rule[-2:] if len(current_rule) >= 2 else None
            
            next_op = None
            
            # 1. Try Trigram transition (more specific context)
            if last_two_ops and last_two_ops in markov_probabilities:
                next_op = get_next_operator(last_two_ops)
            
            # 2. Fallback to Bigram transition
            if not next_op and last_op in markov_probabilities:
                next_op = get_next_operator(last_op)
                
            if not next_op:
                break # Cannot continue the sequence
            
            current_rule += next_op
            
            # Check for completion based on min_len
            if len(current_rule) >= min_len and len(current_rule) <= max_len:
                if is_valid_hashcat_rule(current_rule, _OP_REQS, _VALID_CHARS):
                    generated_rules.add(current_rule)

    print(f"Generated {len(generated_rules)} statistically probable and syntactically valid rules (Target: {target_rules}).")
    
    # Calculate weights for the generated rules for final sorting
    if generated_rules:
        # Create a dummy frequency count for the generated rules
        generated_rule_counts = {rule: 1 for rule in generated_rules}
        
        # Use the pre-built model for weighting
        weighted_output = get_markov_weighted_rules(generated_rule_counts, markov_probabilities, {}) 
        
        # Crucial: Trim the final output to the target number of rules after sorting by weight
        return weighted_output[:target_rules]
        
    return []

# --- Utility Functions (Cleanup and Save) ---
def save_rules_to_file(rules_data, filename, mode):
    """Saves the rules to a file."""
    
    # Check if data is a list of (rule, weight/count) tuples
    if mode in ('frequency', 'statistical') or (isinstance(rules_data, list) and rules_data and isinstance(rules_data[0], tuple)):
        rules_to_save = [r[0] for r in rules_data]
        if mode == 'frequency':
            header = "# Rules extracted and sorted by RAW FREQUENCY (most occurrences).\n"
        else: # mode is 'statistical' or statistically sorted 'combo' / 'markov'
              header = (
                  f"# Rules extracted/generated and sorted by STATISTICAL WEIGHT (Markov Log-Probability). Mode: {mode.upper()}\n"
                  "# NOTE: Rules in this mode may benefit from post-processing with external tools like Hashcat's 'rulefilter' or 'RuleCleaner' to remove semantically useless sequences.\n"
              )

    # Check if data is a set/list of rule strings (Combinatorial mode or just a list of rules)
    elif mode in ('combo', 'markov_generated'):
        rules_to_save = sorted(list(rules_data))
        if mode == 'combo':
            header = (
                "# Rules generated combinatorially from top operators. Rules are SYNTACTICALLY VALID (argument count checked).\n"
                "# WARNING: Generated rules often contain semantically useless sequences (e.g., redundant operations) "
                "and MUST be post-processed with external tools (like Hashcat's cleanup-rules.bin) for optimal performance.\n"
            )
        else:
              header = (
                  "# Rules generated via Markov Model Traversal (statistically probable chains).\n"
                  "# These are generally more effective than pure combinatorial rules. SYNTACTICALLY VALID (argument count checked).\n"
              )
    else:
        rules_to_save = sorted(list(rules_data))
        header = f"# Rules saved: {len(rules_to_save)} total.\n"
            
    print(f"\nSaving {len(rules_to_save)} rules to file '{filename}'...")
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(header)
        for rule in rules_to_save:
            f.write(f"{rule}\n")
            
    print(f"File '{filename}' saved.")
    sys.stdout.flush()

def run_and_rename_cleanup(input_file, command_binary, command_arg):
    RULE_TO_ADD = ":\n"
    temp_output_file = "temp_cleanup.rule"
    
    print(f"\n--- Starting External Cleanup Process ---")
    print(f"Running command: {command_binary} {command_arg} on {input_file}")

    try:
        with open(input_file, 'r') as infile:
            with open(temp_output_file, 'w') as outfile:
                subprocess.run(
                    [command_binary, command_arg],
                    stdin=infile,
                    stdout=outfile,
                    check=True  
                )
        print("Cleanup command finished successfully.")

    except FileNotFoundError:
        print(f"CLEANUP ERROR: Binary file not found: {command_binary}. Skipping cleanup.")
        return False
    except subprocess.CalledProcessError as e:
        print(f"CLEANUP ERROR: Command returned non-zero exit code: {e.returncode}. Skipping cleanup.")
        if os.path.exists(temp_output_file): os.remove(temp_output_file)
        return False
    except Exception as e:
        print(f"CLEANUP ERROR: An unknown error occurred: {e}. Skipping cleanup.")
        if os.path.exists(temp_output_file): os.remove(temp_output_file)
        return False
        
    try:
        print(f"Adding rule ':' to the start of {temp_output_file}...")
        with open(temp_output_file, 'r') as f:
            content = f.read()

        with open(temp_output_file, 'w') as f:
            f.write(RULE_TO_ADD)
            f.write(content)
        print("Rule added successfully.")

    except Exception as e:
        print(f"CLEANUP ERROR: Failed to modify file {temp_output_file}: {e}. Skipping rename.")
        return False
        
    line_count = 0
    try:
        with open(temp_output_file, 'r') as f:
            line_count = sum(1 for _ in f)
        print(f"Final rule count after cleanup (including ':' rule): {line_count} lines.")

    except Exception as e:
        print(f"CLEANUP ERROR: Failed to count lines: {e}. Skipping rename.")
        line_count = 'ERR'

    original_base = os.path.splitext(input_file)[0]
    new_filename = f"{original_base}_CLEANED_{command_arg}_{line_count}.rule"
    
    try:
        os.rename(temp_output_file, new_filename)
        print(f"Successfully renamed cleanup output to '{new_filename}'.")
        try:
            os.remove(input_file)
            print(f"Original file '{input_file}' deleted.")
        except OSError:
            print(f"Warning: Could not delete original file '{input_file}'.")

        return True
    except OSError as e:
        print(f"CLEANUP ERROR: Failed to rename file: {e}")
        return False
    
# --- Combinatorial Generation Functions (for completeness, remain the same) ---

def find_min_operators_for_target(sorted_operators, target_rules, min_len, max_len):
    """Finds the minimum number of top operators needed to generate the target number of rules."""
    current_rule_count = 0
    num_operators = 0
    while current_rule_count < target_rules and num_operators < len(sorted_operators):
        num_operators += 1
        top_ops = [op for op, count in sorted_operators[:num_operators]]
        current_rule_count = 0
        for length in range(min_len, max_len + 1):
            current_rule_count += (len(top_ops) ** length)
            
    return [op for op, count in sorted_operators[:num_operators]]

def generate_rules_for_length_validated(args):
    """Worker function to generate rules for a single length (L) with syntax validation."""
    top_operators, length, op_reqs, valid_chars = args
    generated_rules = set()
    
    for combo in itertools.product(top_operators, repeat=length):
        new_rule = ''.join(combo)
        
        if is_valid_hashcat_rule(new_rule, op_reqs, valid_chars):
            generated_rules.add(new_rule)
            
    return generated_rules

def generate_rules_parallel(top_operators, min_len, max_len):
    """
    Generates all VALID combinatorial rules in parallel based on a list of operators and a length range.
    """
    all_lengths = list(range(min_len, max_len + 1))
    
    tasks = [(top_operators, length, _OP_REQS, _VALID_CHARS) for length in all_lengths]
    
    num_processes = min(os.cpu_count() or 1, len(all_lengths))
    print(f"Generating new VALID rules of length {min_len} to {max_len} using {len(top_operators)} operators across {num_processes} processes...")
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(generate_rules_for_length_validated, tasks)
        
    generated_rules = set().union(*results)
    
    print(f"Generated and validated {len(generated_rules)} syntactically correct rules.")
    return generated_rules

# --- Main Execution Logic ---
if __name__ == '__main__':
    multiprocessing.freeze_support()  
    
    parser = argparse.ArgumentParser(description='Extracts top N rules sorted by raw frequency, statistical probability, or generates VALID combinatorial/Markov rules, with optional post-processing cleanup. Supports recursive folder search.')
    
    parser.add_argument('paths', nargs='+', help='Paths to rule files or directories to analyze. If a directory is provided, it will be searched recursively.')
    
    # Extraction Flags
    parser.add_argument('-t', '--top_rules', type=int, default=10000, help='The number of top existing rules to extract and save. ALSO controls the target number of Markov-generated rules.')
    parser.add_argument('-o', '--output_file', type=str, default='optimized_top_rules.txt', help='The name of the output file for extracted rules (also used as base for Markov output).')
    parser.add_argument('-m', '--max_length', type=int, default=31, help='The maximum length for rules to be extracted. Default is 31.')
    parser.add_argument('-s', '--statistical_sort', action='store_true', help='Sorts EXTRACTED rules by Markov sequence probability instead of raw frequency.')
    
    # Combinatorial Generation Flags
    parser.add_argument('-g', '--generate_combo', action='store_true', help='Enables generating a separate file with combinatorial rules from top operators.')
    parser.add_argument('-gc', '--combo_output_file', type=str, default='generated_combos_validated.txt', help='The name of the output file for generated combinatorial rules.')
    parser.add_argument('-n', '--combo_target', type=int, default=100000, help='The approximate number of rules to generate in combinatorial mode.')
    parser.add_argument('-l', '--combo_length', nargs='+', type=int, default=[1, 3], help='The range of rule chain lengths for combinatorial mode (e.g., 1 3).')
    
    # Statistical (Markov) Generation Flag (NEW)
    parser.add_argument('-gm', '--generate_markov_rules', action='store_true', help='Enables generating statistically probable rules by traversing the Markov model.')
    
    # NEW ARGUMENT FOR MARKOV LENGTH
    parser.add_argument('-ml', '--markov_length', nargs='+', type=int, default=None, help='The range of rule chain lengths for Markov mode (e.g., 1 5). Defaults to --combo_length if not set.')

    # Global/Utility Flags
    parser.add_argument('--temp-dir', type=str, default=None, help='Optional: Specify a directory for temporary files.')
    parser.add_argument('--in-memory', action='store_true', help='Process all rules entirely in RAM.')

    # Cleanup Flags
    parser.add_argument('-cb', '--cleanup-bin', type=str, default=None, 
                         help='Optional: Path to the external cleanup binary (e.g., ./cleanup-rules.bin). If provided, it will run after rule generation.')
    parser.add_argument('-ca', '--cleanup-arg', type=str, default='2', 
                         help='Argument to pass to the cleanup binary (e.g., "2" for hashcat\'s cleanup-rules.bin).')
    
    args = parser.parse_args()
    
    # --- RECURSIVE FILE COLLECTION LOGIC ---
    all_filepaths = []
    print("--- 0. Collecting Rule Files (Recursive Search) ---")
    
    # Need to derive the expected Markov output file name to exclude it from input analysis
    markov_base_name = os.path.splitext(args.output_file)[0]
    markov_ext = os.path.splitext(args.output_file)[1]
    markov_derived_filename = f"{markov_base_name}_markov{markov_ext if markov_ext else '.txt'}"
    
    output_files_to_exclude = [args.output_file, args.combo_output_file, markov_derived_filename]
    
    for path in args.paths:
        if os.path.isfile(path):
            if os.path.basename(path) not in output_files_to_exclude:
                all_filepaths.append(path)
        elif os.path.isdir(path):
            print(f"Searching directory: {path} recursively...")
            for root, _, files in os.walk(path):
                for file in files:
                    if file.lower().endswith(('.rule', '.txt', '.lst')) and \
                       file not in output_files_to_exclude:
                        all_filepaths.append(os.path.join(root, file))
        else:
            print(f"Warning: Path not found or not supported (must be file or directory): {path}")

    if not all_filepaths:
        print("Error: No rule files found to process. Exiting.")
        sys.exit(1)
        
    print(f"Found {len(all_filepaths)} rule files to analyze.")
    
    set_global_flags(args.temp_dir, args.in_memory)
    
    
    # --- LENGTH VALIDATION FOR GENERATION MODES ---
    combo_min_len, combo_max_len = None, None
    markov_min_len, markov_max_len = None, None

    # Combinatorial Length Setup
    if args.generate_combo or args.generate_markov_rules: # Need this if markov falls back to combo length
        length_source = args.combo_length
        if len(length_source) not in [1, 2]:
            print("Error: Invalid chain length range for combinatorial mode (--combo_length). Use 'L' or 'L_min L_max'.")
            sys.exit(1)
        combo_min_len = length_source[0]
        combo_max_len = length_source[-1]
        if combo_min_len > combo_max_len:
            print("Error: Invalid chain length range for combinatorial mode. L_min cannot be greater than L_max.")
            sys.exit(1)

    # Markov Length Setup (uses -ml, falls back to calculated combo length if -ml is not set)
    if args.generate_markov_rules:
        length_source = args.markov_length if args.markov_length is not None else args.combo_length
        
        # We need a valid length source, which should be guaranteed if generate_markov_rules is set.
        if len(length_source) not in [1, 2]:
            # This should ideally not happen due to the check above, but for robustness:
            print("Error: Invalid chain length range for Markov mode (derived from --markov_length or --combo_length). Use 'L' or 'L_min L_max'.")
            sys.exit(1)
            
        markov_min_len = length_source[0]
        markov_max_len = length_source[-1]
        
        if markov_min_len > markov_max_len:
            print("Error: Invalid chain length range for Markov mode. L_min cannot be greater than L_max.")
            sys.exit(1)
            
        source_flag = "(--markov_length)" if args.markov_length is not None else "(--combo_length fallback)"
        print(f"Markov Generation Length Range set to: {markov_min_len}-{markov_max_len} {source_flag}")


    # --- 1. Parallel Rule File Analysis ---
    print("--- 1. Starting Parallel Rule File Analysis ---")
    
    sorted_op_counts, full_rule_counts, all_clean_rules = analyze_rule_files_parallel(all_filepaths, args.max_length)
    
    if not sorted_op_counts:
        print("No operators found in files. Exiting.")
        sys.exit(1)

    # --- 2. Markov Model Building (Needed for both extraction and generation) ---
    markov_probabilities, total_transitions = get_markov_model(full_rule_counts)


    # --- 3. Determine Sorting Method and Extract Top N ---
    if args.statistical_sort:
        mode = 'statistical'
        print("\n--- Mode: Statistical Sort (Markov Weight) ---")
        sorted_rule_data = get_markov_weighted_rules(full_rule_counts, markov_probabilities, total_transitions)
    else:
        mode = 'frequency'
        print("\n--- Mode: Frequency Sort (Raw Count) ---")
        sorted_rule_data = sorted(full_rule_counts.items(), key=lambda item: item[1], reverse=True)
        
    top_rules_data = sorted_rule_data[:args.top_rules]

    # --- 4. Display analysis results and Save Extracted Rules ---
    print(f"\n--- 2. Analysis Results (Mode: {mode.upper()}) ---")
    print("Most frequently used operators (TOP 10):")
    for op, count in sorted_op_counts[:10]:
        print(f"  '{op}': {count} times")
        
    print(f"\nExtracted {len(top_rules_data)} top unique rules (max length: {args.max_length} characters).")
    output_file_name = args.output_file
    save_rules_to_file(top_rules_data, output_file_name, mode)
    
    if args.cleanup_bin:
        print("\n" + "~"*50)
        print("--- Running Cleanup on Extracted Rules ---")
        print("~"*50)
        run_and_rename_cleanup(output_file_name, args.cleanup_bin, args.cleanup_arg)
        
    # --- 5. STATISTICAL (MARKOV) RULE GENERATION ---
    if args.generate_markov_rules:
        print("\n" + "!"*50)
        print("--- 3. Starting STATISTICAL Markov Rule Generation (Validated) ---")
        print("!"*50)
        
        markov_base_name = os.path.splitext(args.output_file)[0]
        markov_ext = os.path.splitext(args.output_file)[1]
        markov_output_file_name = f"{markov_base_name}_markov{markov_ext if markov_ext else '.txt'}"
        
        # Use -t/--top_rules for Markov target
        markov_rules_data = generate_rules_from_markov_model(
            markov_probabilities, 
            args.top_rules, 
            markov_min_len, 
            markov_max_len
        )
        
        save_rules_to_file(markov_rules_data, markov_output_file_name, 'statistical')

        if args.cleanup_bin:
            print("\n" + "~"*50)
            print("--- Running Cleanup on Markov Generated Rules ---")
            print("~"*50)
            run_and_rename_cleanup(markov_output_file_name, args.cleanup_bin, args.cleanup_arg)

    # --- 6. COMBINATORIAL RULE GENERATION ---
    if args.generate_combo:
        print("\n" + "#"*50)
        print("--- 4. Starting COMBINATORIAL Rule Generation (Validated) ---")
        print("#"*50)
        
        # 6a. Find minimum number of top operators needed
        top_operators_needed = find_min_operators_for_target(
            sorted_op_counts, 
            args.combo_target, 
            combo_min_len, 
            combo_max_len
        )
        
        print(f"Using the top {len(top_operators_needed)} operators to approximate {args.combo_target} rules.")
        
        # 6b. Generate rules in parallel with syntax validation
        generated_rules_set = generate_rules_parallel(
            top_operators_needed, 
            combo_min_len, 
            combo_max_len
        )
        
        # 6c. Save the generated rules
        combo_output_file_name = args.combo_output_file
        save_rules_to_file(generated_rules_set, combo_output_file_name, 'combo')

        if args.cleanup_bin:
            print("\n" + "~"*50)
            print("--- Running Cleanup on Combinatorial Generated Rules ---")
            print("~"*50)
            run_and_rename_cleanup(combo_output_file_name, args.cleanup_bin, args.cleanup_arg)

    print("\n--- Processing Complete ---")
    sys.exit(0)
