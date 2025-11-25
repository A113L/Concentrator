# Concentrator v2.0 - GPU Accelerated
# Description: Unified Hashcat Rule Processor with OpenCL Acceleration, Validated Combinatorial Generation, and Statistical (Markov) Rule Generation.

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
import random
import numpy as np
import psutil

# Color codes for terminal output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

try:
    import pyopencl as cl
    OPENCL_AVAILABLE = True
except ImportError:
    OPENCL_AVAILABLE = False
    print(f"{Colors.YELLOW}Warning: PyOpenCL not available. Falling back to CPU mode.{Colors.END}")

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

# --- REGEX OPERATOR LIST BUILD ---
operators_to_escape = [op for op in ALL_OPERATORS if not (op.startswith('$') and len(op) > 1 and op[1].isdigit()) and not (op.startswith('^') and len(op) > 1 and op[1].isdigit())]

REGEX_OPERATORS = [re.escape(op) for op in operators_to_escape]
REGEX_OPERATORS.append(r'\$[0-9]') 
REGEX_OPERATORS.append(r'\^[0-9]') 

COMPILED_REGEX = re.compile('|'.join(filter(None, sorted(list(set(REGEX_OPERATORS)), key=len, reverse=True)))) 
# --- END REGEX OPERATOR LIST BUILD ---

_TEMP_DIR_PATH = None
_IN_MEMORY_MODE = False
_OP_REQS = OPERATORS_REQUIRING_ARGS
_VALID_CHARS = ALL_RULE_CHARS
_OPENCL_CONTEXT = None
_OPENCL_QUEUE = None
_OPENCL_PROGRAM = None

def print_banner():
    """Print the colorized program banner"""
    print(f"\n{Colors.CYAN}{Colors.BOLD}" + "="*70)
    print("          CONCENTRATOR v2.0 - GPU Accelerated Hashcat Rule Processor")
    print("="*70 + f"{Colors.END}")
    print(f"{Colors.YELLOW}Features:{Colors.END}")
    print(f"  {Colors.GREEN}•{Colors.END} OpenCL GPU Acceleration")
    print(f"  {Colors.GREEN}•{Colors.END} Validated Combinatorial Rule Generation") 
    print(f"  {Colors.GREEN}•{Colors.END} Statistical (Markov) Rule Generation")
    print(f"  {Colors.GREEN}•{Colors.END} Parallel File Processing")
    print(f"  {Colors.GREEN}•{Colors.END} Interactive & CLI Modes")
    print(f"{Colors.CYAN}{Colors.BOLD}" + "="*70 + f"{Colors.END}\n")

def print_usage():
    """Print colorized usage information"""
    print(f"{Colors.BOLD}{Colors.CYAN}USAGE:{Colors.END}")
    print(f"  {Colors.WHITE}python concentrator.py [OPTIONS] FILE_OR_DIRECTORY [FILE_OR_DIRECTORY...]{Colors.END}")
    print()
    print(f"{Colors.BOLD}{Colors.CYAN}MODES (choose one):{Colors.END}")
    print(f"  {Colors.GREEN}-e, --extract-rules{Colors.END}     Extract top existing rules from input files")
    print(f"  {Colors.GREEN}-g, --generate-combo{Colors.END}    Generate combinatorial rules from top operators") 
    print(f"  {Colors.GREEN}-gm, --generate-markov-rules{Colors.END} Generate statistically probable Markov rules")
    print()
    print(f"{Colors.BOLD}{Colors.CYAN}EXTRACTION MODE (-e):{Colors.END}")
    print(f"  {Colors.YELLOW}-t, --top-rules INT{Colors.END}     Number of top rules to extract (default: 10000)")
    print(f"  {Colors.YELLOW}-s, --statistical-sort{Colors.END}  Sort by statistical weight instead of frequency")
    print()
    print(f"{Colors.BOLD}{Colors.CYAN}COMBINATORIAL MODE (-g):{Colors.END}")
    print(f"  {Colors.YELLOW}-n, --combo-target INT{Colors.END}  Target number of rules (default: 100000)")
    print(f"  {Colors.YELLOW}-l, --combo-length MIN MAX{Colors.END} Rule length range (default: 1 3)")
    print()
    print(f"{Colors.BOLD}{Colors.CYAN}MARKOV MODE (-gm):{Colors.END}")
    print(f"  {Colors.YELLOW}-gt, --generate-target INT{Colors.END} Target rules (default: 10000)")
    print(f"  {Colors.YELLOW}-ml, --markov-length MIN MAX{Colors.END} Rule length range (default: 1 3)")
    print()
    print(f"{Colors.BOLD}{Colors.CYAN}GLOBAL OPTIONS:{Colors.END}")
    print(f"  {Colors.MAGENTA}-ob, --output-base-name NAME{Colors.END} Base name for output file")
    print(f"  {Colors.MAGENTA}-m, --max-length INT{Colors.END}    Maximum rule length to process (default: 31)")
    print(f"  {Colors.MAGENTA}--temp-dir DIR{Colors.END}        Temporary directory for file mode")
    print(f"  {Colors.MAGENTA}--in-memory{Colors.END}           Process entirely in RAM")
    print(f"  {Colors.MAGENTA}--no-gpu{Colors.END}             Disable GPU acceleration")
    print(f"  {Colors.MAGENTA}-cb, --cleanup-bin PATH{Colors.END} Path to cleanup binary")
    print(f"  {Colors.MAGENTA}-ca, --cleanup-arg ARG{Colors.END}  Cleanup argument (default: '2')")
    print()
    print(f"{Colors.BOLD}{Colors.CYAN}INTERACTIVE MODE:{Colors.END}")
    print(f"  {Colors.WHITE}python concentrator.py{Colors.END}   (run without arguments for interactive mode)")
    print()
    print(f"{Colors.BOLD}{Colors.CYAN}EXAMPLES:{Colors.END}")
    print(f"  {Colors.WHITE}# Extract top 5000 rules with GPU acceleration{Colors.END}")
    print(f"  {Colors.WHITE}python concentrator.py -e -t 5000 --no-gpu rules/*.rule{Colors.END}")
    print()
    print(f"  {Colors.WHITE}# Generate 50k combinatorial rules{Colors.END}")
    print(f"  {Colors.WHITE}python concentrator.py -g -n 50000 -l 2 4 hashcat/rules/{Colors.END}")
    print()
    print(f"  {Colors.WHITE}# Generate Markov rules with cleanup{Colors.END}")
    print(f"  {Colors.WHITE}python concentrator.py -gm -gt 10000 -cb ./cleanup-rules.bin rules/{Colors.END}")
    print()

# --- RAM Usage Monitoring ---
def check_ram_usage():
    """Check current RAM and swap usage and return status information"""
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    
    return {
        'ram_percent': memory.percent,
        'ram_used_gb': memory.used / (1024**3),
        'ram_total_gb': memory.total / (1024**3),
        'swap_percent': swap.percent,
        'swap_used_gb': swap.used / (1024**3),
        'swap_total_gb': swap.total / (1024**3),
        'using_swap': swap.used > 0
    }

def print_memory_status():
    """Print current memory status with colors"""
    mem_info = check_ram_usage()
    
    ram_color = Colors.GREEN
    if mem_info['ram_percent'] > 85:
        ram_color = Colors.RED
    elif mem_info['ram_percent'] > 70:
        ram_color = Colors.YELLOW
        
    print(f"{Colors.CYAN}Memory Status:{Colors.END} {ram_color}RAM {mem_info['ram_percent']:.1f}% ({mem_info['ram_used_gb']:.1f}/{mem_info['ram_total_gb']:.1f} GB){Colors.END}", end="")
    
    if mem_info['swap_total_gb'] > 0:
        if mem_info['using_swap']:
            swap_color = Colors.YELLOW if mem_info['swap_percent'] < 50 else Colors.RED
            print(f" | {Colors.CYAN}SWAP:{Colors.END} {swap_color}{mem_info['swap_percent']:.1f}% ({mem_info['swap_used_gb']:.1f}/{mem_info['swap_total_gb']:.1f} GB){Colors.END}")
        else:
            print(f" | {Colors.CYAN}Swap:{Colors.END} {Colors.GREEN}available ({mem_info['swap_total_gb']:.1f} GB){Colors.END}")
    else:
        print(f" | {Colors.CYAN}Swap:{Colors.END} {Colors.RED}not available{Colors.END}")

def memory_intensive_operation_warning(operation_name):
    """Warn user about memory-intensive operations and check if they want to continue"""
    mem_info = check_ram_usage()
    
    if mem_info['ram_percent'] > 85:
        print(f"{Colors.RED}{Colors.BOLD}WARNING:{Colors.END} {Colors.YELLOW}High RAM usage detected ({mem_info['ram_percent']:.1f}%) for {operation_name}{Colors.END}")
        print_memory_status()
        
        if mem_info['swap_total_gb'] == 0:
            print(f"{Colors.RED}CRITICAL: No swap space available. System may become unstable.{Colors.END}")
            response = input(f"{Colors.YELLOW}Continue with memory-intensive operation? (y/N): {Colors.END}").strip().lower()
            return response in ('y', 'yes')
        else:
            print(f"{Colors.YELLOW}System will use swap space. Performance may be slower.{Colors.END}")
            response = input(f"{Colors.YELLOW}Continue with memory-intensive operation? (Y/n): {Colors.END}").strip().lower()
            return response not in ('n', 'no')
    
    return True

# --- OpenCL Setup and Kernels ---
OPENCL_VALIDATION_KERNEL = """
// OpenCL Rule Validation Kernel

#define MAX_OPERATORS 256
#define MAX_RULE_LENGTH 64

// Lookup tables for operator validation
__constant uchar OPERATORS_REQUIRING_ARGS[256] = {
    ['s'] = 2, ['S'] = 2, ['t'] = 2, ['T'] = 2, ['i'] = 2, ['I'] = 2, 
    ['o'] = 2, ['O'] = 2, ['c'] = 2, ['C'] = 2, ['r'] = 2, ['R'] = 2, 
    ['y'] = 2, ['Y'] = 2, ['z'] = 2, ['Z'] = 2, ['e'] = 2, ['E'] = 2,
    ['f'] = 1, ['F'] = 1, ['x'] = 1, ['X'] = 1, ['d'] = 1, ['D'] = 1, 
    ['p'] = 1, ['P'] = 1, ['b'] = 1, ['B'] = 1, ['q'] = 1, ['Q'] = 1, 
    ['`'] = 1, ['['] = 1, [']'] = 1, ['>'] = 1, ['<'] = 1, ['@'] = 1, 
    ['&'] = 1, ['v'] = 3, ['V'] = 3
};

__constant uchar VALID_CHARS[256] = {
    ['0'] = 1, ['1'] = 1, ['2'] = 1, ['3'] = 1, ['4'] = 1, ['5'] = 1, 
    ['6'] = 1, ['7'] = 1, ['8'] = 1, ['9'] = 1, ['a'] = 1, ['b'] = 1, 
    ['c'] = 1, ['d'] = 1, ['e'] = 1, ['f'] = 1, ['g'] = 1, ['h'] = 1, 
    ['i'] = 1, ['j'] = 1, ['k'] = 1, ['l'] = 1, ['m'] = 1, ['n'] = 1, 
    ['o'] = 1, ['p'] = 1, ['q'] = 1, ['r'] = 1, ['s'] = 1, ['t'] = 1, 
    ['u'] = 1, ['v'] = 1, ['w'] = 1, ['x'] = 1, ['y'] = 1, ['z'] = 1,
    ['A'] = 1, ['B'] = 1, ['C'] = 1, ['D'] = 1, ['E'] = 1, ['F'] = 1, 
    ['G'] = 1, ['H'] = 1, ['I'] = 1, ['J'] = 1, ['K'] = 1, ['L'] = 1, 
    ['M'] = 1, ['N'] = 1, ['O'] = 1, ['P'] = 1, ['Q'] = 1, ['R'] = 1, 
    ['S'] = 1, ['T'] = 1, ['U'] = 1, ['V'] = 1, ['W'] = 1, ['X'] = 1, 
    ['Y'] = 1, ['Z'] = 1, [':'] = 1, [','] = 1, ['.'] = 1, ['l'] = 1, 
    ['u'] = 1, ['#'] = 1, ['('] = 1, [')'] = 1, ['='] = 1, ['%'] = 1, 
    ['!'] = 1, ['?'] = 1, ['|'] = 1, ['~'] = 1, ['+'] = 1, ['*'] = 1, 
    ['-'] = 1, ['^'] = 1, ['$'] = 1, ['s'] = 1, ['S'] = 1, ['t'] = 1, 
    ['T'] = 1, ['i'] = 1, ['I'] = 1, ['o'] = 1, ['O'] = 1, ['c'] = 1, 
    ['C'] = 1, ['r'] = 1, ['R'] = 1, ['y'] = 1, ['Y'] = 1, ['z'] = 1, 
    ['Z'] = 1, ['e'] = 1, ['E'] = 1, ['f'] = 1, ['F'] = 1, ['x'] = 1, 
    ['X'] = 1, ['d'] = 1, ['D'] = 1, ['p'] = 1, ['P'] = 1, ['b'] = 1, 
    ['B'] = 1, ['q'] = 1, ['Q'] = 1, ['`'] = 1, ['['] = 1, [']'] = 1, 
    ['>'] = 1, ['<'] = 1, ['@'] = 1, ['&'] = 1, ['v'] = 1, ['V'] = 1
};

__kernel void validate_rules_batch(
    __global const uchar* rules,
    __global uchar* results,
    const uint rule_stride,
    const uint max_rule_len,
    const uint num_rules)
{
    uint rule_idx = get_global_id(0);
    if (rule_idx >= num_rules) return;
    
    __global const uchar* rule = rules + rule_idx * rule_stride;
    bool valid = true;
    uint i = 0;
    
    while (i < max_rule_len && rule[i] != 0) {
        uchar op = rule[i];
        
        // Handle positional operators $X and ^X
        if ((op == '$' || op == '^') && i + 1 < max_rule_len) {
            uchar digit = rule[i + 1];
            if (digit >= '0' && digit <= '9') {
                i += 2;
                continue;
            }
        }
        
        // Check if operator requires arguments
        uchar required_args = OPERATORS_REQUIRING_ARGS[op];
        if (required_args > 0) {
            if (i + 1 + required_args > max_rule_len) {
                valid = false;
                break;
            }
            
            // Validate arguments
            for (uchar arg = 1; arg <= required_args; arg++) {
                uchar arg_char = rule[i + arg];
                if (VALID_CHARS[arg_char] == 0) {
                    valid = false;
                    break;
                }
            }
            
            if (!valid) break;
            i += 1 + required_args;
        } 
        else if (VALID_CHARS[op] == 1) {
            i += 1;  // Valid simple operator or argument
        }
        else {
            valid = false;
            break;
        }
    }
    
    results[rule_idx] = valid ? 1 : 0;
}

// Enhanced GPU validation with formatted output tracking
__kernel void validate_and_format_rules(
    __global const uchar* rules,
    __global uchar* results,
    __global uchar* formatted_output,
    const uint rule_stride,
    const uint max_rule_len,
    const uint num_rules,
    const uint output_stride)
{
    uint rule_idx = get_global_id(0);
    if (rule_idx >= num_rules) return;
    
    __global const uchar* rule = rules + rule_idx * rule_stride;
    __global uchar* output = formatted_output + rule_idx * output_stride;
    
    bool valid = true;
    uint i = 0;
    uint out_pos = 0;
    
    while (i < max_rule_len && rule[i] != 0) {
        uchar op = rule[i];
        
        // Handle positional operators $X and ^X
        if ((op == '$' || op == '^') && i + 1 < max_rule_len) {
            uchar digit = rule[i + 1];
            if (digit >= '0' && digit <= '9') {
                // Copy both characters for positional operators
                output[out_pos++] = op;
                output[out_pos++] = digit;
                i += 2;
                continue;
            }
        }
        
        // Check if operator requires arguments
        uchar required_args = OPERATORS_REQUIRING_ARGS[op];
        if (required_args > 0) {
            if (i + 1 + required_args > max_rule_len) {
                valid = false;
                break;
            }
            
            // Copy operator
            output[out_pos++] = op;
            
            // Validate and copy arguments
            for (uchar arg = 1; arg <= required_args; arg++) {
                uchar arg_char = rule[i + arg];
                if (VALID_CHARS[arg_char] == 0) {
                    valid = false;
                    break;
                }
                output[out_pos++] = arg_char;
            }
            
            if (!valid) break;
            i += 1 + required_args;
        } 
        else if (VALID_CHARS[op] == 1) {
            // Simple operator - just copy
            output[out_pos++] = op;
            i += 1;
        }
        else {
            valid = false;
            break;
        }
    }
    
    // Null terminate the formatted output
    if (out_pos < output_stride) {
        output[out_pos] = 0;
    }
    
    results[rule_idx] = valid ? 1 : 0;
}

// Combinatorial rule generation kernel
__kernel void generate_combinatorial_rules(
    __global const uchar* operators,
    const uint num_operators,
    const uint min_len,
    const uint max_len,
    __global uchar* output_rules,
    __global uchar* valid_flags,
    const uint output_stride)
{
    uint global_id = get_global_id(0);
    
    // Calculate total combinations up to max_len-1
    uint total_prev = 0;
    uint rules_per_len[MAX_RULE_LENGTH];
    uint current_len = min_len;
    
    for (uint len = min_len; len <= max_len; len++) {
        uint count = 1;
        for (uint i = 0; i < len; i++) {
            count *= num_operators;
        }
        rules_per_len[len - min_len] = count;
        if (len < max_len) {
            total_prev += count;
        }
    }
    
    // Find which length this thread should process
    uint rules_so_far = 0;
    uint target_len = min_len;
    uint comb_index = global_id;
    
    for (uint len = min_len; len <= max_len; len++) {
        uint rules_this_len = rules_per_len[len - min_len];
        if (comb_index < rules_so_far + rules_this_len) {
            target_len = len;
            comb_index = comb_index - rules_so_far;
            break;
        }
        rules_so_far += rules_this_len;
    }
    
    if (target_len > max_len) {
        valid_flags[global_id] = 0;
        return;
    }
    
    __global uchar* output_rule = output_rules + global_id * output_stride;
    
    // Generate the specific combination
    uint temp_index = comb_index;
    for (uint pos = 0; pos < target_len; pos++) {
        uint op_index = temp_index % num_operators;
        output_rule[pos] = operators[op_index];
        temp_index /= num_operators;
    }
    
    // Null terminate
    if (target_len < output_stride) {
        output_rule[target_len] = 0;
    }
    
    // Validate the generated rule
    bool valid = true;
    uint i = 0;
    
    while (i < target_len && output_rule[i] != 0) {
        uchar op = output_rule[i];
        
        // Handle positional operators $X and ^X
        if ((op == '$' || op == '^') && i + 1 < target_len) {
            uchar digit = output_rule[i + 1];
            if (digit >= '0' && digit <= '9') {
                i += 2;
                continue;
            }
        }
        
        uchar required_args = OPERATORS_REQUIRING_ARGS[op];
        if (required_args > 0) {
            if (i + 1 + required_args > target_len) {
                valid = false;
                break;
            }
            
            for (uchar arg = 1; arg <= required_args; arg++) {
                uchar arg_char = output_rule[i + arg];
                if (VALID_CHARS[arg_char] == 0) {
                    valid = false;
                    break;
                }
            }
            
            if (!valid) break;
            i += 1 + required_args;
        } 
        else if (VALID_CHARS[op] == 1) {
            i += 1;
        }
        else {
            valid = false;
            break;
        }
    }
    
    valid_flags[global_id] = valid ? 1 : 0;
}
"""

OPENCL_RULE_PROCESSING_KERNEL = """
// Helper function to convert char digit/letter to int position
unsigned int char_to_pos(unsigned char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'Z') return c - 'A' + 10;
    if (c >= 'a' && c <= 'z') return c - 'a' + 10;
    // Return a value guaranteed to fail bounds checks
    return 0xFFFFFFFF; 
}

// Helper function to get rule length
unsigned int rule_len(__global const unsigned char* rule_ptr, unsigned int max_rule_len) {
    for (unsigned int i = 0; i < max_rule_len; i++) {
        if (rule_ptr[i] == 0) return i;
    }
    return max_rule_len;
}

// Parse a complete rule sequence and apply transformations
void apply_rule_sequence(
    __global const unsigned char* word_ptr,
    unsigned int word_len,
    __global const unsigned char* rule_ptr,
    unsigned int rule_length,
    __global unsigned char* result_ptr,
    unsigned int max_output_len)
{
    // Initialize result with original word
    for(unsigned int i = 0; i < word_len; i++) {
        result_ptr[i] = word_ptr[i];
    }
    unsigned int current_len = word_len;
    
    unsigned int rule_pos = 0;
    while (rule_pos < rule_length) {
        unsigned char op = rule_ptr[rule_pos];
        
        // Handle positional operators $X and ^X
        if ((op == '$' || op == '^') && rule_pos + 1 < rule_length) {
            unsigned char digit = rule_ptr[rule_pos + 1];
            if (digit >= '0' && digit <= '9') {
                unsigned int pos = char_to_pos(digit);
                
                if (op == '$') {
                    // Append character at position (Hashcat $X)
                    if (pos < current_len && current_len + 1 < max_output_len) {
                        result_ptr[current_len] = result_ptr[pos];
                        current_len++;
                    }
                } else { // op == '^'
                    // Prepend character at position (Hashcat ^X)  
                    if (pos < current_len && current_len + 1 < max_output_len) {
                        // Shift right
                        for (int i = current_len; i > 0; i--) {
                            result_ptr[i] = result_ptr[i - 1];
                        }
                        result_ptr[0] = result_ptr[pos + 1]; // +1 because we shifted
                        current_len++;
                    }
                }
                rule_pos += 2;
                continue;
            }
        }
        
        // Handle operators with arguments
        if (op == 'i' || op == 'I' || op == 'o' || op == 'O' || 
            op == 's' || op == 'S' || op == 't' || op == 'T' ||
            op == 'c' || op == 'C' || op == 'r' || op == 'R' ||
            op == 'y' || op == 'Y' || op == 'z' || op == 'Z' ||
            op == 'e' || op == 'E') {
            
            if (rule_pos + 2 >= rule_length) break;
            
            unsigned char arg1 = rule_ptr[rule_pos + 1];
            unsigned char arg2 = rule_ptr[rule_pos + 2];
            
            // Insert character (iXy)
            if (op == 'i') {
                unsigned int pos = char_to_pos(arg1);
                if (pos <= current_len && current_len + 1 < max_output_len) {
                    // Shift right from position
                    for (int i = current_len; i > pos; i--) {
                        result_ptr[i] = result_ptr[i - 1];
                    }
                    result_ptr[pos] = arg2;
                    current_len++;
                }
            }
            // Overwrite character (oXy)
            else if (op == 'o') {
                unsigned int pos = char_to_pos(arg1);
                if (pos < current_len) {
                    result_ptr[pos] = arg2;
                }
            }
            // Swap characters (sXy)
            else if (op == 's') {
                // Find all occurrences of arg1 and replace with arg2
                for (unsigned int i = 0; i < current_len; i++) {
                    if (result_ptr[i] == arg1) {
                        result_ptr[i] = arg2;
                    }
                }
            }
            // Swap case (t, T, c, C)
            else if (op == 't') {
                // Toggle case of all characters
                for (unsigned int i = 0; i < current_len; i++) {
                    unsigned char c = result_ptr[i];
                    if (c >= 'a' && c <= 'z') {
                        result_ptr[i] = c - 32;
                    } else if (c >= 'A' && c <= 'Z') {
                        result_ptr[i] = c + 32;
                    }
                }
            }
            else if (op == 'T') {
                // Toggle case at specific position
                unsigned int pos = char_to_pos(arg1);
                if (pos < current_len) {
                    unsigned char c = result_ptr[pos];
                    if (c >= 'a' && c <= 'z') {
                        result_ptr[pos] = c - 32;
                    } else if (c >= 'A' && c <= 'Z') {
                        result_ptr[pos] = c + 32;
                    }
                }
            }
            // Capitalize (c) / lowercase (C)
            else if (op == 'c') {
                // Capitalize first character
                if (current_len > 0 && result_ptr[0] >= 'a' && result_ptr[0] <= 'z') {
                    result_ptr[0] = result_ptr[0] - 32;
                }
            }
            else if (op == 'C') {
                // Lowercase first character  
                if (current_len > 0 && result_ptr[0] >= 'A' && result_ptr[0] <= 'Z') {
                    result_ptr[0] = result_ptr[0] + 32;
                }
            }
            
            rule_pos += 3;
        }
        // Handle single character operators
        else {
            switch (op) {
                case 'l': // Lowercase all
                    for (unsigned int i = 0; i < current_len; i++) {
                        unsigned char c = result_ptr[i];
                        if (c >= 'A' && c <= 'Z') {
                            result_ptr[i] = c + 32;
                        }
                    }
                    break;
                    
                case 'u': // Uppercase all
                    for (unsigned int i = 0; i < current_len; i++) {
                        unsigned char c = result_ptr[i];
                        if (c >= 'a' && c <= 'z') {
                            result_ptr[i] = c - 32;
                        }
                    }
                    break;
                    
                case 'd': // Duplicate word
                    if (current_len * 2 < max_output_len) {
                        for (unsigned int i = 0; i < current_len; i++) {
                            result_ptr[current_len + i] = result_ptr[i];
                        }
                        current_len *= 2;
                    }
                    break;
                    
                case 'f': // Duplicate and reverse
                    if (current_len * 2 < max_output_len) {
                        for (unsigned int i = 0; i < current_len; i++) {
                            result_ptr[current_len + i] = result_ptr[current_len - 1 - i];
                        }
                        current_len *= 2;
                    }
                    break;
                    
                case 'r': // Reverse entire word
                    for (unsigned int i = 0; i < current_len / 2; i++) {
                        unsigned char temp = result_ptr[i];
                        result_ptr[i] = result_ptr[current_len - 1 - i];
                        result_ptr[current_len - 1 - i] = temp;
                    }
                    break;
                    
                case ':': // Do nothing (no-op)
                    break;
            }
            rule_pos++;
        }
    }
    
    // Null terminate
    if (current_len < max_output_len) {
        result_ptr[current_len] = 0;
    }
}

__kernel void hashcat_rules_kernel(
    __global const unsigned char* base_words_in,
    __global const unsigned char* rules_in,
    __global unsigned char* result_buffer,
    const unsigned int num_words,
    const unsigned int num_rules,
    const unsigned int max_word_len,
    const unsigned int max_rule_len,
    const unsigned int max_output_len)
{
    unsigned int global_id = get_global_id(0);
    unsigned int word_idx = global_id / num_rules;
    unsigned int rule_idx = global_id % num_rules;

    if (word_idx >= num_words || rule_idx >= num_rules) return;

    // Get word
    __global const unsigned char* word_ptr = base_words_in + word_idx * max_word_len;
    unsigned int word_len = 0;
    for (unsigned int i = 0; i < max_word_len; i++) {
        if (word_ptr[i] == 0) {
            word_len = i;
            break;
        }
    }
    if (word_len == 0) word_len = max_word_len;
    
    // Get rule
    __global const unsigned char* rule_ptr = rules_in + rule_idx * max_rule_len;
    unsigned int rule_length = rule_len(rule_ptr, max_rule_len);
    
    // Get result buffer
    __global unsigned char* result_ptr = result_buffer + global_id * max_output_len;
    
    // Apply the rule sequence
    apply_rule_sequence(word_ptr, word_len, rule_ptr, rule_length, result_ptr, max_output_len);
}
"""

def setup_opencl():
    """Initialize OpenCL context and compile kernels"""
    global _OPENCL_CONTEXT, _OPENCL_QUEUE, _OPENCL_PROGRAM
    
    if not OPENCL_AVAILABLE:
        return False
        
    try:
        platforms = cl.get_platforms()
        if not platforms:
            print(f"{Colors.YELLOW}No OpenCL platforms found{Colors.END}")
            return False
            
        devices = platforms[0].get_devices(cl.device_type.GPU)
        if not devices:
            print(f"{Colors.YELLOW}No GPU devices found, trying CPU{Colors.END}")
            devices = platforms[0].get_devices(cl.device_type.CPU)
            
        if not devices:
            print(f"{Colors.YELLOW}No OpenCL devices found{Colors.END}")
            return False
            
        _OPENCL_CONTEXT = cl.Context(devices)
        _OPENCL_QUEUE = cl.CommandQueue(_OPENCL_CONTEXT)
        _OPENCL_PROGRAM = cl.Program(_OPENCL_CONTEXT, OPENCL_VALIDATION_KERNEL).build()
        
        print(f"{Colors.GREEN}OpenCL initialized on: {devices[0].name}{Colors.END}")
        return True
        
    except Exception as e:
        print(f"{Colors.RED}OpenCL initialization failed: {e}{Colors.END}")
        return False

def gpu_validate_rules(rules_list, max_rule_length=64):
    """Validate thousands of rules in parallel on GPU"""
    if not _OPENCL_CONTEXT or not rules_list:
        return [False] * len(rules_list)
    
    try:
        # Prepare data
        num_rules = len(rules_list)
        rule_stride = ((max_rule_length + 15) // 16) * 16  # 16-byte alignment
        
        rules_buffer = np.zeros((num_rules, rule_stride), dtype=np.uint8)
        
        # Fill buffer with rules
        for i, rule in enumerate(rules_list):
            rule_bytes = rule.encode('ascii', 'ignore')
            length = min(len(rule_bytes), rule_stride)
            rules_buffer[i, :length] = np.frombuffer(rule_bytes[:length], dtype=np.uint8)
        
        results = np.zeros(num_rules, dtype=np.uint8)
        
        # Create GPU buffers
        mf = cl.mem_flags
        rules_gpu = cl.Buffer(_OPENCL_CONTEXT, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=rules_buffer)
        results_gpu = cl.Buffer(_OPENCL_CONTEXT, mf.WRITE_ONLY, results.nbytes)
        
        # Execute kernel
        global_size = (num_rules,)
        _OPENCL_PROGRAM.validate_rules_batch(_OPENCL_QUEUE, global_size, None,
                                           rules_gpu, results_gpu,
                                           np.uint32(rule_stride),
                                           np.uint32(max_rule_length),
                                           np.uint32(num_rules))
        
        # Get results
        cl.enqueue_copy(_OPENCL_QUEUE, results, results_gpu)
        _OPENCL_QUEUE.finish()
        
        return [bool(result) for result in results]
        
    except Exception as e:
        print(f"{Colors.RED}GPU validation failed: {e}, falling back to CPU{Colors.END}")
        return [is_valid_hashcat_rule(rule, _OP_REQS, _VALID_CHARS) for rule in rules_list]

def gpu_validate_and_format_rules(rules_list, max_rule_length=64):
    """Validate rules on GPU and return formatted valid rules"""
    if not _OPENCL_CONTEXT or not rules_list:
        return []
    
    try:
        # Prepare data
        num_rules = len(rules_list)
        rule_stride = ((max_rule_length + 15) // 16) * 16
        output_stride = max_rule_length + 1
        
        rules_buffer = np.zeros((num_rules, rule_stride), dtype=np.uint8)
        formatted_output = np.zeros((num_rules, output_stride), dtype=np.uint8)
        results = np.zeros(num_rules, dtype=np.uint8)
        
        # Fill buffer with rules
        for i, rule in enumerate(rules_list):
            rule_bytes = rule.encode('ascii', 'ignore')
            length = min(len(rule_bytes), rule_stride)
            rules_buffer[i, :length] = np.frombuffer(rule_bytes[:length], dtype=np.uint8)
        
        # Create GPU buffers
        mf = cl.mem_flags
        rules_gpu = cl.Buffer(_OPENCL_CONTEXT, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=rules_buffer)
        results_gpu = cl.Buffer(_OPENCL_CONTEXT, mf.WRITE_ONLY, results.nbytes)
        formatted_gpu = cl.Buffer(_OPENCL_CONTEXT, mf.WRITE_ONLY, formatted_output.nbytes)
        
        # Execute enhanced validation kernel
        global_size = (num_rules,)
        _OPENCL_PROGRAM.validate_and_format_rules(_OPENCL_QUEUE, global_size, None,
                                                rules_gpu, results_gpu, formatted_gpu,
                                                np.uint32(rule_stride),
                                                np.uint32(max_rule_length),
                                                np.uint32(num_rules),
                                                np.uint32(output_stride))
        
        # Get results
        cl.enqueue_copy(_OPENCL_QUEUE, results, results_gpu)
        cl.enqueue_copy(_OPENCL_QUEUE, formatted_output, formatted_gpu)
        _OPENCL_QUEUE.finish()
        
        # Extract valid formatted rules
        valid_rules = []
        for i in range(num_rules):
            if results[i]:
                # Extract the formatted rule
                rule_bytes = formatted_output[i]
                rule_length = 0
                for j, byte in enumerate(rule_bytes):
                    if byte == 0:
                        rule_length = j
                        break
                if rule_length > 0:
                    rule = ''.join(chr(b) for b in rule_bytes[:rule_length])
                    valid_rules.append(rule)
        
        return valid_rules
        
    except Exception as e:
        print(f"{Colors.RED}GPU validation with formatting failed: {e}, falling back to CPU{Colors.END}")
        return [rule for rule in rules_list if is_valid_hashcat_rule(rule, _OP_REQS, _VALID_CHARS)]

def gpu_generate_combinatorial_rules(top_operators, min_len, max_len, max_rule_length=64):
    """Generate combinatorial rules on GPU with built-in validation"""
    if not _OPENCL_CONTEXT:
        return set()
    
    try:
        # Filter out multi-character operators for GPU generation
        single_char_operators = [op for op in top_operators if len(op) == 1]
        
        if not single_char_operators:
            print(f"{Colors.YELLOW}No single-character operators available for GPU generation, falling back to CPU{Colors.END}")
            return generate_rules_parallel(top_operators, min_len, max_len)
            
        print(f"{Colors.CYAN}GPU using {len(single_char_operators)} single-character operators (filtered from {len(top_operators)} total){Colors.END}")
        
        # Calculate total combinations
        total_combs = sum(len(single_char_operators) ** l for l in range(min_len, max_len + 1))
        if total_combs == 0:
            return set()
            
        print(f"{Colors.CYAN}GPU generating {total_combs} combinatorial rules...{Colors.END}")
        
        # Prepare operators array - convert single characters to ASCII codes
        ops_array = np.array([ord(op) for op in single_char_operators], dtype=np.uint8)
        num_operators = len(single_char_operators)
        
        # Output buffers
        rule_stride = max_rule_length + 1  # +1 for null terminator
        output_rules = np.zeros(total_combs * rule_stride, dtype=np.uint8)
        valid_flags = np.zeros(total_combs, dtype=np.uint8)
        
        # Create GPU buffers
        mf = cl.mem_flags
        ops_gpu = cl.Buffer(_OPENCL_CONTEXT, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ops_array)
        output_gpu = cl.Buffer(_OPENCL_CONTEXT, mf.WRITE_ONLY, output_rules.nbytes)
        valid_gpu = cl.Buffer(_OPENCL_CONTEXT, mf.WRITE_ONLY, valid_flags.nbytes)
        
        # Execute kernel
        global_size = (total_combs,)
        _OPENCL_PROGRAM.generate_combinatorial_rules(_OPENCL_QUEUE, global_size, None,
                                                   ops_gpu, np.uint32(num_operators),
                                                   np.uint32(min_len), np.uint32(max_len),
                                                   output_gpu, valid_gpu, np.uint32(rule_stride))
        
        # Get results
        cl.enqueue_copy(_OPENCL_QUEUE, output_rules, output_gpu)
        cl.enqueue_copy(_OPENCL_QUEUE, valid_flags, valid_gpu)
        _OPENCL_QUEUE.finish()
        
        # Extract valid rules
        valid_rules = set()
        for i in range(total_combs):
            if valid_flags[i]:
                start = i * rule_stride
                end = start + rule_stride
                rule_bytes = output_rules[start:end]
                
                # Find null terminator
                rule_length = 0
                for j, byte in enumerate(rule_bytes):
                    if byte == 0:
                        rule_length = j
                        break
                
                if rule_length > 0:
                    rule = ''.join(chr(b) for b in rule_bytes[:rule_length])
                    valid_rules.add(rule)
        
        print(f"{Colors.GREEN}GPU generated {len(valid_rules)} valid rules{Colors.END}")
        return valid_rules
        
    except Exception as e:
        print(f"{Colors.RED}GPU combinatorial generation failed: {e}, falling back to CPU{Colors.END}")
        return generate_rules_parallel(top_operators, min_len, max_len)

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
                print(f"{Colors.CYAN}Using temporary directory: {_TEMP_DIR_PATH}{Colors.END}")
            except OSError as e:
                print(f"{Colors.YELLOW}Warning: Could not create temporary directory at {temp_dir_path}. Falling back to default system temp directory. Error: {e}{Colors.END}", flush=True)
                _TEMP_DIR_PATH = None
    elif in_memory_mode:
          print(f"{Colors.CYAN}In-Memory Mode activated. Temporary files will be skipped.{Colors.END}")

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

# --- Recursive File Search with Depth Limit ---
def find_rule_files_recursive(paths, max_depth=3):
    """Find rule files recursively with depth limit"""
    all_filepaths = []
    
    for path in paths:
        if os.path.isfile(path):
            if path.lower().endswith(('.rule', '.txt', '.lst')):
                all_filepaths.append(path)
        elif os.path.isdir(path):
            print(f"{Colors.CYAN}Searching directory: {path} (max depth: {max_depth})...{Colors.END}")
            for root, dirs, files in os.walk(path):
                # Calculate current depth
                current_depth = root[len(path):].count(os.sep)
                if current_depth >= max_depth:
                    # Don't go deeper into subdirectories at max depth
                    dirs.clear()
                    continue
                    
                for file in files:
                    if file.lower().endswith(('.rule', '.txt', '.lst')):
                        full_path = os.path.join(root, file)
                        all_filepaths.append(full_path)
    
    return all_filepaths

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
            print(f"{Colors.GREEN}File analysis complete: {filepath}. Temp rules saved to {temp_rule_filepath}{Colors.END}", flush=True)
            return operator_counts, full_rule_counts, [], temp_rule_filepath
        else:
            # --- IN-MEMORY MODE: Return the list of rules directly ---
            print(f"{Colors.GREEN}File analysis complete: {filepath}. Rules returned in memory.{Colors.END}", flush=True)
            return operator_counts, full_rule_counts, clean_rules_list, None
            
    except Exception as e:
        print(f"{Colors.RED}An error occurred while processing {filepath}: {e}{Colors.END}", flush=True)
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
        print(f"{Colors.YELLOW}Warning: No valid rule files found to process.{Colors.END}")
        return defaultdict(int), defaultdict(int), [], None

    num_processes = min(os.cpu_count() or 1, len(existing_filepaths))
    tasks = [(filepath, max_rule_length) for filepath in existing_filepaths]
    
    print(f"{Colors.CYAN}Starting parallel analysis of {len(existing_filepaths)} files using {num_processes} processes...{Colors.END}")
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
        print(f"\n{Colors.CYAN}Merging temporary rule files into memory for Markov processing...{Colors.END}")
        for temp_filepath in temp_files_to_merge:
            try:
                with open(temp_filepath, 'r', encoding='utf-8') as f:
                    total_all_clean_rules.extend([line.strip() for line in f])
                os.unlink(temp_filepath)
            except Exception as e:
                print(f"{Colors.RED}Error merging temp file {temp_filepath}: {e}{Colors.END}")
            
    print(f"{Colors.GREEN}Total unique rules loaded into memory: {len(total_full_rule_counts)}{Colors.END}")
    
    sorted_op_counts = sorted(total_operator_counts.items(), key=lambda item: item[1], reverse=True)
    return sorted_op_counts, total_full_rule_counts, total_all_clean_rules

# --- Markov and Extraction Functions ---
def get_markov_model(unique_rules):
    """Builds the Markov model (counts) and transition probabilities."""
    if not memory_intensive_operation_warning("Markov model building"):
        return None, None
        
    print(f"\n{Colors.CYAN}--- Building Markov Sequence Probability Model ---{Colors.END}")
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
    if not memory_intensive_operation_warning("Markov weighting"):
        return []
        
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
            continue 
            
        # P(Oi | O_i-1) or P(Oi | O_i-2 O_i-1)
        for i in range(len(rule) - 1):
            # Try Trigram (O_i-1 O_i -> O_i+1) first
            if i >= 1:
                current_prefix = rule[i-1:i+1] # Trigram prefix
                next_char = rule[i+1]
                if current_prefix in markov_probabilities and next_char in markov_probabilities[current_prefix]:
                    probability = markov_probabilities[current_prefix][next_char]
                    log_probability_sum += math.log(probability)
                    continue 
            
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

# --- Markov Rule Generation Logic ---
def generate_rules_from_markov_model(markov_probabilities, target_rules, min_len, max_len):
    """
    Generates new rules by traversing the Markov model, prioritizing high-probability transitions.
    The generation stops attempting new rules once 'target_rules' unique, valid rules are found.
    """
    if not memory_intensive_operation_warning("Markov rule generation"):
        return []
        
    print(f"\n{Colors.CYAN}--- Generating Rules via Markov Model Traversal ({min_len}-{max_len} Operators, Target: {target_rules}) ---{Colors.END}")
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
        # Stop once the target number of unique rules is reached
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

    print(f"{Colors.GREEN}Generated {len(generated_rules)} statistically probable and syntactically valid rules (Target: {target_rules}).{Colors.END}")
    
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
            
    print(f"\n{Colors.CYAN}Saving {len(rules_to_save)} rules to file '{filename}'...{Colors.END}")
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(header)
        for rule in rules_to_save:
            f.write(f"{rule}\n")
            
    print(f"{Colors.GREEN}File '{filename}' saved.{Colors.END}")
    sys.stdout.flush()

def run_and_rename_cleanup(input_file, command_binary, command_arg):
    RULE_TO_ADD = ":\n"
    temp_output_file = "temp_cleanup.rule"
    
    print(f"\n{Colors.CYAN}--- Starting External Cleanup Process ---{Colors.END}")
    print(f"{Colors.CYAN}Running command: {command_binary} {command_arg} on {input_file}{Colors.END}")

    try:
        with open(input_file, 'r') as infile:
            with open(temp_output_file, 'w') as outfile:
                subprocess.run(
                    [command_binary, command_arg],
                    stdin=infile,
                    stdout=outfile,
                    check=True  
                )
        print(f"{Colors.GREEN}Cleanup command finished successfully.{Colors.END}")

    except FileNotFoundError:
        print(f"{Colors.RED}CLEANUP ERROR: Binary file not found: {command_binary}. Skipping cleanup.{Colors.END}")
        return False
    except subprocess.CalledProcessError as e:
        print(f"{Colors.RED}CLEANUP ERROR: Command returned non-zero exit code: {e.returncode}. Skipping cleanup.{Colors.END}")
        if os.path.exists(temp_output_file): os.remove(temp_output_file)
        return False
    except Exception as e:
        print(f"{Colors.RED}CLEANUP ERROR: An unknown error occurred: {e}. Skipping cleanup.{Colors.END}")
        if os.path.exists(temp_output_file): os.remove(temp_output_file)
        return False
        
    try:
        print(f"{Colors.CYAN}Adding rule ':' to the start of {temp_output_file}...{Colors.END}")
        with open(temp_output_file, 'r') as f:
            content = f.read()

        with open(temp_output_file, 'w') as f:
            f.write(RULE_TO_ADD)
            f.write(content)
        print(f"{Colors.GREEN}Rule added successfully.{Colors.END}")

    except Exception as e:
        print(f"{Colors.RED}CLEANUP ERROR: Failed to modify file {temp_output_file}: {e}. Skipping rename.{Colors.END}")
        return False
        
    line_count = 0
    try:
        with open(temp_output_file, 'r') as f:
            line_count = sum(1 for _ in f)
        print(f"{Colors.CYAN}Final rule count after cleanup (including ':' rule): {line_count} lines.{Colors.END}")

    except Exception as e:
        print(f"{Colors.RED}CLEANUP ERROR: Failed to count lines: {e}. Skipping rename.{Colors.END}")
        line_count = 'ERR'

    original_base = os.path.splitext(input_file)[0]
    new_filename = f"{original_base}_CLEANED_{command_arg}_{line_count}.rule"
    
    try:
        os.rename(temp_output_file, new_filename)
        print(f"{Colors.GREEN}Successfully renamed cleanup output to '{new_filename}'.{Colors.END}")
        try:
            os.remove(input_file)
            print(f"{Colors.CYAN}Original file '{input_file}' deleted.{Colors.END}")
        except OSError:
            print(f"{Colors.YELLOW}Warning: Could not delete original file '{input_file}'.{Colors.END}")

        return True
    except OSError as e:
        print(f"{Colors.RED}CLEANUP ERROR: Failed to rename file: {e}{Colors.END}")
        return False
    
# --- Combinatorial Generation Functions ---

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
    if not memory_intensive_operation_warning("combinatorial generation"):
        return set()
        
    all_lengths = list(range(min_len, max_len + 1))
    
    tasks = [(top_operators, length, _OP_REQS, _VALID_CHARS) for length in all_lengths]
    
    num_processes = min(os.cpu_count() or 1, len(all_lengths))
    print(f"{Colors.CYAN}Generating new VALID rules of length {min_len} to {max_len} using {len(top_operators)} operators across {num_processes} processes...{Colors.END}")
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(generate_rules_for_length_validated, tasks)
        
    generated_rules = set().union(*results)
    
    print(f"{Colors.GREEN}Generated and validated {len(generated_rules)} syntactically correct rules.{Colors.END}")
    return generated_rules

# --- Enhanced GPU Extraction Mode ---
def gpu_extract_and_validate_rules(full_rule_counts, top_rules, gpu_enabled):
    """Extract and validate rules using GPU acceleration"""
    print(f"\n{Colors.CYAN}Extracting top {top_rules} rules with GPU validation...{Colors.END}")
    
    # Get all rules sorted by frequency
    all_rules_sorted = sorted(full_rule_counts.items(), key=lambda item: item[1], reverse=True)
    
    if gpu_enabled:
        # Use GPU for validation and formatting
        rules_to_validate = [rule for rule, count in all_rules_sorted[:top_rules*2]]  # Validate more than needed
        validated_rules = gpu_validate_and_format_rules(rules_to_validate)
        
        # Re-sort validated rules by frequency
        validated_with_counts = []
        for rule in validated_rules:
            if rule in full_rule_counts:
                validated_with_counts.append((rule, full_rule_counts[rule]))
        
        # Sort by frequency and take top N
        validated_with_counts.sort(key=lambda x: x[1], reverse=True)
        return validated_with_counts[:top_rules]
    else:
        # CPU fallback
        validated_rules = []
        for rule, count in all_rules_sorted:
            if is_valid_hashcat_rule(rule, _OP_REQS, _VALID_CHARS):
                validated_rules.append((rule, count))
            if len(validated_rules) >= top_rules:
                break
        return validated_rules

# --- Interactive Mode Functions ---
def get_yes_no(prompt, default=None):
    """Get yes/no input from user"""
    while True:
        response = input(prompt).strip().lower()
        if response in ('y', 'yes'):
            return True
        elif response in ('n', 'no'):
            return False
        elif response == '' and default is not None:
            return default
        else:
            print(f"{Colors.YELLOW}Please enter 'y' or 'n'{Colors.END}")

def get_integer(prompt, default=None, min_val=None, max_val=None):
    """Get integer input from user with validation"""
    while True:
        try:
            response = input(prompt).strip()
            if response == '' and default is not None:
                return default
            value = int(response)
            if min_val is not None and value < min_val:
                print(f"{Colors.YELLOW}Value must be at least {min_val}{Colors.END}")
                continue
            if max_val is not None and value > max_val:
                print(f"{Colors.YELLOW}Value must be at most {max_val}{Colors.END}")
                continue
            return value
        except ValueError:
            print(f"{Colors.YELLOW}Please enter a valid number{Colors.END}")

def get_file_paths_interactive():
    """Get file paths interactively from user"""
    print(f"\n{Colors.CYAN}{Colors.BOLD}" + "="*60)
    print("RULE FILE SELECTION")
    print("="*60 + f"{Colors.END}")
    
    paths = []
    while True:
        print(f"\n{Colors.CYAN}Current files/folders to process:{Colors.END}")
        for i, path in enumerate(paths, 1):
            print(f"  {Colors.WHITE}{i}. {path}{Colors.END}")
        
        print(f"\n{Colors.CYAN}Options:{Colors.END}")
        print(f"  {Colors.GREEN}1.{Colors.END} Add a file")
        print(f"  {Colors.GREEN}2.{Colors.END} Add a folder (recursive search, max depth 3)")
        print(f"  {Colors.GREEN}3.{Colors.END} Remove a file/folder")
        print(f"  {Colors.GREEN}4.{Colors.END} Start processing")
        
        choice = input(f"\n{Colors.YELLOW}Enter your choice (1-4): {Colors.END}").strip()
        
        if choice == '1':
            file_path = input(f"{Colors.CYAN}Enter file path: {Colors.END}").strip()
            if os.path.isfile(file_path):
                if file_path not in paths:
                    paths.append(file_path)
                    print(f"{Colors.GREEN}Added: {file_path}{Colors.END}")
                else:
                    print(f"{Colors.YELLOW}File already in list{Colors.END}")
            else:
                print(f"{Colors.RED}File not found or not a file{Colors.END}")
                
        elif choice == '2':
            folder_path = input(f"{Colors.CYAN}Enter folder path: {Colors.END}").strip()
            if os.path.isdir(folder_path):
                if folder_path not in paths:
                    paths.append(folder_path)
                    print(f"{Colors.GREEN}Added: {folder_path} (will search recursively, max depth 3){Colors.END}")
                else:
                    print(f"{Colors.YELLOW}Folder already in list{Colors.END}")
            else:
                print(f"{Colors.RED}Folder not found or not a directory{Colors.END}")
                
        elif choice == '3':
            if not paths:
                print(f"{Colors.YELLOW}No files to remove{Colors.END}")
                continue
            try:
                index = int(input(f"{Colors.CYAN}Enter number to remove (1-{len(paths)}): {Colors.END}")) - 1
                if 0 <= index < len(paths):
                    removed = paths.pop(index)
                    print(f"{Colors.GREEN}Removed: {removed}{Colors.END}")
                else:
                    print(f"{Colors.RED}Invalid number{Colors.END}")
            except ValueError:
                print(f"{Colors.RED}Please enter a valid number{Colors.END}")
                
        elif choice == '4':
            if not paths:
                print(f"{Colors.RED}Please add at least one file or folder first{Colors.END}")
                continue
            break
        else:
            print(f"{Colors.RED}Invalid choice{Colors.END}")
    
    return paths

def get_processing_mode_interactive():
    """Get processing mode from user"""
    print(f"\n{Colors.CYAN}{Colors.BOLD}" + "="*60)
    print("PROCESSING MODE SELECTION")
    print("="*60 + f"{Colors.END}")
    print(f"\n{Colors.CYAN}Available modes:{Colors.END}")
    print(f"  {Colors.GREEN}1.{Colors.END} Extract Rules (-e)")
    print(f"     {Colors.WHITE}- Extract top existing rules from input files{Colors.END}")
    print(f"     {Colors.WHITE}- Can sort by frequency or statistical weight{Colors.END}")
    print()
    print(f"  {Colors.GREEN}2.{Colors.END} Generate Combinatorial Rules (-g)")
    print(f"     {Colors.WHITE}- Generate new rules by combining top operators{Colors.END}")
    print(f"     {Colors.WHITE}- Creates all valid combinations within length range{Colors.END}")
    print()
    print(f"  {Colors.GREEN}3.{Colors.END} Generate Markov Rules (-gm)")
    print(f"     {Colors.WHITE}- Generate statistically probable rules using Markov model{Colors.END}")
    print(f"     {Colors.WHITE}- Creates rules that follow patterns from input data{Colors.END}")
    print()
    
    while True:
        choice = input(f"{Colors.YELLOW}Select mode (1-3): {Colors.END}").strip()
        if choice == '1':
            return 'extraction'
        elif choice == '2':
            return 'combo'
        elif choice == '3':
            return 'markov'
        else:
            print(f"{Colors.RED}Please enter 1, 2, or 3{Colors.END}")

def get_extraction_settings():
    """Get settings for extraction mode"""
    print(f"\n{Colors.CYAN}--- Extraction Settings ---{Colors.END}")
    top_rules = get_integer(f"{Colors.YELLOW}Number of top rules to extract (default 10000): {Colors.END}", default=10000, min_val=1)
    statistical_sort = get_yes_no(f"{Colors.YELLOW}Use statistical sort instead of frequency? (y/N): {Colors.END}", default=False)
    
    return {
        'top_rules': top_rules,
        'statistical_sort': statistical_sort
    }

def get_combo_settings():
    """Get settings for combinatorial generation mode"""
    print(f"\n{Colors.CYAN}--- Combinatorial Generation Settings ---{Colors.END}")
    target_rules = get_integer(f"{Colors.YELLOW}Target number of rules to generate (default 100000): {Colors.END}", default=100000, min_val=1)
    
    print(f"\n{Colors.CYAN}Rule length range:{Colors.END}")
    min_len = get_integer(f"{Colors.YELLOW}Minimum rule length (default 1): {Colors.END}", default=1, min_val=1)
    max_len = get_integer(f"{Colors.YELLOW}Maximum rule length (default 3): {Colors.END}", default=3, min_val=min_len)
    
    return {
        'target_rules': target_rules,
        'min_len': min_len,
        'max_len': max_len
    }

def get_markov_settings():
    """Get settings for Markov generation mode"""
    print(f"\n{Colors.CYAN}--- Markov Generation Settings ---{Colors.END}")
    target_rules = get_integer(f"{Colors.YELLOW}Target number of rules to generate (default 10000): {Colors.END}", default=10000, min_val=1)
    
    print(f"\n{Colors.CYAN}Rule length range:{Colors.END}")
    min_len = get_integer(f"{Colors.YELLOW}Minimum rule length (default 1): {Colors.END}", default=1, min_val=1)
    max_len = get_integer(f"{Colors.YELLOW}Maximum rule length (default 3): {Colors.END}", default=3, min_val=min_len)
    
    return {
        'target_rules': target_rules,
        'min_len': min_len,
        'max_len': max_len
    }

def get_advanced_settings():
    """Get advanced settings from user"""
    print(f"\n{Colors.CYAN}{Colors.BOLD}" + "="*60)
    print("ADVANCED SETTINGS")
    print("="*60 + f"{Colors.END}")
    
    settings = {}
    
    settings['max_length'] = get_integer(f"{Colors.YELLOW}Maximum rule length to process (default 31): {Colors.END}", default=31, min_val=1)
    
    settings['no_gpu'] = not get_yes_no(f"{Colors.YELLOW}Use GPU acceleration if available? (Y/n): {Colors.END}", default=True)
    
    settings['in_memory'] = get_yes_no(f"{Colors.YELLOW}Process entirely in RAM? (y/N): {Colors.END}", default=False)
    
    if not settings['in_memory']:
        temp_dir = input(f"{Colors.YELLOW}Temporary directory path (Enter for system default): {Colors.END}").strip()
        settings['temp_dir'] = temp_dir if temp_dir else None
    else:
        settings['temp_dir'] = None
    
    # Cleanup settings
    use_cleanup = get_yes_no(f"{Colors.YELLOW}Run cleanup after processing? (y/N): {Colors.END}", default=False)
    if use_cleanup:
        cleanup_bin = input(f"{Colors.YELLOW}Path to cleanup binary: {Colors.END}").strip()
        if cleanup_bin and os.path.isfile(cleanup_bin):
            settings['cleanup_bin'] = cleanup_bin
            settings['cleanup_arg'] = input(f"{Colors.YELLOW}Cleanup argument (default '2'): {Colors.END}").strip() or '2'
        else:
            print(f"{Colors.RED}Cleanup binary not found, disabling cleanup{Colors.END}")
            settings['cleanup_bin'] = None
    else:
        settings['cleanup_bin'] = None
    
    return settings

def get_output_filename_interactive(mode):
    """Get output filename from user"""
    print(f"\n{Colors.CYAN}{Colors.BOLD}" + "="*60)
    print("OUTPUT SETTINGS")
    print("="*60 + f"{Colors.END}")
    
    default_names = {
        'extraction': 'concentrator_extracted',
        'combo': 'concentrator_combo', 
        'markov': 'concentrator_markov'
    }
    
    default_name = default_names.get(mode, 'concentrator_output')
    
    while True:
        base_name = input(f"{Colors.YELLOW}Output base filename (default '{default_name}'): {Colors.END}").strip()
        if not base_name:
            base_name = default_name
        
        # Add appropriate suffix
        suffixes = {
            'extraction': '_extracted.txt',
            'combo': '_combo.txt',
            'markov': '_markov.txt'
        }
        
        output_file = base_name + suffixes.get(mode, '.txt')
        
        if os.path.exists(output_file):
            overwrite = get_yes_no(f"{Colors.YELLOW}File '{output_file}' already exists. Overwrite? (y/N): {Colors.END}", default=False)
            if not overwrite:
                continue
        
        return output_file, base_name

def show_summary(settings, mode_specific, output_file):
    """Show summary of selected options"""
    print(f"\n{Colors.CYAN}{Colors.BOLD}" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60 + f"{Colors.END}")
    
    print(f"{Colors.CYAN}Mode:{Colors.END} {Colors.WHITE}{mode_specific['mode'].upper()}{Colors.END}")
    print(f"{Colors.CYAN}Output file:{Colors.END} {Colors.WHITE}{output_file}{Colors.END}")
    print(f"{Colors.CYAN}Files/folders to process:{Colors.END} {Colors.WHITE}{len(settings['paths'])}{Colors.END}")
    
    if mode_specific['mode'] == 'extraction':
        print(f"{Colors.CYAN}Top rules to extract:{Colors.END} {Colors.WHITE}{mode_specific['top_rules']}{Colors.END}")
        print(f"{Colors.CYAN}Sort method:{Colors.END} {Colors.WHITE}{'Statistical' if mode_specific['statistical_sort'] else 'Frequency'}{Colors.END}")
    elif mode_specific['mode'] == 'combo':
        print(f"{Colors.CYAN}Target rules:{Colors.END} {Colors.WHITE}{mode_specific['target_rules']}{Colors.END}")
        print(f"{Colors.CYAN}Rule length range:{Colors.END} {Colors.WHITE}{mode_specific['min_len']}-{mode_specific['max_len']}{Colors.END}")
    elif mode_specific['mode'] == 'markov':
        print(f"{Colors.CYAN}Target rules:{Colors.END} {Colors.WHITE}{mode_specific['target_rules']}{Colors.END}")
        print(f"{Colors.CYAN}Rule length range:{Colors.END} {Colors.WHITE}{mode_specific['min_len']}-{mode_specific['max_len']}{Colors.END}")
    
    print(f"{Colors.CYAN}Max rule length:{Colors.END} {Colors.WHITE}{settings['max_length']}{Colors.END}")
    print(f"{Colors.CYAN}GPU acceleration:{Colors.END} {Colors.WHITE}{'No' if settings['no_gpu'] else 'Yes'}{Colors.END}")
    print(f"{Colors.CYAN}In-memory processing:{Colors.END} {Colors.WHITE}{'Yes' if settings['in_memory'] else 'No'}{Colors.END}")
    
    cleanup_enabled = settings.get('cleanup_bin') is not None
    print(f"{Colors.CYAN}Cleanup:{Colors.END} {Colors.WHITE}{'Yes' if cleanup_enabled else 'No'}{Colors.END}")
    
    print(f"\n{Colors.CYAN}{Colors.BOLD}" + "="*60 + f"{Colors.END}")

def interactive_mode():
    """Main interactive mode function"""
    print(f"\n{Colors.CYAN}{Colors.BOLD}" + "="*60)
    print("CONCENTRATOR v2.0 - Interactive Mode")
    print("="*60 + f"{Colors.END}")
    print(f"\n{Colors.YELLOW}This tool processes Hashcat rules with GPU acceleration and")
    print(f"advanced generation techniques.{Colors.END}")
    print(f"\n{Colors.CYAN}USAGE MINIMIZER TIP:{Colors.END} {Colors.WHITE}After generating rules, consider using")
    print(f"Hashcat's 'rulefilter' or 'cleanup-rules.bin' to remove")
    print(f"redundant or ineffective rules, reducing file size and")
    print(f"improving cracking performance.{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}" + "="*60 + f"{Colors.END}")
    
    # Collect all settings interactively
    paths = get_file_paths_interactive()
    mode = get_processing_mode_interactive()
    
    # Get mode-specific settings
    mode_specific = {'mode': mode}
    if mode == 'extraction':
        mode_specific.update(get_extraction_settings())
    elif mode == 'combo':
        mode_specific.update(get_combo_settings())
    elif mode == 'markov':
        mode_specific.update(get_markov_settings())
    
    # Get output filename
    output_file, base_name = get_output_filename_interactive(mode)
    
    # Get advanced settings
    advanced_settings = get_advanced_settings()
    
    # Combine all settings
    settings = {
        'paths': paths,
        'output_base_name': base_name,
        'max_length': advanced_settings['max_length'],
        'no_gpu': advanced_settings['no_gpu'],
        'in_memory': advanced_settings['in_memory'],
        'temp_dir': advanced_settings['temp_dir'],
        'cleanup_bin': advanced_settings.get('cleanup_bin'),
        'cleanup_arg': advanced_settings.get('cleanup_arg', '2')
    }
    
    # Add mode-specific settings
    settings.update(mode_specific)
    
    # Show summary and confirm
    show_summary(settings, mode_specific, output_file)
    
    if not get_yes_no(f"\n{Colors.YELLOW}Start processing? (Y/n): {Colors.END}", default=True):
        print(f"{Colors.YELLOW}Processing cancelled.{Colors.END}")
        return None
    
    return settings

# --- Main Execution Logic ---
if __name__ == '__main__':
    multiprocessing.freeze_support()  
    
    # Print banner
    print_banner()
    
    # Check RAM usage at startup
    print_memory_status()
    mem_info = check_ram_usage()
    
    if mem_info['ram_percent'] > 85:
        print(f"{Colors.RED}{Colors.BOLD}WARNING:{Colors.END} {Colors.YELLOW}High RAM usage detected ({mem_info['ram_percent']:.1f}%){Colors.END}")
        if mem_info['swap_total_gb'] == 0:
            print(f"{Colors.RED}CRITICAL: No swap space available. System may become unstable.{Colors.END}")
            proceed = get_yes_no(f"{Colors.YELLOW}Continue anyway? (y/N): {Colors.END}", default=False)
            if not proceed:
                sys.exit(1)
        else:
            print(f"{Colors.YELLOW}System will use swap space. Performance may be slower.{Colors.END}")
    
    # Check if we should use interactive mode or show help
    if len(sys.argv) == 1:
        # Interactive mode
        settings = interactive_mode()
        if not settings:
            sys.exit(0)
        
        # Convert interactive settings to argparse-like structure
        class Args:
            def __init__(self, settings):
                self.paths = settings['paths']
                self.output_base_name = settings['output_base_name']
                self.max_length = settings['max_length']
                self.no_gpu = settings['no_gpu']
                self.in_memory = settings['in_memory']
                self.temp_dir = settings['temp_dir']
                self.cleanup_bin = settings['cleanup_bin']
                self.cleanup_arg = settings['cleanup_arg']
                
                # Mode flags
                self.extract_rules = (settings['mode'] == 'extraction')
                self.generate_combo = (settings['mode'] == 'combo')
                self.generate_markov_rules = (settings['mode'] == 'markov')
                
                # Mode-specific settings
                if self.extract_rules:
                    self.top_rules = settings['top_rules']
                    self.statistical_sort = settings['statistical_sort']
                elif self.generate_combo:
                    self.combo_target = settings['target_rules']
                    self.combo_length = [settings['min_len'], settings['max_len']]
                elif self.generate_markov_rules:
                    self.generate_target = settings['target_rules']
                    self.markov_length = [settings['min_len'], settings['max_len']]
        
        args = Args(settings)
        
    elif len(sys.argv) == 2 and sys.argv[1] in ['-h', '--help']:
        # Show colorized help
        print_usage()
        sys.exit(0)
    else:
        # CLI mode (for backward compatibility)
        parser = argparse.ArgumentParser(
            description=f'{Colors.CYAN}GPU Accelerated Hashcat Rule Processor with OpenCL support.{Colors.END}',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=f"""{Colors.CYAN}Examples:{Colors.END}
  {Colors.WHITE}# Extract top rules{Colors.END}
  python concentrator.py -e -t 5000 rules/*.rule

  {Colors.WHITE}# Generate combinatorial rules{Colors.END}  
  python concentrator.py -g -n 50000 -l 2 4 hashcat/rules/

  {Colors.WHITE}# Run interactive mode{Colors.END}
  python concentrator.py
"""
        )
        
        parser.add_argument('paths', nargs='+', help='Paths to rule files or directories to analyze recursively (max depth 3)')
        
        # --- GLOBAL OUTPUT FILENAME ---
        parser.add_argument('-ob', '--output_base_name', type=str, default='concentrator_output', 
                            help='The base name for the output file. The script will append a suffix based on the mode.')
        
        # --- MONO MODE ENFORCEMENT GROUP (The Three Modes) ---
        output_group = parser.add_mutually_exclusive_group(required=True)

        # 1. Extraction Mode (Replaces old -o)
        output_group.add_argument('-e', '--extract_rules', action='store_true', help='Enables rule extraction and sorting from input files. Uses -t for count.')
        parser.add_argument('-t', '--top_rules', type=int, default=10000, help='The number of top existing rules to extract and save (used with -e).')
        parser.add_argument('-s', '--statistical_sort', action='store_true', help='Sorts EXTRACTED rules by Markov sequence probability instead of raw frequency (used with -e).')
        
        # 2. Combinatorial Generation Mode
        output_group.add_argument('-g', '--generate_combo', action='store_true', help='Enables generating combinatorial rules. Uses -n for target count.')
        parser.add_argument('-n', '--combo_target', type=int, default=100000, help='The approximate number of rules to generate in combinatorial mode (used with -g).')
        parser.add_argument('-l', '--combo_length', nargs='+', type=int, default=[1, 3], help='The range of rule chain lengths for combinatorial mode (e.g., 1 3) (used with -g).')
        
        # 3. Statistical (Markov) Generation Mode
        output_group.add_argument('-gm', '--generate_markov_rules', action='store_true', help='Enables generating statistically probable Markov rules. Uses -gt for target count.')
        parser.add_argument('-gt', '--generate_target', type=int, default=10000, help='The target number of rules to generate in Markov mode (used with -gm).')
        parser.add_argument('-ml', '--markov_length', nargs='+', type=int, default=None, help='The range of rule chain lengths for Markov mode (e.g., 1 5) (used with -gm). Defaults to [1, 3].')

        # Global/Utility Flags
        parser.add_argument('-m', '--max_length', type=int, default=31, help='The maximum length for rules to be extracted/considered in analysis. Default is 31.')
        parser.add_argument('--temp-dir', type=str, default=None, help='Optional: Specify a directory for temporary files.')
        parser.add_argument('--in-memory', action='store_true', help='Process all rules entirely in RAM.')
        parser.add_argument('--no-gpu', action='store_true', help='Disable GPU acceleration even if available.')

        # Cleanup Flags
        parser.add_argument('-cb', '--cleanup-bin', type=str, default=None, 
                             help='Optional: Path to the external cleanup binary (e.g., ./cleanup-rules.bin). If provided, it will run after rule generation.')
        parser.add_argument('-ca', '--cleanup-arg', type=str, default='2', 
                             help='Argument to pass to the cleanup binary (e.g., "2" for hashcat\'s cleanup-rules.bin).')
        
        args = parser.parse_args()
    
    # Set Markov length defaults if needed
    if hasattr(args, 'markov_length') and args.markov_length is None:
        args.markov_length = [1, 3]
    
    # Determine active mode and output filename with colors
    if args.extract_rules:
        active_mode = 'extraction'
        output_suffix = '_extracted.txt'
        mode_color = Colors.GREEN
    elif args.generate_combo:
        active_mode = 'combo'
        output_suffix = '_combo.txt'
        mode_color = Colors.BLUE
    elif args.generate_markov_rules:
        active_mode = 'markov'
        output_suffix = '_markov.txt'
        mode_color = Colors.MAGENTA
        
    output_file_name = args.output_base_name + output_suffix
    
    print(f"\n{Colors.CYAN}Active Mode:{Colors.END} {mode_color}{Colors.BOLD}{active_mode.upper()}{Colors.END}")
    print(f"{Colors.CYAN}Output File:{Colors.END} {Colors.WHITE}{output_file_name}{Colors.END}")
    
    # Set length defaults
    if active_mode == 'markov':
        markov_min_len = args.markov_length[0]
        markov_max_len = args.markov_length[-1]
    elif active_mode == 'combo':
        combo_min_len = args.combo_length[0]
        combo_max_len = args.combo_length[-1]

    # Initialize OpenCL
    gpu_enabled = False
    if not args.no_gpu:
        gpu_enabled = setup_opencl()
        if gpu_enabled:
            print(f"{Colors.GREEN}GPU Acceleration: ENABLED{Colors.END}")
        else:
            print(f"{Colors.YELLOW}GPU Acceleration: Disabled (falling back to CPU){Colors.END}")
    else:
        print(f"{Colors.YELLOW}GPU Acceleration: Manually disabled{Colors.END}")
    
    # --- RECURSIVE FILE COLLECTION LOGIC (MAX DEPTH 3) ---
    print(f"{Colors.CYAN}--- 0. Collecting Rule Files (Recursive Search, Max Depth 3) ---{Colors.END}")
    
    # Use the new recursive search function with depth limit
    all_filepaths = find_rule_files_recursive(args.paths, max_depth=3)
    
    # Ensure the determined output file is excluded from analysis
    output_files_to_exclude = {os.path.basename(output_file_name)}
    all_filepaths = [fp for fp in all_filepaths if os.path.basename(fp) not in output_files_to_exclude]

    if not all_filepaths:
        print(f"{Colors.RED}Error: No rule files found to process. Exiting.{Colors.END}")
        sys.exit(1)
        
    print(f"{Colors.GREEN}Found {len(all_filepaths)} rule files to analyze.{Colors.END}")
    
    set_global_flags(args.temp_dir, args.in_memory)

    # --- 1. Parallel Rule File Analysis ---
    print(f"{Colors.CYAN}--- 1. Starting Parallel Rule File Analysis ---{Colors.END}")
    
    sorted_op_counts, full_rule_counts, all_clean_rules = analyze_rule_files_parallel(all_filepaths, args.max_length)
    
    if not sorted_op_counts:
        print(f"{Colors.RED}No operators found in files. Exiting.{Colors.END}")
        sys.exit(1)

    # --- 2. Markov Model Building (CONDITIONAL SKIP APPLIED HERE) ---
    markov_probabilities, total_transitions = None, None
    
    # FIXED: Only build Markov model when actually needed
    # Build Markov Model ONLY for:
    # - Extraction mode with statistical sort (-e -s)
    # - Markov generation mode (-gm)
    build_markov_model = False
    
    if active_mode == 'extraction' and hasattr(args, 'statistical_sort') and args.statistical_sort:
        build_markov_model = True
        print(f"{Colors.CYAN}--- 2. Building Markov Model for Statistical Sort ---{Colors.END}")
    elif active_mode == 'markov':
        build_markov_model = True
        print(f"{Colors.CYAN}--- 2. Building Markov Model for Rule Generation ---{Colors.END}")
    else:
        print(f"{Colors.CYAN}--- 2. Skipping Markov Model Build (Not needed for current mode) ---{Colors.END}")
    
    if build_markov_model:
        markov_probabilities, total_transitions = get_markov_model(full_rule_counts)

    # --- EXECUTE ACTIVE MODE ---
    
    if active_mode == 'extraction':
        # --- Enhanced GPU Extraction Mode ---
        print(f"\n{Colors.CYAN}{Colors.BOLD}" + "~"*50)
        print("--- 3. GPU-Accelerated Rule Extraction and Validation ---")
        print("~"*50 + f"{Colors.END}")
        
        if args.statistical_sort:
            mode = 'statistical'
            
            # Check if model was actually built
            if markov_probabilities is None:
                print(f"{Colors.RED}Error: Statistical sort (-s) requires the Markov model, but it was skipped. Please report this as a script bug.{Colors.END}")
                sys.exit(1)
                
            print(f"\n{Colors.CYAN}--- Sort Mode: Statistical Sort (Markov Weight) ---{Colors.END}")
            sorted_rule_data = get_markov_weighted_rules(full_rule_counts, markov_probabilities, total_transitions)
            
            # GPU validate the statistically sorted rules
            if gpu_enabled and sorted_rule_data:
                rules_to_validate = [rule for rule, weight in sorted_rule_data[:args.top_rules*2]]
                validated_rules = gpu_validate_and_format_rules(rules_to_validate)
                
                # Re-sort validated rules by statistical weight
                validated_with_weights = []
                for rule in validated_rules:
                    for original_rule, weight in sorted_rule_data:
                        if rule == original_rule:
                            validated_with_weights.append((rule, weight))
                            break
                
                top_rules_data = validated_with_weights[:args.top_rules]
                print(f"{Colors.GREEN}GPU validated {len(top_rules_data)} statistically sorted rules{Colors.END}")
            else:
                top_rules_data = sorted_rule_data[:args.top_rules]
                
        else:
            mode = 'frequency'
            print(f"\n{Colors.CYAN}--- Sort Mode: Frequency Sort (Raw Count) with GPU Validation ---{Colors.END}")
            
            # Use GPU for extraction and validation
            top_rules_data = gpu_extract_and_validate_rules(full_rule_counts, args.top_rules, gpu_enabled)
            
        print(f"\n{Colors.GREEN}Extracted {len(top_rules_data)} top unique rules (max length: {args.max_length} characters).{Colors.END}")
        save_rules_to_file(top_rules_data, output_file_name, mode)
        
        if args.cleanup_bin:
            run_and_rename_cleanup(output_file_name, args.cleanup_bin, args.cleanup_arg)
        
    elif active_mode == 'markov':
        # --- STATISTICAL (MARKOV) RULE GENERATION (-gm) ---
        print(f"\n{Colors.MAGENTA}{Colors.BOLD}" + "!"*50)
        print("--- 3. Starting STATISTICAL Markov Rule Generation (Validated) ---")
        print("!"*50 + f"{Colors.END}")
        
        markov_rules_data = generate_rules_from_markov_model(
            markov_probabilities, 
            args.generate_target, 
            markov_min_len, 
            markov_max_len
        )
        
        # GPU validation for Markov-generated rules
        if gpu_enabled and markov_rules_data:
            markov_rules = [rule for rule, weight in markov_rules_data]
            validation_results = gpu_validate_rules(markov_rules, args.max_length)
            
            # Filter and re-weight valid rules
            valid_markov_rules = []
            for (rule, weight), is_valid in zip(markov_rules_data, validation_results):
                if is_valid:
                    valid_markov_rules.append((rule, weight))
            
            print(f"{Colors.GREEN}GPU validated {len(valid_markov_rules)}/{len(markov_rules_data)} Markov rules as syntactically valid{Colors.END}")
            markov_rules_data = valid_markov_rules[:args.generate_target]
        
        save_rules_to_file(markov_rules_data, output_file_name, 'statistical')

        if args.cleanup_bin:
            run_and_rename_cleanup(output_file_name, args.cleanup_bin, args.cleanup_arg)
        
    elif active_mode == 'combo':
        # --- COMBINATORIAL RULE GENERATION (-g) ---
        print(f"\n{Colors.BLUE}{Colors.BOLD}" + "#"*50)
        print("--- 3. Starting COMBINATORIAL Rule Generation (Validated) ---")
        print("#"*50 + f"{Colors.END}")
        
        # 3a. Find minimum number of top operators needed
        top_operators_needed = find_min_operators_for_target(
            sorted_op_counts, 
            args.combo_target, 
            combo_min_len, 
            combo_max_len
        )
        
        print(f"{Colors.CYAN}Using the top {len(top_operators_needed)} operators to approximate {args.combo_target} rules.{Colors.END}")
        
        # 3b. Generate rules with GPU acceleration if available
        if gpu_enabled:
            generated_rules_set = gpu_generate_combinatorial_rules(
                top_operators_needed, 
                combo_min_len, 
                combo_max_len
            )
        else:
            generated_rules_set = generate_rules_parallel(
                top_operators_needed, 
                combo_min_len, 
                combo_max_len
            )
        
        # 3c. Save the generated rules
        save_rules_to_file(generated_rules_set, output_file_name, 'combo')

        if args.cleanup_bin:
            run_and_rename_cleanup(output_file_name, args.cleanup_bin, args.cleanup_arg)

    print(f"\n{Colors.GREEN}{Colors.BOLD}--- Processing Complete ---{Colors.END}")
    
    # Show usage minimizer information with colors
    print(f"\n{Colors.CYAN}{Colors.BOLD}" + "="*60)
    print("USAGE MINIMIZER RECOMMENDATIONS")
    print("="*60 + f"{Colors.END}")
    print(f"{Colors.YELLOW}To optimize your generated rules and reduce file size:{Colors.END}")
    print()
    print(f"{Colors.GREEN}* Hashcat's rulefilter:{Colors.END}")
    print(f"   {Colors.WHITE}./minimizer_cl.py rulesPath{Colors.END}")
    print(f"   {Colors.CYAN}Filters rules based on various criteria{Colors.END}")
    print()
    print(f"{Colors.GREEN}* Hashcat's cleanup-rules.bin:{Colors.END}")
    print(f"   {Colors.WHITE}./cleanup-rules.bin rules.rule{Colors.END}") 
    print(f"   {Colors.CYAN}Removes redundant and ineffective rules{Colors.END}")
    print()
    print(f"{Colors.YELLOW}These tools can significantly reduce rule file size while")
    print(f"maintaining or even improving cracking effectiveness.{Colors.END}")
    print(f"{Colors.CYAN}" + "="*60 + f"{Colors.END}")
    
    if gpu_enabled:
        print(f"{Colors.GREEN}GPU Acceleration was used for improved performance{Colors.END}")
    
    # Final RAM usage check
    print_memory_status()
    
    sys.exit(0)
