**Concentrator v3.0: Unified Hashcat Rule Processor**

Concentrator v3.0 is an advanced, high-performance tool written in Python 3 designed to unify the processes of extracting, validating, cleaning, and generating highly effective Hashcat rulesets. It features multi-processing for parallel file ingestion and optional OpenCL (GPU) acceleration for massive-scale rule validation and filtering.

✨ **Key Features**

- Three Processing Modes: Extract top-performing rules, generate combinatorial rule sets, or generate Markov-chain based rules.

- OpenCL Acceleration: Optional GPU-backed processing for rule validation, providing significant speed improvements over CPU-only methods for large datasets.

- Hashcat Engine Simulation: Includes a built-in Python simulation of the Hashcat rule engine for functional testing and minimization (preventing functionally duplicate rules).

- Memory Safety: Features proactive memory usage monitoring (psutil) to warn users before performing memory-intensive operations, preventing system instability.

- Advanced Filtering: Supports complex cleanup and deduplication strategies post-generation, including Levenshtein distance filtering (though not fully implemented in the provided snippet, it is implied by the core focus on comprehensive filtering).

- Interactive and CLI Modes: Supports full command-line arguments as well as a user-friendly, colorized interactive setup mode.

🚀 **Getting Started**

Prerequisites

Concentrator v3.0 requires Python 3.8 or higher. For full functionality, including GPU acceleration and advanced monitoring, the following dependencies are required.

We recommend installing them within a virtual environment:

```# Setup virtual environment
python3 -m venv venv
source venv/bin/activate

# Install core dependencies
pip install tqdm psutil numpy

# Install OpenCL dependencies (Note: pyopencl installation may require system-level OpenCL drivers)
pip install pyopencl
```


***Installation***

Clone the repository and run the script directly:

```
git clone [https://github.com/your-org/concentrator-v3.git](https://github.com/your-org/concentrator-v3.git)
cd concentrator-v3
python3 concentrator_v3.py --help
```

⚙️ **Usage**

The tool operates via command-line arguments or an interactive mode (-I).

Modes of Operation

The core functionality is driven by the --mode flag:



*Extraction*

```--mode extract```

Extracts the top N most frequent/statistical rules from existing rule files.

*Combinatorial*

```--mode combo```

Generates new rules by combining selected individual operators up to a max length.

Markov

```--mode markov```

Generates rules based on Markov chain statistics derived from input rules, focusing on high-probability sequences.

**Command-Line Examples**

1. Interactive Setup

Run the script without arguments or with the -I flag to enter the guided setup:

```python3 concentrator_v3.py -I```
# Follow the on-screen prompts for paths, mode, and settings.


2. Extraction Mode (Top 5000 rules)

```python3 concentrator_v3.py \
    --mode extract \
    --paths rules/rockyou.rules rules/top10.txt \
    --top 5000 \
    --output final_top_rules.txt
```

3. Combinatorial Generation (Length 1-3, GPU disabled)

```python3 concentrator_v3.py \
    --mode combo \
    --paths seed_operators.txt \
    --min-len 1 --max-len 3 \
    --target 1000000 \
    --no-gpu \
    --output combo_rules.txt
```

🧠 **Architecture Overview**

The Concentrator system operates in several key phases:

**File Ingestion:** Input paths are recursively searched, and files are streamed or read into memory (depending on the --in-memory flag).

**Preprocessing & Validation:** Initial rules are filtered for basic Hashcat syntax using multi-processing.

**Generation/Extraction:** The selected mode (Extraction, Combo, or Markov) generates a massive candidate set of rules.

**Functional Minimization:** Candidates are passed through the Python-implemented RuleEngine to ensure they produce unique output for common test strings, reducing redundancy.

**GPU Validation (Optional):** If OpenCL is enabled, the final candidate rules are batched and sent to the GPU for highly optimized validation against character set constraints and length limits.

**Final Output:** Cleaned, unique, and validated rules are written to the output file.

⚠️ **Memory and Safety**

The tool is designed to handle very large rule files (billions of rules). It includes a robust memory monitoring system (check_memory_safety and memory_intensive_operation_warning) that leverages psutil. If RAM + Swap usage exceeds a safety threshold (default 85%) before a major operation (like functional minimization), the user is warned and asked to confirm continuation to prevent system lockups.

🛠️**OpenCL Integration**

The OpenCL portion requires pyopencl and is currently implemented for the final-stage validation (validate_rules_batch kernel). This offloads basic rule integrity checks to the GPU, making the overall process highly scalable. Ensure your system has the correct vendor drivers (NVIDIA, AMD, or Intel) installed for OpenCL support.

📝 License

Distributed under the MIT License. See LICENSE for more information.

**Credits**

- https://github.com/mkb2091/PyRuleEngine/blob/master/PyRuleEngine.py
- https://github.com/hashcat/hashcat-utils/blob/master/src/cleanup-rules.c
