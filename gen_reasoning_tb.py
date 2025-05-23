#!/usr/bin/env python3
import os
import json
import time
import subprocess
import tempfile
import re
import signal
from openai import OpenAI
from tqdm import tqdm
from contextlib import contextmanager

# Configuration
API_KEY = "YOUR-API-TOKEN"  # Replace with your actual API key
OUTPUT_DIR = "YOUR-OUTPUT-DIR"  # Replace with your desired output directory
INPUT_FILE = "YOUR-INPUT-FILE"  # Replace with your input file path
MAX_RETRIES = 5  # Maximum number of retries for generation/compilation
SIMULATION_TIMEOUT = 30  # Timeout in seconds for simulation

# Timeout handling
class TimeoutError(Exception):
    pass

@contextmanager
def time_limit(seconds):
    """
    Context manager to limit execution time of a block of code.
    Works on Unix systems using signals.
    
    Args:
        seconds: Maximum execution time in seconds
        
    Raises:
        TimeoutError: If the execution time exceeds the limit
    """
    def signal_handler(signum, frame):
        raise TimeoutError("Timed out!")
        
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)  # Disable the alarm

def load_local_dataset(file_path):
    """Load the local dataset from a JSON file."""
    print(f"Loading dataset from {file_path}...")
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Process the data to ensure each entry has an ID and the required fields
        processed_entries = []
        
        # Print the structure for debugging
        if isinstance(data, dict):
            print(f"Dataset is a dictionary with {len(data)} entries")
            print(f"Sample keys: {list(data.keys())[:5]}")
            
            # Check a sample entry to understand structure
            sample_key = next(iter(data))
            sample_value = data[sample_key]
            print(f"Sample entry structure: {sample_key}: {type(sample_value)}")
            if isinstance(sample_value, dict):
                print(f"Sample entry keys: {list(sample_value.keys())}")
            
            # Process each key-value pair in the dictionary
            for i, (key, value) in enumerate(data.items()):
                entry = {'id': f"entry_{i+1}"}
                
                # Handle specific structure found in the sample
                if isinstance(value, dict):
                    # Check for various field names with case insensitivity
                    for field_name in ['instruction', 'Instruction', 'prompt', 'Prompt']:
                        if field_name in value:
                            entry['instruction'] = value[field_name]
                            break
                    
                    # Check for various output field names
                    for field_name in ['response', 'Response', 'output', 'Output', 'code', 'Code']:
                        if field_name in value:
                            # Handle case where response is a list
                            if isinstance(value[field_name], list):
                                entry['output'] = '\n'.join(value[field_name])
                            else:
                                entry['output'] = value[field_name]
                            break
                else:
                    # If the value is not a dict, use it as the instruction
                    entry['instruction'] = str(value)
                
                processed_entries.append(entry)
        elif isinstance(data, list):
            print(f"Dataset is a list with {len(data)} entries")
            if data:
                print(f"Sample entry structure: {type(data[0])}")
                if isinstance(data[0], dict):
                    print(f"Sample entry keys: {list(data[0].keys())}")
            
            # Process each item in the list
            for i, item in enumerate(data):
                entry = {'id': f"entry_{i+1}"}
                
                if isinstance(item, dict):
                    # Check for various field names with case insensitivity
                    for field_name in ['instruction', 'Instruction', 'prompt', 'Prompt']:
                        if field_name in item:
                            entry['instruction'] = item[field_name]
                            break
                    
                    # Check for various output field names
                    for field_name in ['response', 'Response', 'output', 'Output', 'code', 'Code']:
                        if field_name in item:
                            # Handle case where response is a list
                            if isinstance(item[field_name], list):
                                entry['output'] = '\n'.join(item[field_name])
                            else:
                                entry['output'] = item[field_name]
                            break
                else:
                    # If the item is not a dict, use it as the instruction
                    entry['instruction'] = str(item)
                
                processed_entries.append(entry)
        
        # Verify the processed entries have the required fields
        valid_entries = []
        for entry in processed_entries:
            if 'instruction' in entry:
                valid_entries.append(entry)
            else:
                print(f"Warning: Entry {entry.get('id', 'unknown')} has no instruction, skipping")
        
        print(f"Loaded and processed {len(valid_entries)} valid entries.")
        return valid_entries
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Checking the first part of the file for debug purposes:")
        try:
            with open(file_path, 'r') as f:
                first_part = f.read(1000)  # Read first 1000 characters
                print(first_part)
        except Exception as debug_e:
            print(f"Error reading file: {debug_e}")
        return []

def check_verilog_syntax(code):
    """
    Check if the Verilog code has correct syntax using iverilog
    
    Args:
        code (str): Verilog code to check
        
    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        # Create a temporary file to store the Verilog code
        with tempfile.NamedTemporaryFile(suffix='.v', delete=False) as temp_file:
            temp_filename = temp_file.name
            temp_file.write(code.encode('utf-8'))
        
        # Run iverilog to check syntax
        result = subprocess.run(
            ['iverilog', '-t', 'null', temp_filename],
            capture_output=True,
            text=True
        )
        
        # Remove the temporary file
        os.unlink(temp_filename)
        
        if result.returncode == 0:
            print("Verilog syntax is valid.")
            return True, ""
        else:
            print(f"Syntax error in Verilog code: {result.stderr}")
            return False, result.stderr
        
    except Exception as e:
        print(f"Error checking Verilog syntax: {e}")
        return False, str(e)

def generate_reasoning(client, prompt, code, fallback_only=False):
    """
    Generate reasoning about how the code was developed based on the prompt.
    This function uses the OpenAI API to generate the reasoning.
    
    Args:
        client: OpenAI client
        prompt (str): The original prompt
        code (str): The Verilog code
        fallback_only (bool): Whether to use only fallback reasoning
    """
    if fallback_only:
        return generate_fallback_reasoning(prompt, code)
    
    try:
        # Create a system prompt and user prompt for the OpenAI API
        system_content = """
You are an expert Verilog engineer. Given a prompt and corresponding Verilog code, concisely outline the reasoning steps a skilled engineer would follow to implement this design. However, if you think the design is incorrect, just output the keyword "incorrect". If it is correct to you, your reasoning should briefly and clearly cover:

1. Requirement Analysis: Clearly summarize input-output behavior.
2. Logical Design Decisions: State essential logic and conditions.
3. Implementation Choices: Briefly justify chosen implementation (combinational vs. sequential, key signals used).

Be concise but precise. Your explanation should directly guide logical decision-making for the provided implementation.
"""
        
        user_content = f"""
Original prompt: {prompt}

Generated code:
```verilog
{code}
```

Provide a detailed step-by-step reasoning for how you would approach implementing this code.
"""
        
        # Make the API call to GPT-4.1
        completion = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            max_tokens=400,
        )
        
        # Extract the reasoning from the response
        reasoning = completion.choices[0].message.content
        
        return reasoning
        
    except Exception as e:
        print(f"Error generating reasoning with OpenAI API: {e}. Using fallback reasoning.")
        return generate_fallback_reasoning(prompt, code)

def generate_fallback_reasoning(prompt, code):
    """
    Fallback method to generate reasoning when the API call fails.
    """
    reasoning_lines = [
        f"Let me analyze this prompt: \"{prompt}\"",
        "",
        "First, I need to understand what kind of RTL code is required:",
        "- Identifying the main functionality requested",
        "- Determining inputs and outputs",
        "- Planning the module structure",
        "",
        "Based on the requirements, I'll implement the design with:"
    ]
    
    # Add some analysis based on the code
    if "module" in code:
        reasoning_lines.append("- A Verilog module with appropriate ports")
    if "always" in code:
        reasoning_lines.append("- Sequential logic using always blocks")
    if "assign" in code:
        reasoning_lines.append("- Combinational logic using assign statements")
    if "parameter" in code or "localparam" in code:
        reasoning_lines.append("- Parameterized design for flexibility")
    
    reasoning_lines.extend([
        "",
        "The implementation follows best practices for RTL design:",
        "- Clear signal naming",
        "- Proper synchronization",
        "- Handling edge cases"
    ])
    
    return "\n".join(reasoning_lines)

def generate_verilog_code(client, prompt, original_code=None, error_msg=None):
    """
    Generate fixed Verilog code based on the prompt and original code using OpenAI API
    
    Args:
        client: OpenAI client
        prompt: Original prompt
        original_code: Original code to fix (if available)
        error_msg: Error message from syntax check (if available)
    """
    try:
        # Create a system prompt and user prompt for the OpenAI API
        system_content = """
You are an expert Verilog engineer. Given a prompt, create syntactically correct Verilog code that implements the requested functionality. 
Focus on creating code that will pass syntax validation with iverilog. Ensure proper module declaration, port lists, and signal types.
"""
        
        user_content = f"""
Original prompt: {prompt}

Generate syntactically correct Verilog code that implements this functionality. 
The code should pass iverilog syntax validation without errors.
"""

        # If original code and error message are provided, include them for fixing
        if original_code and error_msg:
            user_content = f"""
Original prompt: {prompt}

Here is the original Verilog code that needs fixing:
```verilog
{original_code}
```

The code has the following syntax errors:
{error_msg}

Please fix the code to make it syntactically correct while preserving the intended functionality.
The code should pass iverilog syntax validation without errors.
"""
        
        # Make the API call to GPT-4.1
        completion = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            max_tokens=800,
        )
        
        # Extract the code from the response
        response = completion.choices[0].message.content
        
        # Extract code from markdown if present
        if "```verilog" in response:
            code_blocks = response.split("```verilog")
            if len(code_blocks) > 1:
                code = code_blocks[1].split("```")[0].strip()
                return code
        elif "```" in response:
            code_blocks = response.split("```")
            if len(code_blocks) > 1:
                code = code_blocks[1].strip()
                return code
        
        # If no markdown, return the whole response
        return response
        
    except Exception as e:
        print(f"Error generating code with OpenAI API: {e}. Using fallback code.")
        return "module fallback(input clk, input reset, output reg out); always @(posedge clk) begin if(reset) out <= 0; else out <= ~out; end endmodule"

def identify_module_name(verilog_code):
    """Identify the module name from the Verilog code."""
    match = re.search(r'module\s+(\w+)\s*\(', verilog_code)
    if match:
        return match.group(1)
    # Try alternative format with parameters
    match = re.search(r'module\s+(\w+)\s*#\s*\(', verilog_code)
    if match:
        return match.group(1)
    return None

def generate_testbench(client, verilog_code, module_name, error_msg=None):
    """
    Generate a testbench for the given Verilog code using OpenAI API.
    
    Args:
        client: OpenAI client
        verilog_code: Verilog code to test
        module_name: Name of the module
        error_msg: Optional error message from previous compilation attempt
        
    Returns:
        Generated testbench code
    """
    system_content = """
You are an expert in Verilog HDL and verification. Your task is to create a comprehensive testbench 
for a given Verilog module. The testbench should:

1. Test all functionality of the module
2. Generate test vectors that cover all possible input combinations for small modules or representative test cases for larger modules
3. Save test vectors and results to a text file using $fopen, $fdisplay, and $fclose
4. Include appropriate assertions to verify correct behavior
5. Be clean, well-commented, and synthesis-correct Verilog code
6. Include a task to check for errors and report them
7. Write test vectors to "test_vectors.txt" that can be used for equivalence checking
8. IMPORTANT: Include a timeout mechanism by keeping track of simulation time and finishing after a reasonable amount of time

IMPORTANT: Ensure your testbench code does not contain any syntax errors. Pay special attention to semicolons, parentheses, and brackets.

The format for test_vectors.txt should be one test vector per line with all inputs, 
followed by a space, then all outputs.

The actual file must be created using a file descriptor like this:
integer fd;
initial begin
    fd = $fopen("test_vectors.txt", "w");
    if (fd) $display("File opened successfully");
    else $display("Failed to open file");
    // ... test vector generation and file writing
    $fclose(fd);
end
"""
    
    user_content = f"""
Create a comprehensive Verilog testbench for the following module:

```verilog
{verilog_code}
```

Requirements:
1. The testbench must be named "{module_name}_tb.v"
2. It must generate appropriate test vectors (at least 100)
3. It must save all test vectors and results to "test_vectors.txt" with proper file handling
4. The test vectors file must include ALL inputs and outputs for each test case
5. All test cases must be verified with assertions or if-statements
6. The testbench must finish with $finish after all tests are complete
7. Make sure the test vectors file format is consistent (one vector per line)
8. IMPORTANT: Verify your code has no syntax errors! Every initial/always block must be properly closed.
9. CRITICAL: Add a timeout mechanism using Verilog's `$time` function to ensure the simulation completes in a reasonable time (use #10000 or similar time limit)

Return ONLY the Verilog code with no explanations or markdown.
"""
    
    # If there was a previous error, include it in the prompt
    if error_msg:
        user_content += f"""

IMPORTANT: The previous testbench attempt failed with this error:
{error_msg}

Please fix these issues in your implementation.
"""
    
    try:
        completion = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
        )
        
        # Extract the code from the response
        response = completion.choices[0].message.content
        
        # Clean up response (remove any markdown code blocks if present)
        response = re.sub(r'```verilog|```', '', response).strip()
        
        return response
        
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return None

def run_with_timeout(cmd, timeout, cwd=None):
    """
    Run a command with a timeout.
    
    Args:
        cmd: Command to run as a list of arguments
        timeout: Timeout in seconds
        cwd: Working directory for the command
        
    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd
    )
    
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        return process.returncode, stdout, stderr
    except subprocess.TimeoutExpired:
        # Kill the process if it times out
        process.kill()
        try:
            # Try to capture any output before killing
            stdout, stderr = process.communicate(timeout=1)
        except subprocess.TimeoutExpired:
            stdout, stderr = "", ""
        return None, stdout, f"Process timed out after {timeout} seconds"

def compile_and_run(verilog_code, testbench_code, module_name):
    """
    Compile the Verilog code and testbench with iverilog and run the simulation.
    
    Args:
        verilog_code: Original Verilog code
        testbench_code: Generated testbench code
        module_name: Name of the module
        
    Returns:
        Tuple of (success, simulation_output, error_message, test_vectors)
    """
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Write Verilog code to a file
        verilog_file = os.path.join(temp_dir, f"{module_name}.v")
        with open(verilog_file, "w") as f:
            f.write(verilog_code)
        
        # Write testbench code to a file
        tb_file = os.path.join(temp_dir, f"{module_name}_tb.v")
        with open(tb_file, "w") as f:
            f.write(testbench_code)
        
        # Compile with iverilog
        output_file = os.path.join(temp_dir, "a.out")
        returncode, stdout, stderr = run_with_timeout(
            ["iverilog", "-o", output_file, verilog_file, tb_file],
            timeout=SIMULATION_TIMEOUT,
            cwd=temp_dir
        )
        
        if returncode is None:
            print("Compilation timed out")
            return False, "", "Compilation timed out", ""
        
        if returncode != 0:
            print("Compilation failed:")
            print(stderr)
            return False, "", stderr, ""
        
        # Run simulation with timeout
        returncode, stdout, stderr = run_with_timeout(
            ["vvp", output_file],
            timeout=SIMULATION_TIMEOUT,
            cwd=temp_dir
        )
        
        if returncode is None:
            print("Simulation timed out")
            return False, stdout, "Simulation timed out after {SIMULATION_TIMEOUT} seconds", ""
        
        if returncode != 0:
            print("Simulation failed:")
            print(stderr)
            return False, "", stderr, ""
        
        # Check if simulation indicates a failure
        if "ERROR" in stdout or "FAILED" in stdout:
            return False, stdout, "Simulation reported errors", ""
        
        # Check if test vector file was created
        test_vectors_file = os.path.join(temp_dir, "test_vectors.txt")
        test_vectors = ""
        if os.path.exists(test_vectors_file):
            # Read test vectors file
            with open(test_vectors_file, 'r') as f:
                test_vectors = f.read()
        else:
            print("Test vectors file was not created")
        
        return True, stdout, "", test_vectors

def format_output_with_reasoning(reasoning, verilog_code):
    """Format the output with reasoning and code according to the required format."""
    return f"<think>\n{reasoning}\n</think>\n<answer>\n```verilog\n{verilog_code}\n```\n</answer>"

def verify_equivalence(module1_test_vectors, module2_test_vectors):
    """
    Verify equivalence between two simulation runs by comparing their test vectors.
    
    Args:
        module1_test_vectors: Test vectors from the first run
        module2_test_vectors: Test vectors from the second run
        
    Returns:
        Tuple of (is_equivalent, comparison_report)
    """
    if not module1_test_vectors or not module2_test_vectors:
        return False, "Missing test vectors for comparison"
    
    # Parse the test vectors
    module1_lines = module1_test_vectors.strip().split('\n')
    module2_lines = module2_test_vectors.strip().split('\n')
    
    # Ensure there are test vectors to compare
    if not module1_lines or not module2_lines:
        return False, "No test vectors available for comparison"
    
    # Count of matching and non-matching vectors
    matching = 0
    non_matching = 0
    mismatches = []
    
    # Compare test vectors line by line
    for i, (line1, line2) in enumerate(zip(module1_lines, module2_lines)):
        # Skip lines that don't contain test vectors
        if not line1.strip() or not line2.strip():
            continue
            
        # Parse the input and output values
        parts1 = line1.strip().split()
        parts2 = line2.strip().split()
        
        # Check if both lines have the same format
        if len(parts1) != len(parts2):
            non_matching += 1
            mismatches.append(f"Line {i+1}: Different number of values")
            continue
            
        # Compare the values
        if parts1 == parts2:
            matching += 1
        else:
            non_matching += 1
            # Find which values are different
            diff_indices = [j for j in range(len(parts1)) if parts1[j] != parts2[j]]
            diff_info = ', '.join([f"value {j}: {parts1[j]} vs {parts2[j]}" for j in diff_indices])
            mismatches.append(f"Line {i+1}: {diff_info}")
    
    # Generate a comparison report
    total_vectors = matching + non_matching
    if total_vectors == 0:
        return False, "No valid test vectors found for comparison"
        
    match_percentage = (matching / total_vectors) * 100 if total_vectors > 0 else 0
    
    # Modules are considered equivalent if 95% or more test vectors match
    is_equivalent = match_percentage >= 95
    
    report = f"Equivalence comparison results:\n"
    report += f"Total test vectors: {total_vectors}\n"
    report += f"Matching vectors: {matching} ({match_percentage:.2f}%)\n"
    report += f"Non-matching vectors: {non_matching}\n"
    report += f"Equivalent: {'Yes' if is_equivalent else 'No'}\n\n"
    
    # Include details of mismatches (limit to first 10 for readability)
    if mismatches:
        report += "Mismatch details (first 10):\n"
        for i, mismatch in enumerate(mismatches[:10]):
            report += f"{mismatch}\n"
        if len(mismatches) > 10:
            report += f"... and {len(mismatches) - 10} more mismatches\n"
    
    return is_equivalent, report

def process_entry(client, entry, output_file):
    """Process a single entry from the dataset."""
    entry_id = entry.get('id', 'unknown')
    print(f"\nProcessing entry ID: {entry_id}")
    
    # Get the original prompt/instruction
    prompt = entry.get('instruction', '')
    
    # Print the entry keys for debugging
    print(f"Entry keys: {list(entry.keys())}")
    
    # Extract code from the output field if available
    original_code = None
    if 'output' in entry:
        # Check if the output is already code (without markdown)
        if not '```' in entry['output'] and not '<answer>' in entry['output']:
            original_code = entry['output']
        else:
            # Try to extract code from various formats
            patterns = [
                r'```verilog\s*([\s\S]*?)\s*```',  # Verilog markdown
                r'```\s*([\s\S]*?)\s*```',         # Generic markdown
                r'<answer>\s*```verilog\s*([\s\S]*?)\s*```\s*</answer>',  # Answer with Verilog
                r'<answer>\s*```\s*([\s\S]*?)\s*```\s*</answer>'          # Answer with generic code
            ]
            
            for pattern in patterns:
                match = re.search(pattern, entry['output'])
                if match:
                    original_code = match.group(1).strip()
                    break
    
    # Print what we found
    print(f"Prompt length: {len(prompt)} characters")
    if original_code:
        print(f"Original code length: {len(original_code)} characters")
    else:
        print("No original code found in output field")
    
    if not prompt:
        print("Missing prompt/instruction, skipping entry")
        return None
    
    if not original_code:
        print("Generating new code from prompt")
        # Generate new code if no original code is found
        verilog_code = generate_verilog_code(client, prompt)
    else:
        # Check syntax of original code
        is_syntax_valid, syntax_error = check_verilog_syntax(original_code)
        
        if is_syntax_valid:
            print("Original code syntax is valid.")
            verilog_code = original_code
        else:
            print(f"Original code has syntax errors: {syntax_error}")
            print("Generating fixed code...")
            # Generate fixed code
            verilog_code = generate_verilog_code(client, prompt, original_code, syntax_error)
            
            # Verify the fixed code
            is_fixed_valid, fixed_error = check_verilog_syntax(verilog_code)
            if not is_fixed_valid:
                print(f"Fixed code still has syntax errors: {fixed_error}")
                print("Generating new code from scratch...")
                verilog_code = generate_verilog_code(client, prompt)
                
                # Final check
                is_valid, error = check_verilog_syntax(verilog_code)
                if not is_valid:
                    print(f"Failed to generate valid code after multiple attempts: {error}")
                    return None
    
    # Generate reasoning for the code
    print("Generating reasoning...")
    reasoning = generate_reasoning(client, prompt, verilog_code)
    
    # Check if reasoning contains "incorrect" or similar indications
    if "incorrect" in reasoning.lower() or "wrong" in reasoning.lower() or "not correct" in reasoning.lower():
        print("Reasoning indicates problems with the code, generating new code...")
        # Generate new code
        verilog_code = generate_verilog_code(client, prompt)
        
        # Check the new code
        is_valid, error = check_verilog_syntax(verilog_code)
        if not is_valid:
            print(f"New code has syntax errors: {error}")
            return None
        
        # Generate new reasoning for the new code
        reasoning = generate_reasoning(client, prompt, verilog_code)
    
    # Identify module name
    module_name = identify_module_name(verilog_code)
    if not module_name:
        print("Could not identify module name, skipping testbench generation")
        
        # Create result without testbench
        result = {
            'id': entry_id,
            'instruction': prompt,
            'output': format_output_with_reasoning(reasoning, verilog_code),
            'tb': None,
            'tb_result': None,
            'is_deterministic': None,
            'equivalence_report': None
        }
        
        # Save the result
        save_entry(result, output_file)
        return result
    
    print(f"Module name: {module_name}")
    
    # Generate testbench
    print("Generating testbench...")
    testbench_code = generate_testbench(client, verilog_code, module_name)
    if not testbench_code:
        print("Failed to generate testbench")
        
        # Create result without testbench
        result = {
            'id': entry_id,
            'instruction': prompt,
            'output': format_output_with_reasoning(reasoning, verilog_code),
            'tb': None,
            'tb_result': None,
            'is_deterministic': None,
            'equivalence_report': None
        }
        
        # Save the result
        save_entry(result, output_file)
        return result
    
    # Try to compile and run the testbench
    success = False
    error_msg = None
    test_vectors1 = ""
    
    for retry in range(MAX_RETRIES):
        print(f"Running simulation (attempt {retry+1})...")
        success, simulation_output, error_msg, test_vectors1 = compile_and_run(
            verilog_code, testbench_code, module_name
        )
        
        if success:
            print("Simulation succeeded!")
            break
        
        if retry < MAX_RETRIES - 1:
            print(f"Simulation failed with error: {error_msg}")
            print(f"Regenerating testbench (attempt {retry+2})...")
            # Generate a new testbench with the error message for guidance
            testbench_code = generate_testbench(client, verilog_code, module_name, error_msg)
            if not testbench_code:
                print(f"Failed to regenerate testbench")
                break
        else:
            print(f"All simulation attempts failed")
    
    # If first simulation failed, create and save result without equivalence check
    if not success:
        result = {
            'id': entry_id,
            'instruction': prompt,
            'output': format_output_with_reasoning(reasoning, verilog_code),
            'tb': testbench_code,
            'tb_result': None,
        }
        return result
        
    # If first run succeeds, try a second run to verify determinism
    print("\nRunning simulation (second run for determinism verification)...")
    success2, sim_output2, error_msg2, test_vectors2 = compile_and_run(
        verilog_code, testbench_code, module_name
    )
    
    is_deterministic = False
    equivalence_report = None
    
    if not success2:
        print(f"Second simulation run failed: {error_msg2}")
    else:
        # Both runs succeeded, check if results are the same
        print("Both simulation runs succeeded!")
        vector_count1 = len(test_vectors1.strip().split("\n")) if test_vectors1 else 0
        vector_count2 = len(test_vectors2.strip().split("\n")) if test_vectors2 else 0
        print(f"Test vectors: {vector_count1} vectors in first run, {vector_count2} vectors in second run")
        
        # Compare the test vectors from both runs
        if test_vectors1 and test_vectors2:
            is_deterministic, equivalence_report = verify_equivalence(test_vectors1, test_vectors2)
            
            if is_deterministic:
                print(f"Results are deterministic - both runs produced equivalent test vectors!")
            else:
                print(f"Results are NOT deterministic - runs produced different test vectors")
            
            print(equivalence_report)
    
    # Create the result with determinism check results
    result = {
        'id': entry_id,
        'instruction': prompt,
        'output': format_output_with_reasoning(reasoning, verilog_code),
        'tb': testbench_code,
        'tb_result': test_vectors1,
        #'is_deterministic': is_deterministic,
        #'equivalence_report': equivalence_report
    }
    
    # Save the result
    save_entry(result, output_file)
    return result

def save_entry(entry, output_file):
    """Save a single entry to the output file."""
    with open(output_file, 'a') as f:
        f.write(json.dumps(entry) + "\n")

def main():
    """Main function to process the dataset."""
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Set up output file
    output_file = os.path.join(OUTPUT_DIR, "processed_entries.jsonl")
    
    # Clear output file if it exists
    #if os.path.exists(output_file):
        #os.remove(output_file)
    
    # Load the dataset
    entries = load_local_dataset(INPUT_FILE)
    
    if not entries:
        print("No entries found in the dataset. Exiting.")
        return
    
    # Create OpenAI client
    client = OpenAI(api_key=API_KEY)
    
    # Statistics
    total = len(entries)
    processed = 0
    successful = 0
    deterministic = 0
    
    # Process each entry
    for i, entry in enumerate(tqdm(entries, desc="Processing entries")):
        print(f"\nEntry {i+1}/{total}")
        result = process_entry(client, entry, output_file)
        
        # Update statistics
        processed += 1
        if result and result.get('tb_result'):
            successful += 1
            if result.get('is_deterministic'):
                deterministic += 1
        
        # Print progress
        print(f"Processed: {processed}/{total}, Successful: {successful}, Deterministic: {deterministic}")
        print("=" * 80)
        
        # Pause briefly to avoid API rate limits
        time.sleep(0.5)
        
        # Longer pause every 10 entries
        if i % 10 == 9:
            print("Taking a short break to avoid rate limits...")
            time.sleep(5)
    
    # Print final statistics
    print("\nProcessing complete!")
    print(f"Total entries processed: {processed}/{total}")
    print(f"Successful simulations: {successful}")
    print(f"Deterministic implementations: {deterministic}")
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    print("=" * 80)
    print("Combined RTL Processing Script")
    print("This script generates reasoning, fixes Verilog code, and creates testbenches")
    print("=" * 80)
    
    main()
