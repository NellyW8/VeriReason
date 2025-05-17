import logging
import os
import tempfile
import subprocess
import re
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

def extract_verilog_code(completion: str) -> str:
    """Extract Verilog code from the completion."""
    # First, try to find code within verilog code blocks
    pattern = re.compile(r"```(?:verilog|v)\n(.*?)```", re.DOTALL)
    matches = pattern.findall(completion)
    if matches:
        extracted_answer = matches[-1]  # Take the last code block if multiple exist
        return extracted_answer.strip()
    
    # If the entire completion is a module declaration, return it
    if completion.strip().startswith("module") and "endmodule" in completion:
        module_pattern = re.compile(r"(module\s+\w+.*?endmodule)", re.DOTALL)
        module_matches = module_pattern.findall(completion)
        if module_matches:
            return completion.strip()
    
    # If no code block found but the text looks like Verilog code, return it
    # This handles the case where the model correctly follows instructions and outputs only code
    if "module" in completion and "endmodule" in completion:
        return completion.strip()
    
    # If we still haven't found anything, try to find any module declaration in the text
    module_pattern = re.compile(r"(module\s+\w+.*?endmodule)", re.DOTALL)
    module_matches = module_pattern.findall(completion)
    if module_matches:
        return module_matches[0].strip()
    
    # Last resort: return the completion as is
    return completion.strip()

def check_syntax_with_iverilog(code: str) -> tuple[bool, str, float]:
    """Check if the Verilog code compiles with iverilog.
    
    Returns:
        tuple: (syntax_ok, details, score)
    """
    print("code:", code)
    try:
        with tempfile.NamedTemporaryFile(suffix='.v', delete=False) as f:
            f.write(code.encode('utf-8'))
            temp_filename = f.name
        
        # Run iverilog to check syntax
        result = subprocess.run(
            ['iverilog', '-t', 'null', temp_filename],
            capture_output=True,
            text=True,
            timeout=5  # Add timeout to prevent hanging
        )
        
        os.unlink(temp_filename)  # Clean up the temp file
        
        if result.returncode == 0:
            return True, "Syntax check passed", 1.0
        else:
            error_msg = result.stderr.strip()
            print(f"Syntax error: {error_msg}")
            return False, f"Syntax error: {error_msg}", 0.0
    
    except Exception as e:
        print(f"Error during syntax check: {e}")
        logger.error(f"Error during syntax check: {e}")
        return False, f"Error during syntax check: {str(e)}", 0.0

def extract_module_name(verilog_code: str) -> str:
    """
    Extract the module name from Verilog code.
    
    Args:
        verilog_code: The Verilog code to extract module name from
        
    Returns:
        str: The extracted module name, or None if not found
    """
    import re
    
    # Look for module declaration
    module_match = re.search(r'module\s+(\w+)', verilog_code)
    if module_match:
        return module_match.group(1)
    
    return None

def replace_module_name(verilog_code: str, old_name: str, new_name: str) -> str:
    """
    Replace the module name in Verilog code.
    
    Args:
        verilog_code: The Verilog code to modify
        old_name: The current module name to replace
        new_name: The new module name to use
        
    Returns:
        str: The modified Verilog code with the new module name
    """
    import re
    
    # Replace module declaration
    modified_code = re.sub(
        r'module\s+' + re.escape(old_name) + r'(\s*\(|\s+#)',
        f'module {new_name}\\1',
        verilog_code
    )
    
    # Replace endmodule identifier if it has the module name
    modified_code = re.sub(
        r'endmodule\s+//' + re.escape(old_name),
        f'endmodule // {new_name}',
        modified_code
    )
    
    return modified_code

def run_verilog_with_testbench(verilog_code: str, testbench_code: str, 
                               expected_results: str, golden_code: str = None,
                               timeout: int = 30) -> tuple[bool, str, float]:
    """
    Run Verilog code with the provided testbench and compare results to expected output.
    First replaces the module name in the given code with the golden code's module name
    if golden code is provided.
    
    Args:
        verilog_code: The Verilog module code to test
        testbench_code: The testbench code to use for testing
        expected_results: The expected test vectors from the reference implementation
        golden_code: The golden/reference Verilog code (used to extract the correct module name)
        timeout: Maximum time in seconds to allow for compilation and simulation
        
    Returns:
        tuple: (test_passed, details, score)
    """
    import logging
    import os
    import tempfile
    import subprocess
    import re
    
    logger = logging.getLogger(__name__)
    
    try:
        # If golden code is provided, replace the module name
        if golden_code:
            # Extract module names
            golden_module_name = extract_module_name(golden_code)
            current_module_name = extract_module_name(verilog_code)
            
            if golden_module_name and current_module_name and golden_module_name != current_module_name:
                # Replace the module name in the verilog code
                verilog_code = replace_module_name(verilog_code, current_module_name, golden_module_name)
                print(f"Replaced module name '{current_module_name}' with '{golden_module_name}'")
        
        # Extract module name from the verilog code (which may have been modified above)
        module_match = re.search(r'module\s+(\w+)', verilog_code)
        if not module_match:
            return False, "Could not identify module name", 0.0
        
        module_name = module_match.group(1)
        
        # Create temporary directory for the test
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
            compile_process = subprocess.run(
                ["iverilog", "-o", output_file, verilog_file, tb_file],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=temp_dir
            )
            
            if compile_process.returncode != 0:
                error_msg = compile_process.stderr.strip()
                print(f"Compilation error: {error_msg}")
                return False, f"Compilation failed: {error_msg}", 0.0
            
            # Run simulation
            simulation_process = subprocess.run(
                ["vvp", output_file],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=temp_dir
            )
            
            if simulation_process.returncode != 0:
                error_msg = simulation_process.stderr.strip()
                print(f"Simulation error: {error_msg}")
                return False, f"Simulation failed: {error_msg}", 0.0
            
            # Check for test vector file
            test_vectors_file = os.path.join(temp_dir, "test_vectors.txt")
            if not os.path.exists(test_vectors_file):
                # If the test vectors file was not created, return failure
                print("Test vectors file was not created")
                return False, "Test vectors file was not created", 0.0
            
            # Read generated test vectors
            with open(test_vectors_file, 'r') as f:
                generated_vectors = f.read().strip()
            
            # If no test vectors were generated, return failure
            if not generated_vectors:
                print("No test vectors were generated")
                return False, "No test vectors were generated", 0.0
            
            # Parse generated test vectors
            generated_lines = generated_vectors.strip().split('\n')
            
            # Parse expected test vectors
            expected_lines = expected_results.strip().split('\n')
            
            # Verify vector count
            if len(generated_lines) < 1:
                print("Too few test vectors generated")
                return False, "Too few test vectors generated", 0.0
            
            # Compare vectors for functional equivalence
            matching = 0
            mismatches = []
            
            # Normalize and compare test vectors
            # First, preprocess both vectors to handle formatting differences
            processed_generated = []
            processed_expected = []
            
            # Process generated vectors
            for line in generated_lines:
                if line.strip():  # Skip empty lines
                    # Normalize whitespace and remove any comments
                    normalized = re.sub(r'\s+', ' ', line.strip())
                    normalized = re.sub(r'//.*$', '', normalized).strip()
                    processed_generated.append(normalized)
            
            # Process expected vectors
            for line in expected_lines:
                if line.strip():  # Skip empty lines
                    # Normalize whitespace and remove any comments
                    normalized = re.sub(r'\s+', ' ', line.strip())
                    normalized = re.sub(r'//.*$', '', normalized).strip()
                    processed_expected.append(normalized)
            
            # Now compare the processed vectors
            # If the lengths differ, we'll compare what we can
            compare_len = min(len(processed_generated), len(processed_expected))
            
            # No vectors to compare
            if compare_len == 0:
                print("No valid vectors available for comparison")
                return False, "No valid vectors available for comparison", 0.0
            
            # Compare each vector line by line
            for i in range(compare_len):
                gen_vector = processed_generated[i]
                exp_vector = processed_expected[i]
                
                # Try to split the vectors into components (may be space or tab separated)
                gen_components = gen_vector.split()
                exp_components = exp_vector.split()
                
                # If the number of components matches, compare each component
                if len(gen_components) == len(exp_components):
                    components_match = True
                    for j in range(len(gen_components)):
                        # Try to handle hex/binary/decimal format differences
                        gen_val = gen_components[j]
                        exp_val = exp_components[j]
                        
                        # Try to convert both to integers for numeric comparison
                        try:
                            # Convert values to integers, handling different formats
                            gen_int = int(gen_val, 0) if '0x' in gen_val or '0b' in gen_val else int(gen_val)
                            exp_int = int(exp_val, 0) if '0x' in exp_val or '0b' in exp_val else int(exp_val)
                            
                            if gen_int != exp_int:
                                components_match = False
                                break
                        except ValueError:
                            # If not numeric values, compare as strings
                            if gen_val != exp_val:
                                components_match = False
                                break
                    
                    if components_match:
                        matching += 1
                    else:
                        if len(mismatches) < 10:
                            mismatches.append(f"Line {i+1}: Expected '{exp_vector}', Got '{gen_vector}'")
                else:
                    # Different number of components
                    if len(mismatches) < 10:
                        mismatches.append(f"Line {i+1}: Expected {len(exp_components)} components '{exp_vector}', Got {len(gen_components)} components '{gen_vector}'")
            
            # Calculate match percentage
            match_percentage = (matching / compare_len) * 100
            print(f"Match percentage: {match_percentage:.2f}% ({matching}/{compare_len} vectors match)")
            
            # Determine score based on match percentage
            # Scale from 0 to 1 based on percentage match
            score = matching / compare_len
            
            # Generate detailed report
            details = (
                f"Test vector comparison results:\n"
                f"Total test vectors compared: {compare_len}\n"
                f"Matching vectors: {matching} ({match_percentage:.2f}%)\n"
                f"Non-matching vectors: {compare_len - matching}\n"
                f"Generated vectors length: {len(processed_generated)}\n"
                f"Expected vectors length: {len(processed_expected)}\n"
            )
            
            # Add length mismatch warning if significant
            if abs(len(processed_generated) - len(processed_expected)) > 5:
                details += f"WARNING: Significant difference in vector count between generated and expected results!\n"
            
           
            # Add mismatch details if available
            if mismatches:
                print("mismatches:", mismatches)
                details += "Mismatch details:\n" + "\n".join(mismatches)
                if len(mismatches) == 10 and compare_len - matching > 10:
                    details += f"\n...and {compare_len - matching - 10} more mismatches"
            
            # Test passes if match percentage is 99% or higher
            test_passed = match_percentage >= 99
            
            print(f"Test passed: {test_passed}")
            return test_passed, details, score
            
    except subprocess.TimeoutExpired:
        print(f"Process timed out after {timeout} seconds")
        return False, f"Process timed out after {timeout} seconds", 0.0
    except Exception as e:
        print(f"Error during functional testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, f"Error during functional testing: {str(e)}", 0.0


def run_ast_comparison(generated_code, golden_code) -> tuple[bool, str, float]:
    """Compare the AST of generated code with golden code to determine similarity.
    
    Args:
        generated_code: The Verilog module code to evaluate
        golden_code: The reference Verilog code to compare against
        
    Returns:
        tuple: (comparison_passed, details, score)
    """
    import tempfile
    import subprocess
    import os
    import re
    import ast
    import logging
    import difflib
    
    logger = logging.getLogger(__name__)
    
    try:
        # Helper function to extract AST-relevant statements
        def extract_ast_statements(code):
            # Ensure code is a string
            if not isinstance(code, str):
                if isinstance(code, list):
                    code = '\n'.join(str(item) for item in code)
                else:
                    code = str(code)
            
            # Remove comments
            code_without_comments = re.sub(r'//.*?$|/\*.*?\*/', '', code, flags=re.MULTILINE|re.DOTALL)
            
            # Extract relevant structural elements - we'll use these to compare the ASTs
            statements = []
            
            # Module declaration
            module_match = re.search(r'module\s+(\w+)\s*\((.*?)\);', code_without_comments, re.DOTALL)
            if module_match:
                module_name = module_match.group(1)
                port_list = module_match.group(2)
                statements.append(f"module {module_name}")
                
                # Extract ports
                cleaned_ports = re.sub(r'\s+', ' ', port_list).strip()
                ports = [p.strip() for p in cleaned_ports.split(',')]
                for port in ports:
                    statements.append(f"port {port}")
            
            # Input/output declarations
            for io_type in ['input', 'output', 'inout']:
                io_matches = re.finditer(r'{}\s+(\[.*?\])?\s*(\w+)\s*;'.format(io_type), code_without_comments)
                for match in io_matches:
                    width = match.group(1) if match.group(1) else ""
                    name = match.group(2)
                    statements.append(f"{io_type} {width} {name}")
            
            # Wire/reg declarations
            for var_type in ['wire', 'reg']:
                var_matches = re.finditer(r'{}\s+(\[.*?\])?\s*(\w+)\s*;'.format(var_type), code_without_comments)
                for match in var_matches:
                    width = match.group(1) if match.group(1) else ""
                    name = match.group(2)
                    statements.append(f"{var_type} {width} {name}")
            
            # Assign statements
            assign_matches = re.finditer(r'assign\s+(\w+)\s*=\s*(.*?);', code_without_comments)
            for match in assign_matches:
                lhs = match.group(1)
                rhs = re.sub(r'\s+', ' ', match.group(2)).strip()
                statements.append(f"assign {lhs} = {rhs}")
            
            # Always blocks (simplified)
            always_blocks = re.findall(r'always\s*@\s*\((.*?)\)(.*?)(?=always|endmodule|$)', 
                                       code_without_comments, re.DOTALL)
            for i, (sensitivity, block_content) in enumerate(always_blocks):
                sensitivity = re.sub(r'\s+', ' ', sensitivity).strip()
                statements.append(f"always@({sensitivity})")
                
                # Extract if statements within the always block
                if_matches = re.finditer(r'if\s*\((.*?)\)(.*?)(?=else|end|$)', block_content, re.DOTALL)
                for match in if_matches:
                    condition = re.sub(r'\s+', ' ', match.group(1)).strip()
                    statements.append(f"if({condition})")
                
                # Extract else statements
                else_matches = re.findall(r'else(.*?)(?=end|$)', block_content, re.DOTALL)
                for else_block in else_matches:
                    statements.append("else")
                
                # Extract case statements
                case_matches = re.finditer(r'case\s*\((.*?)\)(.*?)endcase', block_content, re.DOTALL)
                for match in case_matches:
                    case_expr = re.sub(r'\s+', ' ', match.group(1)).strip()
                    statements.append(f"case({case_expr})")
                    
                    # Extract case items
                    case_items = re.finditer(r'(\w+|default)\s*:(.*?)(?=\w+\s*:|default\s*:|endcase|$)', 
                                            match.group(2), re.DOTALL)
                    for case_item in case_items:
                        item = case_item.group(1).strip()
                        statements.append(f"case_item {item}")
            
            # Instantiations (module instances)
            instance_matches = re.finditer(r'(\w+)\s+(\w+)\s*\((.*?)\);', code_without_comments, re.DOTALL)
            for match in instance_matches:
                module_type = match.group(1)
                instance_name = match.group(2)
                statements.append(f"instance {module_type} {instance_name}")
            
            return statements
        
        # Ensure both codes are strings before processing
        if not isinstance(generated_code, str):
            if isinstance(generated_code, list):
                generated_code = '\n'.join(str(item) for item in generated_code)
            else:
                generated_code = str(generated_code)
                
        if not isinstance(golden_code, str):
            if isinstance(golden_code, list):
                golden_code = '\n'.join(str(item) for item in golden_code)
            else:
                golden_code = str(golden_code)
        
        # Extract AST-like representations
        generated_statements = extract_ast_statements(generated_code)
        golden_statements = extract_ast_statements(golden_code)
        
        # Calculate similarity
        matcher = difflib.SequenceMatcher(None, generated_statements, golden_statements)
        similarity = matcher.ratio()
        
        # Calculate unique statements in each
        generated_set = set(generated_statements)
        golden_set = set(golden_statements)
        
        # Calculate coverage (how many of the golden statements are present)
        if golden_set:
            coverage = len(generated_set.intersection(golden_set)) / len(golden_set)
        else:
            coverage = 0.0
            
        # Calculate redundancy (extra statements not in golden)
        if len(generated_set) > 0:
            redundancy = len(generated_set - golden_set) / len(generated_set)
        else:
            redundancy = 1.0  # Maximum redundancy if no statements
            
        # Combined score (weighted average of similarity and coverage)
        score = 0.7 * similarity + 0.3 * coverage - 0.2 * redundancy
        
        # Limit score to range [0, 1]
        score = max(0.0, min(1.0, score))
        
        details = (f"AST similarity: {similarity:.2f}, Coverage: {coverage:.2f}, "
                  f"Redundancy: {redundancy:.2f}, Final score: {score:.2f}")
        
        # Consider test passed if score is above threshold
        passed = score >= 0.7
        
        return passed, details, score
        
    except Exception as e:
        logger.error(f"Error during AST comparison: {e}")
        return False, f"Error during AST comparison: {str(e)}", 0.0

def verilog_format_reward(completions, **kwargs):
        
        pattern = rf"<think>[\s\S]*?</think>\s*<answer>[\s\S]*?```verilog[\s\S]*?```[\s\S]*?</answer>"
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.search(pattern, content, re.DOTALL) for content in completion_contents]
        return [0.1 if match else 0.0 for match in matches]

# Update the verilog_reward function to use the new run_verilog_with_testbench function with golden_code parameter
def verilog_reward(
    completions: List[List[dict]], 
    golden_code: Optional[List[str]] = None,
    testbench_code: Optional[List[str]] = None,
    testbench_result: Optional[List[str]] = None,
    **kwargs
) -> List[float]:
    """
    Enhanced reward function that checks syntax, AST similarity, and functional correctness.
    
    Args:
        completions: List of model completions
        golden_code: Reference Verilog code for each completion
        testbench_code: Testbench code for functional testing
        testbench_result: Expected test vectors from reference implementation
        
    Returns:
        List of reward scores between 0 and 1
    """
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    # Handle missing golden code
    if golden_code is None:
        logger.warning("No golden code provided, will only check syntax")
        golden_code = [None] * len(contents)
    elif isinstance(golden_code, str):
        golden_code = [golden_code] * len(contents)
    
    # Handle missing testbench code
    if testbench_code is None:
        logger.warning("No testbench code provided, will not perform functional testing")
        testbench_code = [None] * len(contents)
    elif isinstance(testbench_code, str):
        testbench_code = [testbench_code] * len(contents)
    
    # Handle missing testbench results
    if testbench_result is None:
        logger.warning("No testbench results provided, will not perform functional testing")
        testbench_result = [None] * len(contents)
    elif isinstance(testbench_result, str):
        testbench_result = [testbench_result] * len(contents)
    
    # Process each completion
    for i, content in enumerate(contents):
        try:
            # Extract Verilog code from completion
            code = extract_verilog_code(content)
            
            # If code is too short or empty, give zero reward
            if not code or len(code) < 20:  # Too short to be a valid module
                rewards.append(0.0)
                logger.info(f"Completion {i}: Code too short, reward = 0.0")
                continue
            
            # Check if it contains module declaration
            if "module" not in code:
                rewards.append(0.0)
                logger.info(f"Completion {i}: No module declaration, reward = 0.0")
                continue
            
            # Initialize reward components
            syntax_score = 0.0
            ast_similarity_score = 0.0
            functional_score = 0.0
            
            # 1. Check syntax using iverilog
            syntax_ok, syntax_details, syntax_score = check_syntax_with_iverilog(code)
            if not syntax_ok:
                logger.info(f"Completion {i}: Syntax check failed, reward = 0.0")
                rewards.append(0.0)
                continue
            
            # 2. Compare AST with golden code if available
            if golden_code[i] is not None:
                ast_ok, ast_details, ast_similarity_score = run_ast_comparison(code, golden_code[i])
                logger.info(f"Completion {i}: AST comparison - {ast_details}")
            else:
                ast_ok, ast_details = False, "No golden code provided"
            
            # 3. Run functional testing if testbench is available
            if (testbench_code[i] is not None and 
                testbench_result[i] is not None and 
                testbench_code[i] and 
                testbench_result[i]):
                
                # Pass golden_code to run_verilog_with_testbench for module name replacement
                func_ok, func_details, functional_score = run_verilog_with_testbench(
                    code, testbench_code[i], testbench_result[i], golden_code=golden_code[i]
                )
                logger.info(f"Completion {i}: Functional testing - {func_details}")
            else:
                func_ok, func_details = False, "No testbench or results provided"
            
            # Calculate weighted final reward
            # If syntax fails, entire reward is 0
            if func_ok:
                reward = 2.1
            elif syntax_ok:
                reward = (
                    0.1 * syntax_score + 
                    1 * ast_similarity_score
                )
            else:
                reward = 0.0
            
            print(f"reward: {reward}")
            rewards.append(reward)
            
            # Log details for debugging
            logger.info(
                f"Completion {i}: syntax_score={syntax_score:.2f}, "
                f"ast_similarity_score={ast_similarity_score:.2f}, "
                f"functional_score={functional_score:.2f}, "
                f"final_reward={reward:.2f}"
            )
            
        except Exception as e:
            logger.warning(f"Error in verilog_reward for completion {i}: {e}")
            rewards.append(0.0)  # Zero reward on error
    
    return rewards