import os
import json
import ast
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import radon.complexity as complexity # For cc_visit
import radon.raw as raw # For analyze
import radon.metrics as metrics # For h_visit, mi_visit
# from radon.metrics import HalsteadMetrics
import math # Needed for math.log2 in Halstead Volume calculation
import json # For pretty-printing JSON output
import pandas as pd # Import pandas

ErrorLogs = []
p = []
pp = []
ee = []

def export_array_of_arrays_to_excel(data, excel_filename="output.xlsx", sheet_name="Sheet1"):
    """
    Exports a Python array of arrays (or list of lists) to an Excel file,
    with each inner array's elements in separate columns.

    Args:
        data (list of lists): The input data, where each inner list represents a row.
        excel_filename (str): The name of the Excel file to create.
        sheet_name (str): The name of the sheet within the Excel file.
    """
    if not data:
        print("Input data is empty. No Excel file will be created.")
        return

    # Create a pandas DataFrame from the list of lists
    # Each inner list becomes a row in the DataFrame
    df = pd.DataFrame(data)

    # Export the DataFrame to an Excel file
    try:
        df.to_excel(excel_filename, index=False, sheet_name=sheet_name)
        print(f"Data successfully exported to '{excel_filename}' on sheet '{sheet_name}'.")
    except Exception as e:
        print(f"An error occurred during Excel export: {e}")

def get_halstead_derived_metrics(code_snippet_for_halstead):
    """
    Calculates Halstead metrics and derived values for a given code snippet.
    Returns a dict of metrics or None if analysis fails.
    """
    try:
        halstead_report = metrics.h_visit(code_snippet_for_halstead)

        n1 = halstead_report.h1
        n2 = halstead_report.h2
        N1 = halstead_report.N1
        N2 = halstead_report.N2

        program_vocabulary = n1 + n2
        program_length = N1 + N2
        
        volume = program_length * (math.log2(program_vocabulary)) if program_vocabulary > 1 else 0
        difficulty = (n1 / 2) * (N2 / n2) if n2 > 0 else 0
        effort = difficulty * volume
        
        bug_fix_time_seconds = effort / 18
        bug_fix_time_minutes = bug_fix_time_seconds / 60
        bug_fix_time_hours = bug_fix_time_minutes / 60

        return {
            "Halstead_UniqueOperators_n1": n1,
            "Halstead_UniqueOperands_n2": n2,
            "Halstead_TotalOperators_N1": N1,
            "Halstead_TotalOperands_N2": N2,
            "Halstead_ProgramVocabulary": program_vocabulary,
            "Halstead_ProgramLength": program_length,
            "Halstead_Volume": volume,
            "Halstead_Difficulty": effort, # Note: Effort is often used as "difficulty" in some contexts
            "Halstead_Effort": effort, # Re-added for clarity
            "Halstead_EstimatedBugFixTime_Sec": bug_fix_time_seconds,
            "Halstead_EstimatedBugFixTime_Min": bug_fix_time_minutes,
            "Halstead_EstimatedBugFixTime_Hrs": bug_fix_time_hours
        }
    except Exception:
        return None

# --- Helper Function to Extract HalsteadReport Values ---
def extract_halstead_report_values(halstead_report_obj):
    """Extracts all attribute values from a radon.metrics.HalsteadReport object (namedtuple)."""
    if hasattr(halstead_report_obj, '_fields'): # Check if it's a namedtuple
        values = {field_name: getattr(halstead_report_obj, field_name)
                  for field_name in halstead_report_obj._fields}
        return values
    return {} # Return empty if not a valid HalsteadReport

# --- Helper Function to Extract RawMetrics Values ---
def extract_raw_metrics_values(raw_metrics_obj):
    """Extracts all attribute values from a radon.raw.RawMetrics object (namedtuple)."""
    if hasattr(raw_metrics_obj, '_fields'): # RawMetrics is also a namedtuple
        values = {f"Raw_{field_name.upper()}": getattr(raw_metrics_obj, field_name)
                  for field_name in raw_metrics_obj._fields}
        return values
    return {} # Return empty if not a valid RawMetrics

# --- Helper Function to Extract Cyclomatic Complexity (CC) Values ---
def extract_cc_values(cc_results_list):
    """
    Extracts and aggregates values from a list of Cyclomatic Complexity results.
    Returns a dict with total complexity and details for each component.
    """
    if not isinstance(cc_results_list, list):
        return {} # Expect a list of Function/Class/Method objects

    extracted_cc_details = []
    total_complexity = 0
    for cc_item in cc_results_list:
        # radon.complexity results are namedtuples (Function, Class, Method)
        # They have attributes like .name, .complexity, .lineno, .type etc.
        if hasattr(cc_item, 'complexity') and hasattr(cc_item, 'name'):
            extracted_cc_details.append({
                "name": cc_item.name,
                "type": getattr(cc_item, 'type', 'function_or_method'), # 'type' can be 'function', 'method', 'class'
                "lineno": cc_item.lineno,
                "complexity": cc_item.complexity
            })
            total_complexity += cc_item.complexity
    
    return {
        "Total_Cyclomatic_Complexity": total_complexity,
        "Details": extracted_cc_details
    }


def get_halstead_derived_metrics(code_snippet):
    """
    Analyzes a Python code snippet using Radon to extract Halstead metrics
    and then calculates Effort, Time (Bug-Fix Time), and Difficulty.

    Args:
        code_snippet (str): The Python code to analyze.

    Returns:
        dict: A dictionary containing the Halstead metrics and derived metrics,
              or None if analysis fails.
    """
    try:
        # Use h_visit to get the HalsteadReport namedtuple
        # This function directly takes the code string.
        halstead_report = h_visit(code_snippet)

        n1 = halstead_report.h1 # Unique operators
        n2 = halstead_report.h2 # Unique operands
        N1 = halstead_report.N1 # Total operators
        N2 = halstead_report.N2 # Total operands

        # Derived Halstead metrics
        program_vocabulary = n1 + n2
        program_length = N1 + N2
        
        # Volume (V)
        # Handle cases where vocabulary is 0 or 1 to avoid math domain errors
        if program_vocabulary > 1:
            volume = program_length * (math.log2(program_vocabulary))
        else:
            volume = 0 # If vocabulary is 0 or 1, volume is 0.

        # Difficulty (D)
        # Avoid division by zero if n2 (unique operands) is 0
        difficulty = (n1 / 2) * (N2 / n2) if n2 > 0 else 0

        # Effort (E)
        effort = difficulty * volume

        # Time (T) - often referred to as "Bug-Fix Time" or "Development Time"
        # Assuming 18 instructions per second (Stroud number)
        bug_fix_time_seconds = effort / 18
        bug_fix_time_minutes = bug_fix_time_seconds / 60
        bug_fix_time_hours = bug_fix_time_minutes / 60

        return {
            "Unique Operators (n1)": n1,
            "Unique Operands (n2)": n2,
            "Total Operators (N1)": N1,
            "Total Operands (N2)": N2,
            "Program Vocabulary (n)": program_vocabulary,
            "Program Length (N)": program_length,
            "Halstead Volume (V)": volume,
            "Halstead Difficulty (D)": difficulty,
            "Halstead Effort (E)": effort,
            "Estimated Bug-Fix Time (seconds)": bug_fix_time_seconds,
            "Estimated Bug-Fix Time (minutes)": bug_fix_time_minutes,
            "Estimated Bug-Fix Time (hours)": bug_fix_time_hours
        }
    except Exception as e:
        # print(f"Error during Halstead analysis: {e}") # You might want to log this
        return None

def extract_code_from_var(CodeVar):
    content = CodeVar
    code_lines = [line for line in content.split('\n') if not is_code_line(line)]

    code_only_content = '\n'.join(code_lines)
 

    return code_only_content

def is_code_line(line):
    try:
        ast.parse(line)
        return True
    except:
        return False

def remove_triple_backticks(text):
    """Removes all triple backtick characters (```) from a string.

    Args:
        text (str): The input string.

    Returns:
        str: The string with all triple backticks removed.
    """
    return text.replace("```", "")

def remove_python_word(input_string):
    """
    Removes all occurrences of the word "python" (case-insensitive) from a given string.
    It handles word boundaries, meaning it won't remove "python" from words like "Jupyter".

    Args:
        input_string (str): The string from which to remove "python".

    Returns:
        str: The new string with "python" removed.
    """
    # Use the re (regular expression) module for powerful string replacement.
    import re

    # re.compile for efficiency if used multiple times, but for a single call,
    # re.sub is sufficient directly.

    # Pattern:
    # r"..." makes it a raw string, useful for regex to avoid backslash issues.
    # r"\bpython\b" :
    #   \b  : Word boundary. This ensures we match the whole word "python"
    #         and not "python" within "Jupyter" or "micropython".
    #   python: The literal string to match.
    # re.IGNORECASE: Makes the match case-insensitive (e.g., "Python", "PYTHON", "python" will all be matched).
    
    # re.sub(pattern, replacement, string, count=0, flags=0)
    # pattern: The regex pattern to search for.
    # replacement: The string to replace matches with (an empty string to remove it).
    # input_string: The string to perform the replacement on.
    # flags=re.IGNORECASE: Apply case-insensitive matching.
    
    modified_string = re.sub(r"\bpython\b", "", input_string, flags=re.IGNORECASE)
    
    # After removal, there might be double spaces if "python" was between two words.
    # This step replaces one or more spaces with a single space and strips leading/trailing spaces.
    modified_string = re.sub(r"\s+", " ", modified_string).strip()

    return modified_string

# def remove_first_and_last_lines(text):
#     """
#     Removes the first and last lines from a given string.

#     Args:
#         text (str): The input string.

#     Returns:
#         str: The string with the first and last lines removed.
#               Returns an empty string if the input has less than 3 lines.
#     """
#     lines = text.splitlines()
#     if len(lines) <= 1:
#         return ""  # Return empty string if there are fewer than 3 lines
#     return "\n".join(lines[1:-1])

def extract_method_name_from_json(code_string):
    """
    Extracts the method name from a JSON file containing a list of code snippets.

    Args:
        code_string (str): A string containing the python code.

    Returns:
        str: the name of the method
    """
    method_name = None
    try:
        try:
            # Parse the code string into an AST (Abstract Syntax Tree).
            tree = ast.parse(code_string)

            # Iterate through the top-level nodes in the AST.
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # If a FunctionDef node is found, append its name to the list.
                    method_name = node.name
                    break  # Add this break to only extract the first function def

        except SyntaxError:
            print(f"SyntaxError in code: {code_string[:20]}...")  # print first 20 chars of code

    except Exception as e:
        print(f"Error:  {e}")
        return None  # Handle invalid JSON input.

    return method_name


def GetFromLLM(promptQuestion, code_from_json):
    r = ""
    try:
        # llm = Ollama(model="mistral")
        # llm = Ollama(model="gemma2:2b")
        # llm = Ollama(model="phi3:3.8b")
        llm = Ollama(model="codellama:latest")
        

        prompt = PromptTemplate(
            input_variables=["question"],
            template="Question: {question}",
        )

        chain = LLMChain(llm=llm, prompt=prompt)

        question = f"""
        {promptQuestion} The method name must be called {extract_method_name_from_json(code_from_json)} and show only code and do not enter any other text or indentation or any backtick or grave accent. Make the code at least four lines. """
        print("$$$$$$$")
        print(question)
        response = chain.run(question)

        # print("^^^^^^^^")
        # print(response)
        # print("^^^^^^^^")

        

        # r = extract_python_code(response)
        # r = response
        
        r  = remove_triple_backticks(response)
        print(r)
        print(("$$$$$$$"))
        # export_to_txt(response, "python_solve_5.txt")
    except ValueError as e:
        print(f"Error: {e}")
        print("Ensure Ollama is running and the LLM model is pulled.")
    except ImportError as e:
        print(f"Error: {e}")
        print("Ensure that langchain and the ollama python library are installed.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return r

def GetDescriptionFromLLM(generated_code):
    try:
        # llm = Ollama(model="mistral")
        # llm = Ollama(model="gemma2:2b")
        # llm = Ollama(model="phi3:3.8b")
        llm = Ollama(model="codellama:latest")

        prompt = PromptTemplate(
            input_variables=["question"],
            template="Question: {question}",
        )

        chain = LLMChain(llm=llm, prompt=prompt)

        question = f"""
        In no more than two sentences, describe what the following code do: {generated_code}"""
        print("$$$$$$$")
        print(question)
        response = chain.run(question)
        print(response)
        print(("$$$$$$$"))
        # export_to_txt(response, "python_solve_5.txt")
    except ValueError as e:
        print(f"Error: {e}")
        print("Ensure Ollama is running and the LLM model is pulled.")
    except ImportError as e:
        print(f"Error: {e}")
        print("Ensure that langchain and the ollama python library are installed.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return response

def GetSummarizeLLM(BugDesc):
    try:
        # llm = Ollama(model="mistral")
        # llm = Ollama(model="gemma2:2b")
        # llm = Ollama(model="phi3:3.8b")
        llm = Ollama(model="codellama:latest")

        

        prompt = PromptTemplate(
            input_variables=["question"],
            template="Question: {question}",
        )

        chain = LLMChain(llm=llm, prompt=prompt)

        question = f"""
        In no more than 20 words, give a title for the following bug description: {BugDesc}"""
        print("$$$$$$$")
        print(question)
        response = chain.run(question)
        print(response)
        print(("$$$$$$$"))
        # export_to_txt(response, "python_solve_5.txt")
    except ValueError as e:
        print(f"Error: {e}")
        print("Ensure Ollama is running and the LLM model is pulled.")
    except ImportError as e:
        print(f"Error: {e}")
        print("Ensure that langchain and the ollama python library are installed.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return response


def execute_tests_from_json(file_path):
    """
    Reads a JSON file, extracts code and test cases, and executes the tests.

    Args:
        file_path (str): The path to the JSON file.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {file_path}")
        return

    for item in data:
        prompt = item.get("prompt")
        code_from_json = item.get("code")
        task_id = item.get("task_id")
        code_to_execute = GetFromLLM(prompt, code_from_json)  # Get the code from LLM

        # code_to_execute = remove_first_and_last_lines(code_to_execute)
        code_to_execute = remove_python_word(code_to_execute)

        test_list = item.get("test_list", [])

        if prompt:
            print(f"\n--- Task: {prompt} ---")

        if code_to_execute:
            try:
                FailedCases = 0
                PassedCases = 0
                ErrorTestCases = 0
                Pass1Score = 0
                # Parse and execute the response
                tree = ast.parse(code_to_execute)
                for node in tree.body:
                    if isinstance(node, ast.FunctionDef):
                        # Extract the function name
                        func_name = node.name
                        # Compile the function definition
                        compiled_func = compile(ast.unparse(node), '<string>', 'exec')
                        # Execute the compiled function in the current scope
                        print("task_id " + str(task_id))
                        exec(compiled_func, globals())
                        # *IMPORTANT*: Call the function by its name.
                        if test_list:
                            print("Running tests:")
                            for test_case in test_list:
                                # Replace the function name in the test case with the actual name.
                                test_case = test_case.replace(func_name, func_name)
                                try:
                                    exec(test_case, globals())
                                    print(f"  ✅ {test_case}")
                                    PassedCases+=1
                                except AssertionError:
                                    print(f"  ❌ {test_case}")
                                    FailedCases+=1
                                except Exception as e:
                                    print(f"  🔥 Error executing '{test_case}': {e}")
                                    ErrorTestCases+=1
                            if(PassedCases>0 and FailedCases==0 and ErrorTestCases==0):
                                Pass1Score = 1
                            else:
                                Pass1Score = 0
                            p.append([str(task_id), PassedCases, FailedCases, ErrorTestCases, code_to_execute, code_from_json, Pass1Score, prompt])
                        else:
                            print("No test cases found for this task.")
                        break  # Stop after the first function definition.

            except Exception as e:
                print("task_id " + str(task_id))
                for test_case in test_list:
                    print(test_case)
                print(f"Error executing code:\n{code_to_execute}\nError: {e}")
                ErrorLogs.append([str(task_id),code_to_execute,test_list, code_from_json])
                ee.append([str(task_id)])
                # print("Correcting code from error..")
                # print(extract_code_from_var(str(code_to_execute)))
                # print("finished correcting..")

        else:
            print("No code found for this task.")


if __name__ == "__main__":
    json_file_path = 'sanitized-mbpp-2.json'  # Replace with the actual path to your JSON file
    execute_tests_from_json(json_file_path)
    print(ErrorLogs)
    print("------")
    print("printing correct code")
    print(p)
    print("------")
    print("calculating radon")
    c,m,r,h,f = [],[],[],[],[]
    for i in p:
        print(i[4])
        code_block_sample = i[4]
        # Compute Cyclomatic Complexity
        print("--- Generating Sample Radon Objects ---")
        # Get actual Radon objects for demonstration
        cc_results_sample = complexity.cc_visit(code_block_sample)
        mi_result_sample = metrics.mi_visit(code_block_sample, multi=True)
        raw_metrics_sample = raw.analyze(code_block_sample)
        halstead_report_sample = metrics.h_visit(code_block_sample)

        # --- Construct `my_array` with various Radon objects ---
        # This array simulates mixed types of Radon outputs you might process.
        my_array = [
            # Direct Radon objects
            halstead_report_sample,
            mi_result_sample,
            raw_metrics_sample,
            cc_results_sample, # This is a list of Function/Class objects

            # A nested structure similar to your original example, but now with different types
            [('halstead_nested', halstead_report_sample)],
            [('raw_nested', raw_metrics_sample)],
            [('cc_nested', cc_results_sample)],
            [('mi_nested', mi_result_sample)],
            
            # An example of invalid data
            "This is just a string, not a Radon object",
            [('bad_nested', {'key': 'value'})] # Nested, but contains a non-Radon dict
        ]

        print("Original `my_array` structure (simulated mixed types):")
        for item in my_array:
            print(item)
        print("-" * 50)

        # --- Main Loop to Process and Extract All Metric Types ---
        extracted_data = []

        for idx, item in enumerate(my_array):
            current_extracted_item = {
                "Input_Index": idx,
                "Input_Type": str(type(item)),
                "Extracted_Metrics": {} # This will hold the structured metrics
            }
            
            # --- Check and process each type of Radon object ---

            # 1. Handle radon.metrics.HalsteadReport
            # We use type(halstead_report_sample) to get the exact Radon class type
            if isinstance(item, type(halstead_report_sample)):
                print(f"\nProcessing direct HalsteadReport object at index {idx}:")
                current_extracted_item["Metrics_Category"] = "HalsteadReport"
                current_extracted_item["Extracted_Metrics"] = extract_halstead_report_values(item)
                current_extracted_item["Derived_Halstead_Metrics"] = get_halstead_derived_metrics(code_block_sample) # Re-compute for full derived set

            # 2. Handle float (Maintainability Index)
            elif isinstance(item, float):
                print(f"\nProcessing Maintainability Index (float) at index {idx}:")
                current_extracted_item["Metrics_Category"] = "MaintainabilityIndex"
                current_extracted_item["Extracted_Metrics"] = {"Maintainability_Index": item}

            # 3. Handle radon.raw.RawMetrics
            elif isinstance(item, type(raw_metrics_sample)):
                print(f"\nProcessing RawMetrics object at index {idx}:")
                current_extracted_item["Metrics_Category"] = "RawMetrics"
                current_extracted_item["Extracted_Metrics"] = extract_raw_metrics_values(item)

            # 4. Handle list of Cyclomatic Complexity results (e.g., [Function(...), Method(...)])
            # Check if it's a list and if its first element is a Radon Function/Class/Method object
            elif isinstance(item, list) and item and isinstance(item[0], type(cc_results_sample[0])):
                print(f"\nProcessing Cyclomatic Complexity results at index {idx}:")
                current_extracted_item["Metrics_Category"] = "CyclomaticComplexity"
                current_extracted_item["Extracted_Metrics"] = extract_cc_values(item)

            # 5. Handle nested list/tuple structures like [('label', RadonObject)]
            elif isinstance(item, list) and item and isinstance(item[0], tuple) and len(item[0]) > 1:
                print(f"\nProcessing nested list/tuple structure at index {idx}:")
                label, nested_obj = item[0]
                current_extracted_item["Metrics_Category"] = "NestedStructure"
                current_extracted_item["Nested_Label"] = label
                
                # Check type of the nested object
                if isinstance(nested_obj, type(halstead_report_sample)):
                    current_extracted_item["Extracted_Metrics"]["Nested_Halstead"] = extract_halstead_report_values(nested_obj)
                    current_extracted_item["Extracted_Metrics"]["Nested_Derived_Halstead"] = get_halstead_derived_metrics(code_block_sample)
                elif isinstance(nested_obj, float):
                    current_extracted_item["Extracted_Metrics"]["Nested_Maintainability_Index"] = nested_obj
                elif isinstance(nested_obj, type(raw_metrics_sample)):
                    current_extracted_item["Extracted_Metrics"]["Nested_Raw_Metrics"] = extract_raw_metrics_values(nested_obj)
                elif isinstance(nested_obj, list) and nested_obj and isinstance(nested_obj[0], type(cc_results_sample[0])):
                    current_extracted_item["Extracted_Metrics"]["Nested_Cyclomatic_Complexity"] = extract_cc_values(nested_obj)
                else:
                    # Handle unknown nested object types
                    print(f"  Warning: Unrecognized nested object type within tuple: {type(nested_obj)}")
                    current_extracted_item["Extracted_Metrics"]["Unrecognized_Nested_Item"] = str(nested_obj)

            # 6. Handle any other unrecognized top-level item types
            else:
                print(f"\nSkipping unrecognized top-level item at index {idx} (Type: {type(item)}).")
                current_extracted_item["Metrics_Category"] = "Unrecognized"
                current_extracted_item["Extracted_Metrics"] = {"Original_Value": str(item)}

            extracted_data.append(current_extracted_item)

        print("\n" + "="*50)
        print("           FINAL STRUCTURED METRICS DATA (JSON-like)")
        print("="*50)
        # Use json.dumps for pretty printing and proper JSON formatting
        print(json.dumps(extracted_data, indent=2))
        

        f.append([i,json.dumps(extracted_data)])

    print("------")
    print("printing error code")
    print(ee)
    print("------")
    print("------")
    print("printing final p")
    print(f)
    print("------")
    OUTPUT_CSV_FILENAME = 'filtered_combined_metrics.csv'

    # --- Define the categories you want to include in the final output ---
    DESIRED_CATEGORIES = [
        "HalsteadReport",
        "MaintainabilityIndex",
        "RawMetrics",
        "CyclomaticComplexity"
        # You can add "NestedStructure" here if you want those metrics included too
    ]

    all_combined_records = [] # This will store one flattened dictionary per Record_ID

    # --- Process each entry in the 'f' list ---
    for record_id_data, json_string_data in f:
        # 1. Extract the primary identifier from 'record_id_data'
        # Assuming the first element of the first inner list is the ID you want
        record_id = record_id_data[0] if isinstance(record_id_data, list) and record_id_data else 'N/A'

        # Initialize a dictionary to hold all combined metrics for the current record_id
        current_record_metrics = {
            'Record_ID': record_id
        }

        # 2. Parse the JSON string back into a Python list of dictionaries
        try:
            extracted_data_list = json.loads(json_string_data)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON for record ID '{record_id}': {e}")
            continue # Skip to the next record if JSON is invalid

        # 3. Iterate through each metric dictionary within the parsed JSON data
        for item_dict in extracted_data_list:
            category = item_dict.get('Metrics_Category') or item_dict.get('Metrics_Category') # Check both potential keys

            # Check if the category is one of the desired ones
            if category in DESIRED_CATEGORIES:
                
                # Extract and flatten 'Extracted_Metrics'
                if 'Extracted_Metrics' in item_dict and isinstance(item_dict['Extracted_Metrics'], dict):
                    for k, v in item_dict['Extracted_Metrics'].items():
                        # Special handling for "Details" in Cyclomatic Complexity:
                        # Convert list of dicts to JSON string for a single CSV cell
                        if k == "Details" and isinstance(v, list):
                            current_record_metrics[f"{category}_Details"] = json.dumps(v)
                        elif k.startswith("Raw_"): # Raw metrics already prefixed
                            current_record_metrics[k] = v
                        elif category == "HalsteadReport" and (k == "total" or k == "functions"):
                            # Handle 'total' and 'functions' list from HalsteadReport
                            current_record_metrics[f"{category}_{k}"] = json.dumps(v)
                        else:
                            # Prefix with category for clarity in the final flat row
                            current_record_metrics[f"{category}_{k}"] = v

                # Extract and flatten 'Derived_Halstead_Metrics' (if present)
                if 'Derived_Halstead_Metrics' in item_dict and isinstance(item_dict['Derived_Halstead_Metrics'], dict):
                    for k, v in item_dict['Derived_Halstead_Metrics'].items():
                        current_record_metrics[f"Derived_Halstead_{k}"] = v # Prefix with 'Derived_Halstead_'

                # Handle specific cases like Maintainability Index directly at the top level
                if category == "MaintainabilityIndex" and "Maintainability_Index" in item_dict['Extracted_Metrics']:
                    current_record_metrics["Maintainability_Index"] = item_dict['Extracted_Metrics']['Maintainability_Index']
                
                # For "NestedStructure" category, flatten its content if desired and filter
                if category == "NestedStructure":
                    # Check for nested keys like "Nested_Halstead", "Nested_Raw_Metrics", etc.
                    for nested_metric_key, nested_metric_value in item_dict['Extracted_Metrics'].items():
                        if isinstance(nested_metric_value, dict):
                            # For example, Nested_Halstead or Nested_Raw_Metrics
                            for sub_k, sub_v in nested_metric_value.items():
                                current_record_metrics[f"Nested_{nested_metric_key}_{sub_k}"] = sub_v
                        elif isinstance(nested_metric_value, (float, int, str)):
                            # For example, Nested_Maintainability_Index
                            current_record_metrics[f"Nested_{nested_metric_key}"] = nested_metric_value
                        elif isinstance(nested_metric_value, list):
                            # For example, Nested_Cyclomatic_Complexity
                            # Convert list of dicts to JSON string
                            current_record_metrics[f"Nested_{nested_metric_key}"] = json.dumps(nested_metric_value)

            # If it's an unrecognized category, you can choose to include it or skip it
            elif category == "Unrecognized" or category is None:
                # Optionally add a column for unrecognized content
                if "Original_Value" in item_dict.get('Extracted_Metrics', {}):
                    current_record_metrics["Unrecognized_Content"] = item_dict['Extracted_Metrics']['Original_Value']
                # Also capture the Nested_Label if it's an unrecognized nested item
                if item_dict.get('Nested_Label'):
                    current_record_metrics["Unrecognized_Nested_Label"] = item_dict['Nested_Label']
                if "Unrecognized_Nested_Item" in item_dict.get('Extracted_Metrics', {}):
                    current_record_metrics["Unrecognized_Nested_Item_Content"] = item_dict['Extracted_Metrics']['Unrecognized_Nested_Item']


        # Add the final combined dictionary for this record_id to the list
        all_combined_records.append(current_record_metrics)

    # --- Convert to Pandas DataFrame and Export to CSV ---
    print("\n" + "="*50)
    print(f"           EXPORTING {len(all_combined_records)} RECORDS TO CSV")
    print("="*50)

    if all_combined_records:
        # Use json_normalize on the list of *already combined* dictionaries
        # This will create a single row per Record_ID
        final_df = pd.json_normalize(all_combined_records, sep='_')

        # Optional: Further clean up column names or reorder
        # This part depends on the exact output column names after flattening
        final_df.columns = final_df.columns.str.replace('HalsteadReport_', '')
        final_df.columns = final_df.columns.str.replace('MaintainabilityIndex_', '')
        final_df.columns = final_df.columns.str.replace('RawMetrics_', '')
        final_df.columns = final_df.columns.str.replace('CyclomaticComplexity_', '')
        final_df.columns = final_df.columns.str.replace('Metrics_', '') # Remove the general 'Metrics_' prefix

        # Reorder columns to have Record_ID first (if not already)
        cols = final_df.columns.tolist()
        if 'Record_ID' in cols:
            cols.insert(0, cols.pop(cols.index('Record_ID')))
        final_df = final_df[cols]

        # Export to CSV
        final_df.to_csv(OUTPUT_CSV_FILENAME, index=False)
        print(f"\nSuccessfully exported combined and filtered metrics to '{OUTPUT_CSV_FILENAME}'")
    else:
        print("No valid data to export to CSV.")

    print("printing p again")
    for i in p:
        print(i)

    countError = 0

    export_array_of_arrays_to_excel(p, excel_filename="Test_cases_data.xlsx")

    print("printing error code again")
    for i in ee:
        print(i)
        countError+=1

    print(f'Error count {countError}')
    export_array_of_arrays_to_excel(ee, excel_filename="error_data.xlsx")
    export_array_of_arrays_to_excel(ErrorLogs, excel_filename="error_data_2.xlsx")

if os.path.isfile("/Users/workingDir/Test_cases_data.xlsx"):
    df = pd.read_excel('Test_cases_data.xlsx')
    a = []
    for index, row in df.iterrows():
        print(f"Index: {row[0]}, 4:{row[4]}, 7: {row[7]}")
        jsonQuestion = row[7]
        generated_code = row[4]
        DescribeLLM = GetDescriptionFromLLM(generated_code)
        a.append([row[0], generated_code, jsonQuestion , DescribeLLM])
    print(a)
    export_array_of_arrays_to_excel(a, excel_filename="code_LLM_describe.xlsx")

# if os.path.isfile("/Users/workingDir/mozilla_firefox_sample.csv"):
#     df = pd.read_csv('mozilla_firefox_sample.csv')
#     k = []
#     for index, row in df.iterrows():
#         print(f"Index: {row[0]}, 4:{row[4]}, 7: {row[5]}")
#         BugTitle = row[4]
#         BugDesc = row[5]
#         SummarizeLLM = GetSummarizeLLM(BugDesc)
#         k.append([row[0], BugTitle , BugDesc, SummarizeLLM])
#     print(k)
#     export_array_of_arrays_to_excel(k, excel_filename="Mozilla_things.xlsx")
    


