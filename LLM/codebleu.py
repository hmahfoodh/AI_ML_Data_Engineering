from collections import Counter
import math
import re # Import the regular expression module
import pandas as pd

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


def calculate_ngram_precision(reference_ngrams, candidate_ngrams, n):
    """Calculates precision for a given n-gram length."""
    common_ngrams = 0
    candidate_ngram_counts = Counter(candidate_ngrams)
    reference_ngram_counts = Counter(reference_ngrams)

    for ngram, count in candidate_ngram_counts.items():
        common_ngrams += min(count, reference_ngram_counts[ngram])

    # Ensure division by zero is handled
    return common_ngrams / sum(candidate_ngram_counts.values()) if sum(candidate_ngram_counts.values()) > 0 else 0

def get_ngrams(tokens, n):
    """Generates n-grams from a list of tokens."""
    # Handle cases where tokens list is too short for n-grams
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def simple_code_bleu_like(reference_code, candidate_code, max_n=4, weights=None):
    """
    A simplified BLEU-like score for code.
    This example only considers lexical matching (n-grams of tokens)
    with improved tokenization.
    """
    # --- IMPROVED TOKENIZATION ---
    # Regex to split on whitespace and also capture punctuation/operators
    # This pattern captures words, numbers, and common code symbols as separate tokens.
    # It handles spaces, but also keeps symbols like (),:+- etc.
    tokenizer_pattern = re.compile(r'\b\w+\b|[^a-zA-Z0-9_ \t\n\r\f\v]')
    
    reference_tokens = [token for token in tokenizer_pattern.findall(reference_code) if token.strip()]
    candidate_tokens = [token for token in tokenizer_pattern.findall(candidate_code) if token.strip()]
    # --- END IMPROVED TOKENIZATION ---

    if not weights:
        weights = [1/max_n] * max_n # Uniform weights

    brevity_penalty = 1.0
    # Calculate brevity penalty. Add a small epsilon to avoid division by zero.
    if len(candidate_tokens) < len(reference_tokens) and len(candidate_tokens) > 0:
        brevity_penalty = math.exp(1 - len(reference_tokens) / len(candidate_tokens))
    elif len(candidate_tokens) == 0:
        return 0.0 # If candidate is empty, score is 0

    # Calculate n-gram precisions
    precision_scores = []
    for n in range(1, max_n + 1):
        ref_ngrams = get_ngrams(reference_tokens, n)
        cand_ngrams = get_ngrams(candidate_tokens, n)
        
        # If candidate n-grams are empty, precision for this n is 0.
        # This is crucial for avoiding issues with shorter candidates.
        if not cand_ngrams:
            precision_scores.append(0.0)
        else:
            precision_scores.append(calculate_ngram_precision(ref_ngrams, cand_ngrams, n))

    # Combine precisions (geometric mean for BLEU)
    # The issue was often here: if ANY precision_score is 0, the log (below) will be -infinity.
    # To avoid this, we can return 0 immediately if any precision is 0.
    
    # BLEU uses log-sum for stability, then exponentiates.
    # If any precision is 0, the final BLEU score is 0.
    log_precision_sum = 0.0
    for i, p in enumerate(precision_scores):
        if p == 0:
            return 0.0 # If any precision is 0, the overall score is 0
        log_precision_sum += weights[i] * math.log(p)

    geometric_mean = math.exp(log_precision_sum)
    
    return brevity_penalty * geometric_mean

# --- Example Usage ---


# Load the workbook
df = pd.read_excel('Test_cases_data.xlsx')
a = []
for index, row in df.iterrows():
    print(f"Index: {row[0]}, 4: {row[4]}, 5: {row[5]}")
    # Reference code snippet
    # reference_code_1 = "def add_numbers(a, b): return a + b"
    # candidate_code_1_good = "def add_n(a, c): return a + c"
    reference_code_1 = row[5]
    candidate_code_1_good = row[4]

    # print("--- Example 1: `add_numbers` ---")
    # print(f"Reference: '{reference_code_1}'")

    score_good = simple_code_bleu_like(reference_code_1, candidate_code_1_good)
    print(f"Candidate score: '{candidate_code_1_good}' - Score: {score_good:.4f}")
    print("-" * 30)
    a.append([row[0],score_good])

df = pd.read_excel('error_data_2.xlsx')
b = []
for index, row in df.iterrows():
    print(f"Index: {row[0]}, 1: {row[1]}, 3: {row[3]}")
    # Reference code snippet
    # reference_code_1 = "def add_numbers(a, b): return a + b"
    # candidate_code_1_good = "def add_n(a, c): return a + c"
    reference_code_1 = row[3]
    candidate_code_1_good = row[1]

    # print("--- Example 1: `add_numbers` ---")
    # print(f"Reference: '{reference_code_1}'")

    score_good = simple_code_bleu_like(reference_code_1, candidate_code_1_good)
    print(f"Candidate score: '{candidate_code_1_good}' - Score: {score_good:.4f}")
    print("-" * 30)
    b.append([row[0],score_good])

print(a)
print(b)

export_array_of_arrays_to_excel(a, excel_filename="codebleu-passed.xlsx")
export_array_of_arrays_to_excel(b, excel_filename="codebleu-error.xlsx")




