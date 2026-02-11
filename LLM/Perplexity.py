import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForCausalLM, AutoModelForCausalLM
import math
import os
import torch # Need torch for PyTorch models
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

# Suppress logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# TensorFlow Perplexity Calculator
def calculate_perplexity_tf(code_string, model_name, window_size=512, step_size=256, try_from_pt=False):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        if try_from_pt:
            print(f"  (TF) Attempting to load {model_name} from PyTorch weights (requires torch installed).")
            model = TFAutoModelForCausalLM.from_pretrained(model_name, from_pt=True) 
        else:
            print(f"  (TF) Attempting to load {model_name} with native TensorFlow weights.")
            model = TFAutoModelForCausalLM.from_pretrained(model_name)

        # Tokenize the entire code string
        input_ids_full = tokenizer.encode(code_string, return_tensors='tf', add_special_tokens=False)
        
        if tf.size(input_ids_full).numpy() == 0:
            print("Warning: Tokenized input is empty.")
            return float('inf')

        # If the code is shorter than the window, just calculate perplexity for the whole code
        if tf.size(input_ids_full).numpy() <= window_size:
            outputs = model(input_ids_full, labels=input_ids_full)
            loss = outputs.loss
            return math.exp(loss.numpy())

        # Sliding window approach for perplexity calculation
        nlls = [] # Negative Log Likelihoods
        num_tokens = tf.size(input_ids_full).numpy()

        for i in range(0, num_tokens - window_size + 1, step_size):
            input_ids_chunk = input_ids_full[:, i : i + window_size]
            
            # Ensure labels are also just the chunk
            outputs = model(input_ids_chunk, labels=input_ids_chunk)
            
            # The loss returned by Hugging Face models is already the average negative log-likelihood
            # for the tokens in the sequence.
            nlls.append(outputs.loss.numpy())

        if not nlls:
            return float('inf') # Should not happen if num_tokens > window_size
            
        # Average the negative log likelihoods
        avg_nll = sum(nlls) / len(nlls)
        perplexity = math.exp(avg_nll)
        return perplexity

    except Exception as e:
        print(f"  (TF) An error occurred: {e}")
        print(f"  (TF) Make sure '{model_name}' has TensorFlow weights available, or can be converted from PyTorch (if try_from_pt=True).")
        return None

# PyTorch Perplexity Calculator (New function)
def calculate_perplexity_pt(code_string, model_name, window_size=512, step_size=256):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Load PyTorch model
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.eval() # Set model to evaluation mode (disable dropout, etc.)

        # Tokenize the entire code string
        input_ids_full = tokenizer.encode(code_string, return_tensors='pt', add_special_tokens=False)

        if input_ids_full.numel() == 0: # Check if tensor is empty
            print("Warning: Tokenized input is empty.")
            return float('inf')

        # If the code is shorter than the window, just calculate perplexity for the whole code
        if input_ids_full.numel() <= window_size:
            with torch.no_grad(): # Disable gradient calculation for inference
                outputs = model(input_ids_full, labels=input_ids_full)
                loss = outputs.loss 
            return math.exp(loss.item()) # Use .item() for PyTorch scalar

        # Sliding window approach for perplexity calculation
        nlls = [] # Negative Log Likelihoods
        num_tokens = input_ids_full.numel()

        with torch.no_grad(): # Disable gradient calculation for inference
            for i in range(0, num_tokens - window_size + 1, step_size):
                input_ids_chunk = input_ids_full[:, i : i + window_size]
                
                # Ensure labels are also just the chunk
                outputs = model(input_ids_chunk, labels=input_ids_chunk)
                
                # The loss returned by Hugging Face models is already the average negative log-likelihood
                # for the tokens in the sequence.
                nlls.append(outputs.loss.item())

        if not nlls:
            return float('inf') # Should not happen if num_tokens > window_size

        # Average the negative log likelihoods
        avg_nll = sum(nlls) / len(nlls)
        perplexity = math.exp(avg_nll)
        return perplexity

    except Exception as e:
        print(f"  (PT) An error occurred: {e}")
        print(f"  (PT) Make sure '{model_name}' has PyTorch weights available.")
        return None

df = pd.read_excel('Test_cases_data.xlsx')
GeneratedCode = []
for index, row in df.iterrows():
    TestID= row[0]
    candidate_code = row[4]
    GeneratedCode.append([TestID, candidate_code])
   
if __name__ == "__main__":
    r = []
    print(f"TensorFlow GPU available: {tf.config.list_physical_devices('GPU')}")
    print(f"PyTorch GPU available: {torch.cuda.is_available()}")

    models_to_test = [
        {"name": "microsoft/CodeGPT-small-py", "framework": "tf"},
         {"name": "gpt2", "framework": "tf"}, 
        {"name": "distilgpt2", "framework": "tf"}, # New TensorFlow-native model
    ]

    # Define window_size and step_size
    # window_size: The maximum number of tokens the model can handle at once. 
    #              Typical values are 512 or 1024. Adjust based on your model's context window.
    # step_size: How many tokens to slide the window by for each new chunk. 
    #            Smaller step_size means more overlap and potentially more stable perplexity, 
    #            but takes longer.
    default_window_size = 512 
    # default_window_size = 256
    default_step_size = 256    

    for model_info in models_to_test:
        model_name = model_info["name"]
        framework = model_info["framework"]

        print(f"\n--- Calculating Perplexity for using {model_name} ({framework}) ---")
        print(f"  Window Size: {default_window_size}, Step Size: {default_step_size}")
        for i in range (0, len(GeneratedCode)):
            ppl = None
            print(f"  Processing Test ID: {GeneratedCode[i][0]}")
            
            if framework == "tf":
                ppl = calculate_perplexity_tf(
                    GeneratedCode[i][1], 
                    model_name=model_name, 
                    window_size=default_window_size, 
                    step_size=default_step_size, 
                    try_from_pt=False
                )
            elif framework == "tf_from_pt":
                ppl = calculate_perplexity_tf(
                    GeneratedCode[i][1], 
                    model_name=model_name, 
                    window_size=default_window_size, 
                    step_size=default_step_size, 
                    try_from_pt=True
                )
            elif framework == "pt":
                ppl = calculate_perplexity_pt(
                    GeneratedCode[i][1], 
                    model_name=model_name, 
                    window_size=default_window_size, 
                    step_size=default_step_size
                )

            if ppl is not None:
                print(f"  Perplexity for Test ID {GeneratedCode[i][0]}: {ppl:.2f}")
                r.append([GeneratedCode[i][0], GeneratedCode[i][1], ppl, model_name, default_window_size, default_step_size])
            else:
                print(f"  Could not calculate perplexity for {model_name} (Test ID: {GeneratedCode[i][0]}).")

    print("\n--- Interpretation ---")
    print("Lower perplexity indicates that the code snippet is more 'expected' or 'natural'")
    print("according to the language model's training data.")
    print("Note: Perplexity does NOT indicate functional correctness or performance.")

    export_array_of_arrays_to_excel(r, excel_filename="perplexity-passed.xlsx")