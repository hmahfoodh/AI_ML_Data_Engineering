import math # Not directly used after removing perplexity, but kept for general utility
import os
# from langchain_community.llms import Ollama
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain

# Install these if you haven't already:
# pip install scikit-learn transformers evaluate torch

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Import evaluation libraries
import evaluate
import pandas as pd # Import pandas

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

# Load evaluators once to save time
# BLEU, ROUGE, and BERTScore are commonly used for summarization/translation tasks.
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore") # Requires transformers and torch

# --- Helper Functions for Metrics ---

def calculate_cosine_similarity(text1, text2):
    """
    Calculates cosine similarity between two texts using TF-IDF vectors.
    Returns 0.0 if either text is empty.
    """
    if not text1 or not text2:
        return 0.0
    
    vectorizer = TfidfVectorizer().fit([text1, text2])
    tfidf_text1 = vectorizer.transform([text1])
    tfidf_text2 = vectorizer.transform([text2])
    return cosine_similarity(tfidf_text1, tfidf_text2)[0][0]

def calculate_jaccard_similarity(text1, text2):
    """
    Calculates Jaccard similarity between two texts based on word sets.
    Returns 0.0 if either text is empty or no common words.
    """
    if not text1 or not text2:
        return 0.0

    set1 = set(text1.lower().split())
    set2 = set(text2.lower().split())
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    if union == 0:
        return 0.0 # Avoid division by zero if both sets are empty
    return intersection / union

# --- Main Example Usage ---

if __name__ == "__main__":
    # Define your full text and summarized text
    # In a real scenario, 'reference_text' would be a human-written summary
    # or the original document you want to compare against.
    r = []

    if os.path.isfile("/Users/Hussain/Desktop/devpython1/code_LLM_describe.xlsx"):
        df = pd.read_excel('code_LLM_describe.xlsx')
        a = []
        for index, row in df.iterrows():
            # print(f"Index: {row[0]}, 4:{row[4]}, 7: {row[7]}")
            full_text = row[2]
            summarized_text_good = row[3]
            # summarized_text_bad = "Cats bark at the moon." # Bad example (unrelated)
            
            # Let's test with 'summarized_text_good' first, comparing it to the 'full_text' as a reference.
            # For actual summarization evaluation, 'reference_text' should be a gold-standard summary.
            candidate_text_1 = summarized_text_good
            reference_text_1 = full_text # Treat full text as reference for now
            
            print(f"--- Evaluating 'Good' Summary ---")
            print(f"Full Text (Reference): \"{reference_text_1}\"")
            print(f"Candidate Summary: \"{candidate_text_1}\"")

            # Cosine Similarity
            cos_sim = calculate_cosine_similarity(candidate_text_1, reference_text_1)
            print(f"\nCosine Similarity: {cos_sim:.4f}")

            # Jaccard Similarity
            jac_sim = calculate_jaccard_similarity(candidate_text_1, reference_text_1)
            print(f"Jaccard Similarity: {jac_sim:.4f}")

            # BERTScore
            # Note: BERTScore needs a list of predictions and references.
            # model_type can be 'bert-base-uncased', 'distilbert-base-uncased', etc.
            try:
                bs_results = bertscore.compute(predictions=[candidate_text_1], references=[reference_text_1], model_type="distilbert-base-uncased")
                bert_f1 = bs_results["f1"][0] # Take the F1 score for the first (and only) prediction
                print(f"BERTScore F1: {bert_f1:.4f}")
            except Exception as e:
                print(f"Error calculating BERTScore: {e}")
                print(f"  (Hint: Ensure 'transformers' and 'torch' are installed and a valid model_type is specified.)")

            # BLEU Score
            # Note: BLEU expects a list of candidate strings and a list of lists of reference strings.
            try:
                bleu_results = bleu.compute(predictions=[candidate_text_1], references=[[reference_text_1]])
                bleu_score = bleu_results["bleu"]
                print(f"BLEU Score: {bleu_score:.4f}")
            except Exception as e:
                print(f"Error calculating BLEU: {e}")

            # ROUGE Scores
            # Note: ROUGE expects lists of strings for predictions and references.
            try:
                rouge_results = rouge.compute(predictions=[candidate_text_1], references=[reference_text_1], use_stemmer=True)
                print(f"ROUGE-1 F1: {rouge_results['rouge1']:.4f}")
                print(f"ROUGE-2 F1: {rouge_results['rouge2']:.4f}")
                print(f"ROUGE-L F1: {rouge_results['rougeL']:.4f}")
            except Exception as e:
                print(f"Error calculating ROUGE: {e}")
            a.append([row[0],row[1],row[2],row[3],cos_sim,jac_sim,bert_f1,bleu_score, rouge_results['rouge1'], rouge_results['rouge2'],rouge_results['rougeL']])

    print(a)
    export_array_of_arrays_to_excel(a, excel_filename="code_LLM_describe_scores.xlsx")


    print("\n--- Understanding the Metrics ---")
    print("- **Cosine Similarity** and **Jaccard Similarity** measure lexical (word-level) overlap. Higher values mean more common words.")
    print("- **BERTScore** measures semantic similarity using powerful language models. It's great for seeing if summaries capture the meaning, even with different words.")
    print("- **BLEU** primarily measures precision of n-grams. It tells you how much of your summary's phrases are found in the reference.")
    print("- **ROUGE** primarily measures recall of n-grams and longest common subsequences. It tells you how much of the reference's important information is covered by your summary.")
    print("\nFor summarization, a good summary typically has high BERTScore and ROUGE values.")