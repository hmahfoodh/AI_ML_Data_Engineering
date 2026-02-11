import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
import warnings


# --- Plotting Imports ---
import matplotlib.pyplot as plt
import seaborn as sns
# ------------------------


# Suppress ConvergenceWarning from MLPClassifier for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)


def create_combined_data(base_scores, task_data_list):
   """
   Combines a set of base scores (representing average performance)
   with task-specific data into a single DataFrame.
   """
   metrics = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore F1"]
  
   # Create a list of dictionaries for the base scores
   base_scores_list = []
   # Use an instance_id that is distinct from the task data's instance_ids
   # This assumes task_data_list is not empty
   max_instance_id = max(d['instance_id'] for d in task_data_list) if task_data_list else -1
   base_instance_id = max_instance_id + 1
  
   for model, scores in base_scores.items():
       # Ensure base scores are aligned with metric names
       row = {'instance_id': base_instance_id, 'model': model}
       row.update({metric: value for metric, value in zip(metrics, scores)})
       base_scores_list.append(row)
  
   # Combine the base scores with the task data
   combined_data = base_scores_list + task_data_list
  
   return pd.DataFrame(combined_data)


def create_pairwise_data(df):
   """
   Converts the dataset into a pairwise comparison format for RankNet.
   This version uses an INNER MERGE on 'instance_id' to correctly align
   data even if sample counts or instance IDs differ between models.
   """
   models = df['model'].unique()
   metrics = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore F1"]
  
   pairwise_data = []
  
   # Generate all unique pairs of models
   for i in range(len(models)):
       for j in range(i + 1, len(models)):
           model_A = models[i]
           model_B = models[j]
          
           # Extract data for the two models
           df_A_raw = df[df['model'] == model_A]
           df_B_raw = df[df['model'] == model_B]
          
           # CRITICAL FIX: Merge on 'instance_id' using 'inner' join.
           merged_df = pd.merge(
               df_A_raw[['instance_id'] + metrics],
               df_B_raw[['instance_id'] + metrics],
               on='instance_id',
               how='inner',
               suffixes=('_A', '_B')
           )
          
           # Now iterate over the perfectly aligned, common instances
           for idx in range(len(merged_df)):
               row = merged_df.iloc[idx]
              
               # Get the features for Model A and Model B from the merged row
               features_A = row[[f'{m}_A' for m in metrics]].values
               features_B = row[[f'{m}_B' for m in metrics]].values
              
               # The input features for RankNet are the differences in metrics
               diffs = features_A - features_B
              
               # The target is 1 if A > B, 0 if B > A. We define "better" as a higher average score.
               avg_A = np.mean(features_A)
               avg_B = np.mean(features_B)
              
               if avg_A > avg_B:
                   label = 1
               elif avg_B > avg_A:
                   label = 0
               else:
                   # Skip if there's a tie
                   continue
                  
               pairwise_data.append({'features': diffs, 'label': label})


   # Prepare data for sklearn
   X = np.array([p['features'] for p in pairwise_data])
   y = np.array([p['label'] for p in pairwise_data])
  
   return X, y, metrics


def run_ranknet_gridsearch(X, y):
   """
   Sets up and runs GridSearchCV on an MLPClassifier to find the best
   hyperparameters for the RankNet model.
   """
   print("\nStep 3: Running RankNet with GridSearchCV...")
  
   # The RankNet model is an MLPClassifier
   model = MLPClassifier(random_state=42, max_iter=2000)
  
   # Hyperparameter grid to search
   param_grid = {
       'hidden_layer_sizes': [(50,), (100,), (50, 25)],
       'activation': ['relu', 'tanh'],
       'solver': ['adam'],
       'alpha': [0.0001, 0.001],
   }
  
   # Use GridSearchCV to find the best model
   # cv=3 for a reasonable cross-validation, n_jobs=-1 to use all cores
   grid_search = GridSearchCV(model, param_grid, cv=3, verbose=1, n_jobs=-1, scoring='accuracy')
   grid_search.fit(X, y)
  
   print("\n--- GridSearchCV Results ---")
   print(f"Best parameters found: {grid_search.best_params_}")
   print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
  
   return grid_search.best_estimator_


def create_comparison_matrix(best_model, df):
   """
   Generates a matrix showing the probability of each model winning
   against all other models, based on the trained RankNet.
  
   This version uses an INNER MERGE on 'instance_id' for correct alignment.
   """
   models = sorted(df['model'].unique())
   metrics = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore F1"]
  
   # Initialize the matrix with zeros
   prob_matrix = pd.DataFrame(np.zeros((len(models), len(models))), index=models, columns=models)
  
   print("\nStep 4: Generating and displaying the pairwise comparison matrix...")
   print("Matrix shows P(Row_Model > Column_Model)")
  
   for i, model_A in enumerate(models):
       for j, model_B in enumerate(models):
           if i == j:
               continue
          
           # Get the raw data for these two models
           df_A_raw = df[df['model'] == model_A]
           df_B_raw = df[df['model'] == model_B]
          
           # CRITICAL FIX: Merge on 'instance_id' using 'inner' join.
           # This ensures we only predict on instances where both models have results.
           merged_df = pd.merge(
               df_A_raw[['instance_id'] + metrics],
               df_B_raw[['instance_id'] + metrics],
               on='instance_id',
               how='inner',
               suffixes=('_A', '_B')
           )
          
           # Get features for Model A and Model B from the merged dataframe
           features_A = merged_df[[f'{m}_A' for m in metrics]].values
           features_B = merged_df[[f'{m}_B' for m in metrics]].values


           # Prepare test data (the differences)
           X_test_diffs = features_A - features_B
          
           # Predict the probability of model_A winning over model_B
           if len(X_test_diffs) == 0:
               avg_win_prob_A_vs_B = np.nan # Cannot compare if no common instances exist
           else:
               # predict_proba returns [P(winner=0), P(winner=1)]
               win_probs = best_model.predict_proba(X_test_diffs)
              
               # The average probability of A winning over B across all *common* instances
               # The second column (index 1) is the probability of a positive label (win for model A)
               avg_win_prob_A_vs_B = np.mean(win_probs[:, 1])


           prob_matrix.loc[model_A, model_B] = avg_win_prob_A_vs_B
          
   # Set the diagonal to NaN for clarity
   np.fill_diagonal(prob_matrix.values, np.nan)
   print(prob_matrix.round(4))
   return prob_matrix


def derive_final_ranking(prob_matrix):
   """
   Derives a final ranking by summing the win probabilities for each model.
   """
   models = prob_matrix.index
   ranking_scores = {}
  
   # A simple way to rank is to sum up the win probabilities against all other models
   for model in models:
       # Sum of probabilities of this model winning against all others, skipping NaN (no common data)
       ranking_scores[model] = prob_matrix.loc[model].sum(skipna=True)
  
   # Sort models based on their ranking score
   final_ranking = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
  
   print("\nStep 5: Final Model Ranking")
   for rank, (model, score) in enumerate(final_ranking):
       print(f"Rank {rank+1}: {model} (Win Score: {score:.4f})")
      
   return final_ranking # Return the sorted list for plotting


def plot_ranking_results(final_ranking):
   """
   Plots the final model ranking scores using seaborn.
   """
   models = [item[0] for item in final_ranking]
   scores = [item[1] for item in final_ranking]


   # Convert to DataFrame for easier seaborn plotting
   df_ranking = pd.DataFrame({'Model': models, 'Win Score': scores})
  
   print("\nStep 6: Plotting final ranking results...")
  
   plt.figure(figsize=(10, 6))
   # Use a clear color palette, plot models vs aggregated win score
   sns.barplot(x='Win Score', y='Model', data=df_ranking, palette='viridis')
  
   plt.title('Final RankNet Model Ranking (Sum of Win Probabilities)', fontsize=16)
   plt.xlabel('Aggregated Win Score (Sum of P(Model > Other Models))', fontsize=12)
   plt.ylabel('Model', fontsize=12)
   plt.gca().invert_yaxis() # Highest rank (highest score) on top
   plt.grid(axis='x', linestyle='--', alpha=0.6)
  
   # Add score labels to bars
   for index, score in enumerate(scores):
       # Place text slightly to the right of the bar end
       plt.text(score + 0.01, index, f'{score:.4f}', va='center')


   plt.tight_layout()
   plt.show()


if __name__ == "__main__":
   # Step 1: Combine base scores with task data
   print("Step 1: Combining base scores with provided task data...")
  
   # Define fixed base scores (average values) for each model across all metrics
   base_scores = {
       'M1': np.array([0.75, 0.72, 0.78, 0.81]),
       'M2': np.array([0.70, 0.68, 0.73, 0.76]),
       'M3': np.array([0.65, 0.63, 0.68, 0.71]),
       'M4': np.array([0.60, 0.58, 0.63, 0.66]),
   }
  
   # Example data simulating non-aligned and different counts:
   # M4 is missing instance_id 9 and has an extra instance_id 10.
   your_data_list = [
       # Instance 0
       {'instance_id': 0, 'model': 'M1', 'ROUGE-1': 0.76, 'ROUGE-2': 0.71, 'ROUGE-L': 0.79, 'BERTScore F1': 0.82},
       # {'instance_id': 0, 'model': 'M2', 'ROUGE-1': 0.71, 'ROUGE-2': 0.69, 'ROUGE-L': 0.74, 'BERTScore F1': 0.77},
       # {'instance_id': 0, 'model': 'M3', 'ROUGE-1': 0.68, 'ROUGE-2': 0.65, 'ROUGE-L': 0.70, 'BERTScore F1': 0.72},
       # {'instance_id': 0, 'model': 'M4', 'ROUGE-1': 0.61, 'ROUGE-2': 0.59, 'ROUGE-L': 0.64, 'BERTScore F1': 0.68},
      
       # Instance 1-8 (Common instances)
       {'instance_id': 1, 'model': 'M1', 'ROUGE-1': 0.74, 'ROUGE-2': 0.73, 'ROUGE-L': 0.78, 'BERTScore F1': 0.80},
       {'instance_id': 1, 'model': 'M2', 'ROUGE-1': 0.69, 'ROUGE-2': 0.67, 'ROUGE-L': 0.72, 'BERTScore F1': 0.75},
       {'instance_id': 1, 'model': 'M3', 'ROUGE-1': 0.64, 'ROUGE-2': 0.62, 'ROUGE-L': 0.67, 'BERTScore F1': 0.69},
       {'instance_id': 1, 'model': 'M4', 'ROUGE-1': 0.58, 'ROUGE-2': 0.57, 'ROUGE-L': 0.62, 'BERTScore F1': 0.65},


       # {'instance_id': 2, 'model': 'M1', 'ROUGE-1': 0.75, 'ROUGE-2': 0.70, 'ROUGE-L': 0.77, 'BERTScore F1': 0.83},
       # {'instance_id': 2, 'model': 'M2', 'ROUGE-1': 0.72, 'ROUGE-2': 0.71, 'ROUGE-L': 0.76, 'BERTScore F1': 0.78},
       # {'instance_id': 2, 'model': 'M3', 'ROUGE-1': 0.67, 'ROUGE-2': 0.66, 'ROUGE-L': 0.71, 'BERTScore F1': 0.73},
       # {'instance_id': 2, 'model': 'M4', 'ROUGE-1': 0.59, 'ROUGE-2': 0.60, 'ROUGE-L': 0.65, 'BERTScore F1': 0.67},
      
       {'instance_id': 3, 'model': 'M1', 'ROUGE-1': 0.77, 'ROUGE-2': 0.74, 'ROUGE-L': 0.80, 'BERTScore F1': 0.84},
       {'instance_id': 3, 'model': 'M2', 'ROUGE-1': 0.70, 'ROUGE-2': 0.69, 'ROUGE-L': 0.75, 'BERTScore F1': 0.76},
       # {'instance_id': 3, 'model': 'M3', 'ROUGE-1': 0.66, 'ROUGE-2': 0.64, 'ROUGE-L': 0.69, 'BERTScore F1': 0.70},
       # {'instance_id': 3, 'model': 'M4', 'ROUGE-1': 0.62, 'ROUGE-2': 0.61, 'ROUGE-L': 0.66, 'BERTScore F1': 0.69},


       {'instance_id': 4, 'model': 'M1', 'ROUGE-1': 0.78, 'ROUGE-2': 0.75, 'ROUGE-L': 0.81, 'BERTScore F1': 0.85},
       {'instance_id': 4, 'model': 'M2', 'ROUGE-1': 0.73, 'ROUGE-2': 0.71, 'ROUGE-L': 0.77, 'BERTScore F1': 0.79},
       {'instance_id': 4, 'model': 'M3', 'ROUGE-1': 0.69, 'ROUGE-2': 0.67, 'ROUGE-L': 0.72, 'BERTScore F1': 0.74},
       {'instance_id': 4, 'model': 'M4', 'ROUGE-1': 0.63, 'ROUGE-2': 0.60, 'ROUGE-L': 0.65, 'BERTScore F1': 0.68},


       {'instance_id': 5, 'model': 'M1', 'ROUGE-1': 0.79, 'ROUGE-2': 0.76, 'ROUGE-L': 0.82, 'BERTScore F1': 0.86},
       {'instance_id': 5, 'model': 'M2', 'ROUGE-1': 0.74, 'ROUGE-2': 0.72, 'ROUGE-L': 0.78, 'BERTScore F1': 0.80},
       {'instance_id': 5, 'model': 'M3', 'ROUGE-1': 0.70, 'ROUGE-2': 0.68, 'ROUGE-L': 0.73, 'BERTScore F1': 0.75},
       {'instance_id': 5, 'model': 'M4', 'ROUGE-1': 0.64, 'ROUGE-2': 0.61, 'ROUGE-L': 0.66, 'BERTScore F1': 0.69},


       {'instance_id': 6, 'model': 'M1', 'ROUGE-1': 0.80, 'ROUGE-2': 0.77, 'ROUGE-L': 0.83, 'BERTScore F1': 0.87},
       {'instance_id': 6, 'model': 'M2', 'ROUGE-1': 0.75, 'ROUGE-2': 0.73, 'ROUGE-L': 0.79, 'BERTScore F1': 0.81},
       {'instance_id': 6, 'model': 'M3', 'ROUGE-1': 0.71, 'ROUGE-2': 0.69, 'ROUGE-L': 0.74, 'BERTScore F1': 0.76},
       {'instance_id': 6, 'model': 'M4', 'ROUGE-1': 0.65, 'ROUGE-2': 0.62, 'ROUGE-L': 0.67, 'BERTScore F1': 0.70},


       {'instance_id': 7, 'model': 'M1', 'ROUGE-1': 0.81, 'ROUGE-2': 0.78, 'ROUGE-L': 0.84, 'BERTScore F1': 0.88},
       {'instance_id': 7, 'model': 'M2', 'ROUGE-1': 0.76, 'ROUGE-2': 0.74, 'ROUGE-L': 0.80, 'BERTScore F1': 0.82},
       {'instance_id': 7, 'model': 'M3', 'ROUGE-1': 0.72, 'ROUGE-2': 0.70, 'ROUGE-L': 0.75, 'BERTScore F1': 0.77},
       {'instance_id': 7, 'model': 'M4', 'ROUGE-1': 0.66, 'ROUGE-2': 0.63, 'ROUGE-L': 0.68, 'BERTScore F1': 0.71},


       {'instance_id': 8, 'model': 'M1', 'ROUGE-1': 0.82, 'ROUGE-2': 0.79, 'ROUGE-L': 0.85, 'BERTScore F1': 0.89},
       {'instance_id': 8, 'model': 'M2', 'ROUGE-1': 0.77, 'ROUGE-2': 0.75, 'ROUGE-L': 0.81, 'BERTScore F1': 0.83},
       {'instance_id': 8, 'model': 'M3', 'ROUGE-1': 0.73, 'ROUGE-2': 0.71, 'ROUGE-L': 0.76, 'BERTScore F1': 0.78},
       {'instance_id': 8, 'model': 'M4', 'ROUGE-1': 0.67, 'ROUGE-2': 0.64, 'ROUGE-L': 0.69, 'BERTScore F1': 0.72},
      
       # Instance 9 (M4 is missing this instance)
       {'instance_id': 9, 'model': 'M1', 'ROUGE-1': 0.83, 'ROUGE-2': 0.80, 'ROUGE-L': 0.86, 'BERTScore F1': 0.90},
       {'instance_id': 9, 'model': 'M2', 'ROUGE-1': 0.78, 'ROUGE-2': 0.76, 'ROUGE-L': 0.82, 'BERTScore F1': 0.84},
       {'instance_id': 9, 'model': 'M3', 'ROUGE-1': 0.74, 'ROUGE-2': 0.72, 'ROUGE-L': 0.77, 'BERTScore F1': 0.79},


       # Instance 10 (Only M4 has this instance, should be ignored by inner merge)
       {'instance_id': 10, 'model': 'M4', 'ROUGE-1': 0.60, 'ROUGE-2': 0.57, 'ROUGE-L': 0.62, 'BERTScore F1': 0.65},
   ]


   df_data = create_combined_data(base_scores, your_data_list)
  
   print(f"Loaded data for {len(df_data['instance_id'].unique())} unique instances and {len(df_data['model'].unique())} models.")
  
   # Step 2: Generate pairwise training data
   print("\nStep 2: Generating pairwise training data...")
   X_pairwise, y_pairwise, metrics = create_pairwise_data(df_data)
   print(f"Created {len(X_pairwise)} pairwise samples with {len(metrics)} features each.")
  
   # Split the pairwise data for training and testing the GridSearch
   X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_pairwise, y_pairwise, test_size=0.2, random_state=42)
  
   # Step 3: Run RankNet with GridSearch
   best_ranknet_model = run_ranknet_gridsearch(X_train_p, y_train_p)
  
   # Step 4: Create and display pairwise comparison matrix
   comparison_matrix = create_comparison_matrix(best_ranknet_model, df_data)
  
   # Step 5: Derive and display the final ranking
   final_ranking = derive_final_ranking(comparison_matrix)
  
   # Step 6: Plot the final ranking results
   plot_ranking_results(final_ranking)
