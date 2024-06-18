import pandas as pd
from sklearn.metrics import average_precision_score

# Define the path to your file
file_path = '../../benchmark/DeepReg/results/AFfiltered/03_norm_Com/prediction_result.tsv'

# Read the file into a list of strings
with open(file_path, 'r') as file:
    lines = file.readlines()

# Initialize variables to store results
folds_data = []
current_fold_data = []

# Process each line to separate fold data
for line in lines:
    line = line.strip()
    if line.startswith("Fold"):
        if current_fold_data:
            folds_data.append(current_fold_data)
            current_fold_data = []
    elif line:
        current_fold_data.append(line)

# Add the last fold data if exists
if current_fold_data:
    folds_data.append(current_fold_data)

# Function to process a single fold's data
def process_fold(fold_lines):
    fold_str = "\n".join(fold_lines)
    from io import StringIO
    fold_df = pd.read_csv(StringIO(fold_str), delimiter='\t')
    fold_df.columns = fold_df.columns.str.strip()
    fold_df['score'] = fold_df['score'].astype(float)
    fold_df['exp_result'] = fold_df['exp_result'].astype(int)
    return fold_df

# Calculate average precision score for each fold
for i, fold_lines in enumerate(folds_data):
    fold_df = process_fold(fold_lines)
    y_scores = fold_df['score']
    y_true = fold_df['exp_result']
    average_precision = average_precision_score(y_true, y_scores)
    print(f'Fold {i+1} Average Precision Score: {average_precision}')