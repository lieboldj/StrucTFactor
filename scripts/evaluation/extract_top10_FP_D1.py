import pandas as pd

# load two prediction_results.tsv files
data1 = pd.read_csv('../../results/CV_AFfiltered_expTFs_clu03_norm_spatial_Com/prediction_result.tsv', sep='\t', header=1)

# read ../../results/CV_AFfiltered_expTFs_clu03_norm_seq_Com/prediction_result.tsv into data1 but remove all lines starting with Fold or sequence_ID
data2 = pd.read_csv('../../results/CV_AFfiltered_expTFs_clu03_norm_seq_Com/prediction_result.tsv', sep='\t',  header=1)

# remove entry if column prediction is not a number
data1 = data1[pd.to_numeric(data1['prediction'], errors='coerce').notna()]
data2 = data2[pd.to_numeric(data2['prediction'], errors='coerce').notna()]

#only keep rows where exp_result is 0
data1 = data1[data1['exp_result'] == "0"]
data2 = data2[data2['exp_result'] == "0"]

# order data1 and data2 by prediction
data1 = data1.sort_values(by='score', ascending=False)
data2 = data2.sort_values(by='score', ascending=False)

data1_set = set(data1.head(10)['sequence_ID'])
data2_set = set(data2.head(10)['sequence_ID'])

print("Top10 FP proteins by StrucTFactor: ", data1_set)
print("Top10 FP proteins by DeepTFactor: ",data2_set)

print("Only predicted by StrucTFactor: ", data1_set - data2_set)
print("Sanitiy check, predicted by both: ", data1_set.intersection(data2_set))
