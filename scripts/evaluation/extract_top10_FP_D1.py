import pandas as pd

# load two prediction_results.tsv files
data1 = pd.read_csv('../../results/CV_AFfiltered_expTFs_clu03_norm_spatial0_Com/prediction_result.tsv', sep='\t', header=1)

# read ../../results/CV_AFfiltered_expTFs_clu03_norm_seq_Com/prediction_result.tsv into data1 but remove all lines starting with Fold or sequence_ID
data2 = pd.read_csv('../../results/CV_AFfiltered_expTFs_clu03_norm_seq0_Com/prediction_result.tsv', sep='\t',  header=1)

# remove entry if column prediction is not a number
data1 = data1[pd.to_numeric(data1['prediction'], errors='coerce').notna()]
data2 = data2[pd.to_numeric(data2['prediction'], errors='coerce').notna()]

#only keep rows where exp_result is 1
data1 = data1[data1['exp_result'] == "1"]
data2 = data2[data2['exp_result'] == "1"]

# order data1 and data2 by prediction
data1 = data1.sort_values(by='score', ascending=False).reset_index(drop=True)
data2 = data2.sort_values(by='score', ascending=False).reset_index(drop=True)

data1_set = set(data1.head(1)['sequence_ID'])
data2_set = set(data2.head(1)['sequence_ID'])

print("Top1 TP protein by StrucTFactor: ", data1_set)
print("Top1 TP protein by DeepTFactor: ",data2_set)

#print("Only predicted by StrucTFactor: ", data1_set - data2_set)
#print("Sanitiy check, predicted by both: ", data1_set.intersection(data2_set))

print("Score for P11470 by StrucTFactor: ", data1[data1['sequence_ID'] == 'P11470']['score'].values[0])
print("Score for P11470 by DeepTFactor: ", data2[data2['sequence_ID'] == 'P11470']['score'].values[0])


# print the top 10 TP proteins by StrucTFactor
#print("Top10 TP proteins by StrucTFactor: ")
#for i in range(10):
#    print(data1['sequence_ID'][i], data1['score'][i])

# print the top 10 TP proteins by DeepTFactor
#print("Top10 TP proteins by DeepTFactor: ")
#for i in range(10):
#    print(data2['sequence_ID'][i], data2['score'][i])
# for FP analysis
'''
# write the top 150 FP proteins by DeepTFactor to file only if score > 0.5
data2['score'] = data2['score'].astype(float)
data1['score'] = data1['score'].astype(float)
data_score2 = data2[data2['score'] > 0.5]
data_score1 = data1[data1['score'] > 0.5]
# remove last column
data_score2 = data_score2.drop(columns=['exp_result'])
data_score1 = data_score1.drop(columns=['exp_result'])
# remove prediction column
data_score2 = data_score2.drop(columns=['prediction'])
data_score1 = data_score1.drop(columns=['prediction'])

data_score2.to_csv('FP_proteins_DeepTFactor.tsv', sep='\t', index=False)
data_score1.to_csv('FP_proteins_StrucTFactor.tsv', sep='\t', index=False)

# change column score to score2
data_score2 = data_score2.rename(columns={'score': 'score2'})

# merge data_score1 and data_score2
data_score = pd.merge(data_score1, data_score2, on='sequence_ID', how='outer')
data_score.to_csv('FP_proteins_DeepTFactor_StrucTFactor.tsv', sep='\t', index=False)
'''