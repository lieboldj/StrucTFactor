import numpy as np
import pandas as pd
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns


def extract_info(text):
    info = {}
    #folds = re.findall(r'Fold: (\d+)\naccuracy: (\d+\.\d+), sensitivity: (\d+\.\d+), specificity: (\d+\.\d+), recall: (\d+\.\d+).*?TP: (\d+).*?FP: (\d+).*?TN: (\d+).*?FN: (\d+).*?ROC curve \(area = (\d+\.\d+)\).*?AUPRC \(AP\): (\d+\.\d+).*?MCC: (-?\d+\.\d+)', text, re.DOTALL)
    folds = re.findall(r'Fold: (\d+)\naccuracy: (nan|\d+\.\d+), sensitivity: (nan|\d+\.\d+), specificity: (nan|\d+\.\d+), recall: (nan|\d+\.\d+).*?TP: (\d+).*?FP: (\d+).*?TN: (\d+).*?FN: (\d+).*?ROC curve \(area = (\d+\.\d+)\).*?AUPRC \(AP\): (nan|\d+\.\d+).*?MCC: (nan|-?\d+\.\d+)', text, re.DOTALL)

    for i, fold in enumerate(folds):
        info[f'Fold {fold[0]}'] = {
            'accuracy': float(fold[1]),
            'sensitivity': float(fold[2]),
            'specificity': float(fold[3]),
            'recall': float(fold[4]),
            'TP': int(fold[5]),
            'FP': int(fold[6]),
            'TN': int(fold[7]),
            'FN': int(fold[8]),
            'AU-ROC': float(fold[9]),
            'AU-PRC': float(fold[10]),
            'MCC': float(fold[11]),
            'fold': int(fold[0])
        }
    return info

ratio = ["0.01","0.001","0.003"]
clust = ["10","30","50"]
exps = ["16", "32", "64", "128", "256"]
tfs = ["_Com"]
tf = tfs[0]
mode = ["spatial","seq","reg"]
eval_metrics = ["AU-PRC", "MCC"]

all_results = np.zeros((len(eval_metrics), 5, len(exps),len(clust),len(ratio),len(mode)))

for p, eval_metric in enumerate(eval_metrics):
    for l, mod in enumerate(mode):                       
        for e, exp in enumerate(exps):#exp = exps[0]
            for r, rat in enumerate(ratio):    
                for c, clu in enumerate(clust):  
                    j = 0
                    # Read text file
                    if os.path.exists(f'../../results/CV_AFfiltered_expTFs_clu03_norm_{mod}_{clu}_{rat}_{exp}/result_info.txt'):
                        with open(f'../../results/CV_AFfiltered_expTFs_clu03_norm_{mod}_{clu}_{rat}_{exp}/result_info.txt', 'r') as file:
                            text = file.read()
                    else:
                        #path_name = exp.split('_')[1]
                        if os.path.exists(f'../../benchmark/DeepReg/results/AFfiltered/03_norm_Com_{exp}_{clu}_{rat}/result_info.txt'):
                            with open(f'../../benchmark/DeepReg/results/AFfiltered/03_norm_Com_{exp}_{clu}_{rat}/result_info.txt', 'r') as file:
                                text = file.read()
                        else:
                            print(f'{exp}_{clu}_{rat}_{mod}')
                            continue

                    # Extract information             
                    result_info = extract_info(text)
                    num_folds = len(result_info)
                    if mod == "reg":
                        if num_folds == 0:
                            print("No results")
                            
                            for _ in range(5):
                                all_results[p,j,e,c,r,l] = 0
                                j += 1
                            continue

                    if True:#mod != "reg" or (bar_plot and rat == "large") or clu == "No" or exp == "CV_AFall_expTFs_clu":
                        for fold in result_info.values():
                            for key, value in fold.items():                   
                                #print(key, value)
                                if eval_metric == "mis":
                                    if key == "FP":
                                        tmp = value
                                    if key == "TN":
                                        all_results[p,j,e,c,r,l] = tmp/(value+tmp)

                                if key == eval_metric:
                                    all_results[p,j,e,c,r,l] = value

                            j += 1

# create array with mean values for each setting
#mean_results = np.zeros((len(eval_metrics), len(exps),len(ratio),len(mode),len(clust)))
mean_results = np.nanmean(all_results, axis=1)
#print(mean_results.shape)
#print(mean_results)
tools = ["StrucTFactor", "DeepTFactor", "DeepReg"]
index = pd.MultiIndex.from_product(
    [eval_metrics, exps, clust, ratio, tools],
    names=["Metric", "Batch_Size", "Epochs", "Learning_Rate", "Tool"]
)

# reshape mean results to -1, 12
mean_results_flatten = mean_results.flatten()
# add columns: 
# (D(rl,nr,3), D(rl,nr,5), D(rl,nr,10), D(rl,r,3), D(rl,r,5), D(rl,r,10), D(a,nr,3), D(a,nr,5), D(a,nr,10), D(a,r,3), D(a,r,5), D(a,r,10))
# add these names to mean_results as column names as pandas dataframe
#mean_results = pd.DataFrame(mean_results, columns=["D(rl,nr,3)", "D(rl,nr,5)", "D(rl,nr,10)", "D(rl,r,3)", "D(rl,r,5)", "D(rl,r,10)", "D(a,nr,3)", "D(a,nr,5)", "D(a,nr,10)", "D(a,r,3)", "D(a,r,5)", "D(a,r,10)"])
#mean_results = pd.DataFrame(mean_results, columns=["D(rl,nr,3)", "D(rl,nr,5)", "D(rl,nr,10)", "D(rl,r,3)", "D(rl,r,5)", "D(rl,r,10)", "D(a,nr,3)", "D(a,nr,5)", "D(a,nr,10)", "D(a,r,3)", "D(a,r,5)", "D(a,r,10)"])
df = pd.DataFrame(mean_results_flatten, index=index, columns=["Value"]).reset_index()
# Convert columns to appropriate data types
df["Batch_Size"] = df["Batch_Size"].astype(int)
df["Epochs"] = df["Epochs"].astype(int)
df["Learning_Rate"] = df["Learning_Rate"].astype(float)
def print_top_settings(df, metric):
    metric_df = df[df["Metric"] == metric]
    for tool in tools:
        tool_df = metric_df[metric_df["Tool"] == tool]

        top_values = tool_df.nlargest(len(tool_df), 'Value')
        print(f"Ordered {metric} values for {tool}:")
        for i, row in top_values.iterrows():
            print(f"Value: {row['Value']:.4f}, Batch Size: {row['Batch_Size']}, Epochs: {row['Epochs']}, Learning Rate: {row['Learning_Rate']}")
        print("\n")

# Print top 3 settings for MCC
print_top_settings(df, "MCC")

# Print top 3 settings for AU-PRC
print_top_settings(df, "AU-PRC")
# Plotting MCC
exit()
mcc_df = df[df["Metric"] == "MCC"]
top_scores = mcc_df.groupby("Tool")["Value"].nlargest(3)
# Set up the figure and axes for MCC
fig, axes = plt.subplots(2, 1, figsize=(15, 10))

# MCC line plot
sns.lineplot(data=mcc_df, x='Batch_Size', y='Value', hue='Tool', style='Tool', markers=True, dashes=False, ax=axes[0])
axes[0].set_title('MCC vs. Batch Size for Different Tools')
axes[0].set_xlabel('Batch Size')
axes[0].set_ylabel('MCC')

# Facet grid for MCC
g = sns.FacetGrid(mcc_df, col='Epochs', row='Learning_Rate', hue='Tool', margin_titles=True, despine=False)
g.map(sns.lineplot, 'Batch_Size', 'Value', marker='o')
g.add_legend()
g.set_axis_labels('Batch Size', 'MCC')
g.fig.suptitle('MCC for Different Tools across Epochs and Learning Rates', y=1.02)
plt.savefig("metrics_plot_MCC.png")

# Plotting AU-PRC
auprc_df = df[df["Metric"] == "AU-PRC"]

# Set up the figure and axes for AU-PRC
fig, axes = plt.subplots(2, 1, figsize=(15, 10))

# AU-PRC line plot
sns.lineplot(data=auprc_df, x='Batch_Size', y='Value', hue='Tool', style='Tool', markers=True, dashes=False, ax=axes[1])
axes[1].set_title('AU-PRC vs. Batch Size for Different Tools')
axes[1].set_xlabel('Batch Size')
axes[1].set_ylabel('AU-PRC')

# Facet grid for AU-PRC
g = sns.FacetGrid(auprc_df, col='Epochs', row='Learning_Rate', hue='Tool', margin_titles=True, despine=False)
g.map(sns.lineplot, 'Batch_Size', 'Value', marker='o')
g.add_legend()
g.set_axis_labels('Batch Size', 'AU-PRC')
g.fig.suptitle('AU-PRC for Different Tools across Epochs and Learning Rates', y=1.02)
plt.savefig("metrics_plot.png")
print(df)
exit()
with open(f'table_all_results.tex', 'w') as file:
    with open (f'table_d1_folds.tex', 'w') as file2:
        for p, metric in enumerate(eval_metrics):
            for m, mod in enumerate(mode):
                #file.write(r"\\\hline"+f"\n")
                if m == 0:
                    file.write(r"StrucTFactor &\multicolumn{1}{c|}{\textbf{")
                    file2.write(r"StrucTFactor &\multicolumn{1}{c|}{\textbf{")
                if m == 1: 
                    file.write(r"DeepTFactor &\multicolumn{1}{c|}{")
                    file2.write(r"DeepTFactor &\multicolumn{1}{c|}{")
                if m == 2:
                    file.write(r"DeepReg &\multicolumn{1}{c|}{")
                    file2.write(r"DeepReg &\multicolumn{1}{c|}{")
                for e, exp in enumerate(exps):
                    for c, clu in enumerate(clust):
                        for r, rat in enumerate(ratio):
                            if e+c+r == len(exps)+len(clust)+len(ratio)-3:
                                if m ==0:
                                    file.write(f"{mean_results[p,e,r,m,c]:.4f}"+r"}}\\\hline"+f"\n")
                                else:
                                    file.write(f"{mean_results[p,e,r,m,c]:.4f}"+r"}\\\hline"+f"\n")
                            
                            else:
                                if m == 0:
                                    file.write(f"{mean_results[p,e,r,m,c]:.4f}" + r"}} &\multicolumn{1}{c|}{\textbf{")
                                else: 
                                    file.write(f"{mean_results[p,e,r,m,c]:.4f}" + r"} &\multicolumn{1}{c|}{")
                            if e == 0 and rat == "norm":
                                if clu == "03":
                                    if m == 0:
                                        for i in range(5):
                                            if i == 3:
                                                file2.write(f"{all_results[p,i,e,r,m,c]:.4f}"+r"}} &\multicolumn{1}{c||}{\textbf{")
                                            else:
                                                file2.write(f"{all_results[p,i,e,r,m,c]:.4f}"+r"}} &\multicolumn{1}{c|}{\textbf{")
                                        file2.write(f"{mean_results[p,e,r,m,c]:.4f}"+r"}}\\\hline"+f"\n")
                                    else:
                                        for i in range(5):
                                            if i == 3:
                                                file2.write(f"{all_results[p,i,e,r,m,c]:.4f}" + r"} &\multicolumn{1}{c||}{")
                                            else:
                                                file2.write(f"{all_results[p,i,e,r,m,c]:.4f}" + r"} &\multicolumn{1}{c|}{")
                                        file2.write(f"{mean_results[p,e,r,m,c]:.4f}"+r"}\\\hline"+f"\n")
