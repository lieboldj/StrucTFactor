import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re


def extract_info(text):
    info = {}
    folds = re.findall(r'Fold: (\d+).*?accuracy: (\d+\.\d+), sensitivity: (\d+\.\d+), specificity: (\d+\.\d+), recall: (\d+\.\d+).*?TP: (\d+).*?FP: (\d+).*?TN: (\d+).*?FN: (\d+).*?ROC curve \(area = (\d+\.\d+)\).*?AUPRC \(AP\): (\d+\.\d+).*?MCC: (\d+\.\d+)', text, re.DOTALL)
    for i, fold in enumerate(folds):
        info[f'Fold {i}'] = {
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
            'MCC': float(fold[11])
        }
    return info
plot = False
bar_plot = False
fold_plot = True
clust = ["03"]
ratio = ["norm"]#,"small","large"]#,"small","large"]
exps = ["CV_AFfiltered_expTFs_clu"]#, "CV_AFall_expTFs_clu"]#,"CV_AFfiltered_expTFs_clu"]
tfs = ["_Com"]
tf = tfs[0]
mode = ["spatial", "seq"]
eval_metric = "AU-PRC"
if not os.path.exists("../plots"):
    os.makedirs("../plots")

print(eval_metric, clust[0], ratio[0], exps[0], tf)
all_results = np.zeros((len(clust)*len(ratio)*5, len(mode)*len(exps)))
compares = []
for l, mod in enumerate(mode):                       
    exp = exps[0]
    compares.append(f'{exp}_{mod}{tfs[0]}')
    j = 0
    i = l
    for rat in ratio:    
        for clu in clust:  
            # Read text file
            if os.path.exists(f'../../results/{exp}{clu}_{rat}_{mod}{tf}/result_info.txt'):
                with open(f'../../results/{exp}{clu}_{rat}_{mod}{tf}/result_info.txt', 'r') as file:
                    text = file.read()
            else:
                path_name = exp.split('_')[1]
                if os.path.exists(f'../../benchmark/DeepReg/results/{path_name}/{clu}_{rat}_Com/result_info.txt'):
                    with open(f'../../benchmark/DeepReg/results/{path_name}/{clu}_{rat}_Com/result_info.txt', 'r') as file:
                        text = file.read()

                else:
                    print(f'{exp}{clu}_{rat}_{mod}')
                    continue


            # Extract information             
            result_info = extract_info(text)
            num_folds = len(result_info)
            for fold in result_info.values():
                for key, value in fold.items():
                    #print(key, value)
                    if eval_metric == "mis":
                        if key == "FP":
                            tmp = value
                        if key == "TN":
                            all_results[j,i] = tmp/(value+tmp)
                    if key == eval_metric:
                        all_results[j,i] = value

                j += 1

if bar_plot:
    plt.rcParams["font.family"] = "DejaVu Serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams["font.size"] = 20
    # Create the data DataFrame
    data = pd.DataFrame(all_results,
                    index=[f'{clu}_{rat}_{i}' for clu in clust for rat in ratio for i in range(1, 6)],
                    columns=["StrucTFactor", "DeepTFactor"])
    # Calculate the average and standard deviation for the desired index ranges
    averages = []
    std_devs = []

    for i in range(0, all_results.shape[0], 5):
        subset = data.iloc[i:i+5]
        averages.append(subset.mean())
        std_devs.append(subset.std())

    # Convert averages and std_devs to DataFrames for plotting
    averages_df = pd.DataFrame(averages)
    std_devs_df = pd.DataFrame(std_devs)

    # Plot the bar plot with error bars
    plt.figure()
    ax = averages_df.plot(kind='bar', yerr=std_devs_df, rot=0, color=['lightskyblue', 'grey'])
    #x_ticks = ["$D_{10}$", "$D_{11}$", "$D_{12}$"]
    x_ticks = ["$D_{1}$", "$D_{2}$", "$D_{3}$"]
    
    plt.xticks(range(3), x_ticks) 

    plt.xlabel(f"Dataset", labelpad=15)
    plt.ylabel(eval_metric, labelpad=15)
    plt.ylim(0.9, 1)
    plt.legend(["StrucTFactor", "DeepTFactor"], loc="upper right")
    #plt.title(f"{exps[0].split('_')[1]} {tf[1:]} Clustering: {clust[0]}")
    plt.tight_layout()
    print(f"../plots/{eval_metric}_{exps[0].split('_')[1]}{tf}_{clust[0]}_bar_plot_percent.png")
    #plt.savefig(f"../plots/AFall_Com_03_folds/{eval_metric}_{exps[0].split('_')[1]}{tf}_{clust[0]}_bar_plot.pdf")

if fold_plot:
    plt.rcParams["font.family"] = "DejaVu Serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams["font.size"] = 20
    # Create the data DataFrame
    data = pd.DataFrame(all_results,
                    index=[f'{clu}_{rat}_{i}' for clu in clust for rat in ratio for i in range(1, 6)],
                    columns=["Our", "DeepTFactor"])

    # Plot the bar plot with error bars
    plt.figure()
    ax = data.plot(kind='bar', rot=0, color=['lightskyblue', 'grey'])
    #x_ticks = [np.arange(1, 6)]
    
    plt.xticks(range(5), range(1,6))#, x_ticks) 

    plt.xlabel(f"Cross-validation fold of the $D_1$ dataset", labelpad=15)
    plt.ylabel(eval_metric, labelpad=15)
    plt.ylim(0, 1)
    plt.legend(["StrucTFactor", "DeepTFactor"], loc="lower right")
    #plt.title(f"{exps[0].split('_')[1]} {tf[1:]} Clustering: {clust[0]}")
    plt.tight_layout()
    plt.savefig(f"../plots/{eval_metric}_{exps[0].split('_')[1]}{tf}_{clust[0]}_{ratio[0]}_folds1.pdf")

