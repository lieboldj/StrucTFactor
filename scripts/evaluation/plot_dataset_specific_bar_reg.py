import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re


def extract_info(text):
    info = {}
    folds = re.findall(r'Fold: (\d+)\naccuracy: (\d+\.\d+), sensitivity: (\d+\.\d+), specificity: (\d+\.\d+), recall: (\d+\.\d+).*?TP: (\d+).*?FP: (\d+).*?TN: (\d+).*?FN: (\d+).*?ROC curve \(area = (\d+\.\d+)\).*?AUPRC \(AP\): (\d+\.\d+).*?MCC: (-?\d+\.\d+)', text, re.DOTALL)
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

plot = False
bar_plot = True
fold_plot = False

clust = ["03"]#,"No"]
ratio = ["norm","small","large"]#,"small","large"]
exps = ["CV_AFall_expTFs_clu"]#, "CV_AFall_expTFs_clu"]#,"CV_AFfiltered_expTFs_clu"]
tfs = ["_Com"]
tf = tfs[0]
mode = ["spatial","seq","reg"]
eval_metric = "MCC"
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
            if mod == "reg":
                if num_folds == 0:
                    print("No results")
                    for _ in range(5):
                        all_results[j,i] = 0
                        j += 1
                    continue
            start_fold = 0
            if mod != "reg" or (bar_plot and rat == "large") or clu == "No" or exp == "CV_AFall_expTFs_clu":
                for fold in result_info.values():
                    for key, value in fold.items():                   
                        #print(key, value)
                        if eval_metric == "mis":
                            if key == "FP":
                                tmp = value
                            if key == "TN":
                                all_results[j,i] = tmp/(value+tmp)

                        if key == eval_metric:
                            if mod != "reg":
                                all_results[j,i] = value
                            else:
                                print(fold['fold'], start_fold)
                                if start_fold != fold['fold']:
                                    for _ in range(fold['fold']-start_fold):
                                        all_results[j,i] = np.nan
                                        print(_)
                                        j += 1
                                    print("Here")
                                print(value)

                                all_results[j,i] = value
                    j += 1
                    start_fold = fold['fold'] + 1

            # hardcode the results for the overfitting folds of the DeepReg model
            elif eval_metric == "AU-ROC" and rat == "norm":
                for _ in range(5):    
                    if _ == 0:
                        all_results[j,i] = 0.3229
                    elif _ == 1:
                        all_results[j,i] = 0.3686
                    elif _ == 2:
                        all_results[j,i] = 0.3643
                    elif _ == 3:
                        all_results[j,i] = 0.3705
                    elif _ == 4:
                        all_results[j,i] = 0.4562
                    j += 1
            elif eval_metric == "AU-ROC" and rat == "small":
                for _ in range(5):    
                    if _ == 0:
                        all_results[j,i] = 0.5021
                    elif _ == 1:
                        all_results[j,i] = 0.4247
                    elif _ == 2:
                        all_results[j,i] = 0.2657
                    elif _ == 3:
                        all_results[j,i] = 0.7681
                    elif _ == 4:
                        all_results[j,i] = 0.3374
                    j += 1
            
            elif eval_metric == "AU-PRC" and rat == "norm":
                for _ in range(5):    
                    if _ == 0:
                        all_results[j,i] = 0.1789
                    elif _ == 1:
                        all_results[j,i] = np.nan
                    elif _ == 2:
                        all_results[j,i] = np.nan
                    elif _ == 3:
                        all_results[j,i] = np.nan
                    elif _ == 4:
                        all_results[j,i] = 0.2176
                    j += 1
            elif eval_metric == "AU-PRC" and rat == "small":
                for _ in range(5):    
                    if _ == 0:
                        all_results[j,i] = 0.1556
                    elif _ == 1:
                        all_results[j,i] = 0.1344
                    elif _ == 2:
                        all_results[j,i] = 0.1084
                    elif _ == 3:
                        all_results[j,i] = 0.3321
                    elif _ == 4:
                        all_results[j,i] = np.nan
                    j += 1

            elif eval_metric == "MCC" and rat == "norm":
                for _ in range(5):    
                    if _ == 0:
                        all_results[j,i] = -0.0993
                    elif _ == 1:
                        all_results[j,i] = np.nan
                    elif _ == 2:
                        all_results[j,i] = np.nan
                    elif _ == 3:
                        all_results[j,i] = np.nan
                    elif _ == 4:
                        all_results[j,i] = -0.0763
                    j += 1
            elif eval_metric == "MCC" and rat == "small":
                for _ in range(5):    
                    if _ == 0:
                        all_results[j,i] = -0.0569
                    elif _ == 1:
                        all_results[j,i] = 0.0549
                    elif _ == 2:
                        all_results[j,i] = -0.1253
                    elif _ == 3:
                        all_results[j,i] = 0.1919
                    elif _ == 4:
                        all_results[j,i] = np.nan
                    j += 1
            
print(all_results)
if fold_plot:
    plt.rcParams["font.family"] = "DejaVu Serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams["font.size"] = 18
    # Create the data DataFrame
    data = pd.DataFrame(all_results,
                    index=[f'{clu}_{rat}_{i}' for clu in clust for rat in ratio for i in range(1, 6)],
                    columns=["StrucTFactor", "DeepTFactor", "DeepReg"])

    # Plot the bar plot with error bars
    plt.figure()
    ax = data.plot(kind='bar', rot=0, color=['blue', 'darkgrey', 'darkseagreen'])
    #x_ticks = [np.arange(1, 6)]
    
    plt.xticks(range(5), range(1,6))#, x_ticks) 
    plt.yticks(np.arange(-0.2, 1.2, 0.2), [f"{int(i*100)}%" for i in np.arange(-0.2, 1.2, 0.2)])
    # add a thick red horizontal line at y=0
    
    plt.xlabel(f"Cross-validation fold of the $D_1$ dataset", labelpad=15)
    plt.ylabel(eval_metric, labelpad=15)
    if eval_metric == "MCC":
        plt.ylim(-0.13, 1)
    else:
        plt.ylim(0, 1)
    # remove the legend
    ax.get_legend().remove()
    #plt.legend(["StrucTFactor", "DeepTFactor", "DeepReg"], loc="lower center")
    if eval_metric == "MCC":
        plt.axhline(y=0, color='black', linewidth=2)
    elif eval_metric == "AU-ROC":
        plt.axhline(y=0.5, color='black', linewidth=2)
        plt.text(-0.5, 0.48, '50%', fontsize=12, ha='right')
    #plt.title(f"{exps[0].split('_')[1]} {tf[1:]} Clustering: {clust[0]}")
    plt.tight_layout()
    plt.savefig(f"../plots/Bars/spatialnoReg{eval_metric}_{exps[0].split('_')[1]}{tf}_{clust[0]}_{ratio[0]}_folds1.pdf")

if bar_plot:
    plt.rcParams["font.family"] = "DejaVu Serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams["font.size"] = 18
    # Create the data DataFrame
    data = pd.DataFrame(all_results,
                    index=[f'{clu}_{rat}_{i}' for clu in clust for rat in ratio for i in range(1, 6)],
                    columns=["StrucTFactor", "DeepTFactor", "DeepReg"])
    print(data)
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

    print(averages_df)

    # Plot the bar plot with error bars
    plt.figure()
    ax = averages_df.plot(kind='bar', yerr=std_devs_df, rot=0, color=['blue', 'darkgrey', 'darkseagreen'])
    #x_ticks = ["$D_{10}$", "$D_{11}$", "$D_{12}$"]
    x_ticks = ["$D_{1}$", "$D_{2}$", "$D_{3}$"]
    
    plt.xticks(range(3), x_ticks) 
    # yticks should be 0% to 100% instead of 0.0 to 1.0
    plt.yticks(np.arange(-0.2, 1.2, 0.2), [f"{int(i*100)}%" for i in np.arange(-0.2, 1.2, 0.2)])
    if eval_metric == "MCC":
        plt.ylim(-0.13, 1)
    else:
        plt.ylim(0, 1)
    # remove the legend
    ax.get_legend().remove()
    if eval_metric == "MCC":
        plt.axhline(y=0, color='black', linewidth=2)
    elif eval_metric == "AU-ROC":
        plt.axhline(y=0.5, color='black', linewidth=2)
        plt.text(-0.5, 0.48, '50%', fontsize=12, ha='right')
    plt.xlabel(f"Dataset", labelpad=15)
    plt.ylabel(eval_metric, labelpad=15)
    #plt.legend(["StrucTFactor", "DeepTFactor", "DeepReg"], loc="lower center")
    #plt.title(f"{exps[0].split('_')[1]} {tf[1:]} Clustering: {clust[0]}")
    plt.tight_layout()
    print(f"../plots/{eval_metric}_{exps[0].split('_')[1]}{tf}_{clust[0]}_bar_plot_percent.png")
    plt.savefig(f"../plots/Bars/{eval_metric}_{exps[0].split('_')[1]}{tf}_{clust[0]}_bar_plot.pdf")

