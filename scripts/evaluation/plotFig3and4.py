import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re

def extract_info(text):
    info = {}
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

# Settings
eval_metrics = ["AU-ROC", "AU-PRC", "MCC"]
plot = True
fold_plot = True
ratio = ["norm", "small", "large"]
save_format = "pdf"
clust = ["03"]
exps = ["CV_AFfiltered_expTFs_clu"]
tfs = ["_Com"]
tf = tfs[0]
mode = ["spatial0", "seq0", "reg"]

if not os.path.exists("../plots"):
    os.makedirs("../plots")

# Data storage
all_results = np.zeros((len(clust) * len(ratio) * 5, len(mode) * len(exps)))

# Open PDF
#with PdfPages('../plots/combined_plots.pdf') as pdf:
# start a 2x2 subplot
if True:
    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams["font.size"] = 18
    fig, axs = plt.subplots(2, 2, figsize=(10, 7.5))
    for k, eval_metric in enumerate(eval_metrics):
        for l, mod in enumerate(mode):
            exp = exps[0]
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
                    if mod == "reg" and num_folds == 0:
                        print("No results")
                        for _ in range(5):
                            all_results[j, i] = 0
                            j += 1
                        continue

                    start_fold = 0
                    for fold in result_info.values():
                        for key, value in fold.items():
                            if eval_metric == "mis":
                                if key == "FP":
                                    tmp = value
                                if key == "TN":
                                    all_results[j, i] = tmp / (value + tmp)

                            if key == eval_metric:
                                all_results[j, i] = value

                        j += 1
                        start_fold = fold['fold'] + 1

        # Plotting
        fontsize = 16
        data = pd.DataFrame(all_results,
                                index=[f'{clu}_{rat}_{i}' for clu in clust for rat in ratio for i in range(1, 6)],
                                columns=["StrucTFactor", "DeepTFactor", "DeepReg"])
        if k == 0:
            ax_idx = (0, 0)
        elif k == 1:
            ax_idx = (1, 0)
        elif k == 2:
            ax_idx = (1, 1)

        if len(ratio) == 1:
            # Create the data DataFrame
            
            # Plot the bar plot with error bars
            data.plot(kind='bar', ax=axs[ax_idx], rot=0, color=['blue', 'darkorange', 'darkseagreen'])
            #data.plot(kind='bar', ax=axs[0,0], rot=0, color=['blue', 'darkorange', 'darkseagreen'])
            axs[ax_idx].set_xticks(range(5), range(1, 6))
            axs[ax_idx].set_xticklabels(range(1, 6), fontsize=fontsize)
            axs[ax_idx].set_xlabel(f"Test set", labelpad=10, fontsize=fontsize)
        
        if len(ratio) == 3:
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

            averages_df.plot(kind='bar', ax=axs[ax_idx], yerr=std_devs_df, rot=0, color=['blue', 'darkorange', 'darkseagreen'])
            #x_ticks = ["$D_{10}$", "$D_{11}$", "$D_{12}$"]
            x_ticks = ["$D(rl,nr,3)$", "$D(rl,nr,5)$", "$D(rl,nr,10)$"]
            axs[ax_idx].set_xlabel(f"Dataset", labelpad=10, fontsize=fontsize)
            # LEGEND
            #ax.legend(["StrucTFactor", "DeepTFactor", "DeepReg"], bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            axs[ax_idx].set_xticks(range(3), x_ticks)
            axs[ax_idx].set_xticklabels(x_ticks, fontsize=fontsize-2)

        # y-axis for both plots the same
        axs[ax_idx].set_yticks(np.arange(0, 1.2, 0.2))
        axs[ax_idx].set_yticklabels([f"{int(i * 100)}%" for i in np.arange(0, 1.2, 0.2)], fontsize=fontsize)
        if eval_metric == "MCC":
            axs[ax_idx].set_ylim(-0.05, 1)
        else:
            axs[ax_idx].set_ylim(0, 1)
        # remove the legend
        axs[ax_idx].get_legend().remove()
        if eval_metric == "MCC":
            axs[ax_idx].axhline(y=0, color='black', linewidth=2)
        elif eval_metric == "AU-ROC":
            axs[ax_idx].axhline(y=0.5, color='black', linewidth=2)
            axs[ax_idx].text(-0.5, 0.48, '50%', fontsize=12, ha='right')
        elif eval_metric == "AU-PRC" and (len(ratio) == 3 or len(ratio) == 4):
            # Define x positions for each line based on where the bars are located
            group_width = 0.6  # Adjust as necessary to match the width of each bar group

            # Add lines for each group separately
            axs[ax_idx].plot([-0.3, -0.3 + group_width], [0.25, 0.25], color='black', linewidth=2)  # First group
            axs[ax_idx].plot([-0.3 + 1, -0.3 + 1 + group_width], [0.167, 0.167], color='black', linewidth=2)  # Second group
            axs[ax_idx].plot([-0.3 + 2, -0.3 + 2 + group_width], [0.09, 0.09], color='black', linewidth=2)  # Third group
        elif eval_metric == "AU-PRC" and len(ratio) == 1:
            axs[ax_idx].axhline(y=0.25, color='black', linewidth=2)
            axs[ax_idx].text(-0.5, 0.24, '25%', fontsize=12, ha='right')

        axs[ax_idx].set_ylabel(eval_metric, labelpad=10, fontsize=fontsize)

    # Create legend separately
    #fig, ax = plt.subplots()
    handles = [plt.Line2D([0], [0], color=color, lw=8) for color in ['blue', 'darkorange', 'darkseagreen']]
    labels = ["StrucTFactor", "DeepTFactor", "DeepReg"]
    axs[0,1].legend(handles, labels, loc='center', fontsize=fontsize)
    axs[0,1].axis('off')
    #pdf.savefig(fig)
    #plt.close()
    plt.tight_layout()
    if len(ratio) == 1:
        plt.savefig(f'../plots/Figure3.{save_format}', format=save_format)
    else:
        plt.savefig(f'../plots/Figure4.{save_format}', format=save_format)

print("Combined PDF saved successfully.")
