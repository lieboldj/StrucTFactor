import matplotlib.pyplot as plt
import csv
import re
import os

# Define function to extract information from text
def extract_info(text):
    info = {}
    folds = re.findall(r'Fold: (\d+).*?accuracy: (\d+\.\d+), sensitivity: (\d+\.\d+), specificity: (\d+\.\d+), recall: (\d+\.\d+).*?ROC curve \(area = (\d+\.\d+)\).*?AUPRC \(AP\): (\d+\.\d+).*?MCC: (\d+\.\d+)', text, re.DOTALL)
    
    for i, fold in enumerate(folds):
        info[f'Fold {i}'] = {
            'accuracy': float(fold[1]),
            'sensitivity': float(fold[2]),
            'specificity': float(fold[3]),
            'recall': float(fold[4]),
            'AU-ROC': float(fold[5]),
            'AU-PRC': float(fold[6]),
            'MCC': float(fold[7])
        }
    return info

clust = ["03", "No"]
ratio = ["norm","small","large"]
mode = ["X", "spatial"] # X = DeepReg
exps = ["CV_AFfiltered"]#_expTFs_clu"]
# if ../plots not exists, create it
if not os.path.exists("../plots"):
    os.makedirs("../plots")
fig, ax = plt.subplots()
eval_modes = ["AU-PRC", "MCC", "AU-ROC"]
for eval_mode in eval_modes:
    seq_mccs = []
    spatial_mccs = []
    colors = []  # List to store colors for each point
    legend_labels = []  # List to store legend labels
    markers = []  # List to store markers for each point
    for mod in mode:
                           
        for exp in exps:
            for rat in ratio:
                for clu in clust:
                    print(f'{exp}{clu}_{rat}_{mod}')
                    # Read text file
                    if os.path.exists(f'../../results/{exp}_expTFs_clu{clu}_{rat}_{mod}_Com/result_info.txt'):
                        with open(f'../../results/{exp}_expTFs_clu{clu}_{rat}_{mod}_Com/result_info.txt', 'r') as file:
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
                    average_fold = {}                
                    result_info = extract_info(text)
                    num_folds = len(result_info)
                    if num_folds < 5:
                        if mod == "X":
                            seq_mccs.append(0.0)
                        continue
                    for fold in result_info.values():
                        for key, value in fold.items():
                            if key not in average_fold:
                                average_fold[key] = value
                            else:
                                average_fold[key] += value

                    for key in average_fold:
                        average_fold[key] /= num_folds

                    if mod == "X":
                        seq_mccs.append(average_fold[eval_mode])
                    else:
                        spatial_mccs.append(average_fold[eval_mode])

                        legend_labels.append(f"{exp.split('_')[1]}({clu}), {eval_mode}, {rat} allTFs")

    print(len(seq_mccs), len(spatial_mccs))
    for i in range(len(seq_mccs)):
        if "norm" in legend_labels[i]:
            colors.append("black")
        elif "small" in legend_labels[i]:
            colors.append("green")
        elif "large" in legend_labels[i]:
            colors.append("orange")

        if "03" in legend_labels[i]:
            alpha = 1
        else:
            alpha = 0.5

        if eval_mode == "AU-ROC":
            markers.append("o")
        elif eval_mode == "AU-PRC":
            markers.append("x")
        else:
            markers.append("^")

        ax.scatter(seq_mccs[i], spatial_mccs[i], c=colors[i], label=legend_labels[i],\
                   marker=markers[i], s=40, alpha=alpha)

ax.set_xlabel("DeepReg")
ax.set_ylabel("StrucTFactor")
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_aspect('equal', 'box')
ax.plot([0, 1], [0, 1], color="black")

legend_elements = []
f = lambda m,c,a: plt.plot([],[],marker=m, color=c, alpha=a, ls="none")[0]
#my_colors = ["red", "maroon", "blue", "teal"]
my_colors = ["black", "green", "orange"]
handles = [f("s", my_colors[i]  ,1) for i in range(3)]
legend_elements.append(plt.Line2D([0], [1], color='white', linestyle='-', linewidth=1))
legend_elements += handles

# Horizontal line between methods and dataset_names
legend_elements.append(plt.Line2D([0], [1], color='white', linestyle='-', linewidth=1))

#handles += [f("s", "k", dataset_density[i]) for i in range(3)]
legend_elements += [f("s", "k", [1,0.5][i]) for i in range(2)]
legend_elements.append(plt.Line2D([0], [1], color='white', linestyle='-', linewidth=1))
#handles += [f(metric_marker[i], "k",1) for i in range(5)]
legend_elements += [f(["x","^"][i], "k",1) for i in range(2)]

dataset_names = ["$D_{Con}$", "$D_{Engy}$", "$D_{Evol}$"]
#labels = ["$\mathbf{Dataset}$"] +["AFall, ExpTFs","AFall, allTFs","AFfiltered, ExpTFs","AFfiltered, allTFs"] + ["$\mathbf{Clustering}$"] + ["0.3","No"]+ ["$\mathbf{Performance}$\n$\mathbf{measure}$"]+eval_modes
labels = ["$\mathbf{Dataset}$"] + ["1:3","1:5","1:10"] + ["$\mathbf{Clustering}$"] + ["0.3","No"]+ ["$\mathbf{Performance}$\n$\mathbf{measure}$"]+eval_modes


# Add legend
legend = ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Experiment (Cluster), Performance measure')
for handle in legend.legendHandles:
    handle.set_sizes([30.0])  # Set marker size in legend
leg = plt.legend(legend_elements, labels, loc='lower left', bbox_to_anchor=(1.02, 0.05))
plt.title("Comparison of StrucTFactor to DeepReg".format(exp.split('_')[1]))

plt.savefig(f"../plots/DeepReg/scatter{exp.split('_')[1]}.png", bbox_inches='tight')
print(f"../plots/DeepReg/scatter{exp.split('_')[1]}.png")

