import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
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
bar_plot = True
fold_plot = False
clust = ["03", "No"]
ratio = ["norm","small","large"]#,"small","large"]
exps = ["CV_AFfiltered_expTFs_clu", "CV_AFall_expTFs_clu"]#,"CV_AFfiltered_expTFs_clu"]"CV_AFfiltered_expTFs_clu", 
tfs = ["_Com"]
tf = tfs[0]
mode = ["spatial", "reg"] # use the term "reg" here when comparing to DeepReg
eval_metric = "AU-ROC"
if not os.path.exists("../plots"):
    os.makedirs("../plots")
print(eval_metric, clust[0], ratio[0], exps[0])
all_results = np.zeros((len(clust)*len(ratio)*5, len(mode)*len(exps)))

result_list_1 = []
result_list_2 = []
exp_names = []

compares = []
for l, mod in enumerate(mode):     
    for exp in exps:                  
    #exp = exps[0]
        compares.append(f'{exp}_{mod}')
        j = 0
        i = l
        for clu in clust:
            for rat in ratio:    
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
                exp_name = exp[3:8]
                if exp_name == "AFall":
                    exp_name = "allAF"
                else: 
                    exp_name = "confidentAF"
                
                if mod != "reg" or rat == "large" or exp_name == "allAF" or clu == "No":
                    for fold in result_info.values():
                        for key, value in fold.items():
                            #print(key, value)
                            if eval_metric == "mis":
                                if key == "TP":
                                    tmp = value
                                if key == "FP":
                                    all_results[j,i] = tmp/(value+tmp)
                                    if mod == "spatial":
                                        result_list_1.append(tmp/(value+tmp))
                                        exp_names.append(f'{exp_name}_{clu}_{rat}')
                                    else: 
                                        result_list_2.append(tmp/(value+tmp))
                            if key == eval_metric:
                                all_results[j,i] = value
                                if mod == "spatial":
                                    result_list_1.append(value)
                                    exp_names.append(f'{exp_name}_{clu}_{rat}')
                                else: 
                                    result_list_2.append(value)

                        j += 1
                # hardcode the results for the datasets which overfit for DeepReg
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
                        result_list_2.append(all_results[j,i])
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
                        result_list_2.append(all_results[j,i])
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
                        result_list_2.append(all_results[j,i])
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
                        result_list_2.append(all_results[j,i])
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
                        result_list_2.append(all_results[j,i])
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
                        result_list_2.append(all_results[j,i])
                        j += 1

                elif eval_metric == "mis" and rat == "small":
                    for _ in range(5):    
                        if _ == 0:
                            all_results[j,i] = 0.1
                        elif _ == 1:
                            all_results[j,i] = 0.1724
                        elif _ == 2:
                            all_results[j,i] = 0.02899
                        elif _ == 3:
                            all_results[j,i] = 0.3768
                        elif _ == 4:
                            all_results[j,i] = 0
                        result_list_2.append(all_results[j,i])
                        j += 1
                
                elif eval_metric == "mis" and rat == "norm":
                    for _ in range(5):    
                        if _ == 0:
                            all_results[j,i] = 0.2286
                        elif _ == 1:
                            all_results[j,i] = np.nan
                        elif _ == 2:
                            all_results[j,i] = np.nan
                        elif _ == 3:
                            all_results[j,i] = np.nan
                        elif _ == 4:
                            all_results[j,i] = 0.2116
                        result_list_2.append(all_results[j,i])
                        j += 1

statistic, p_value = wilcoxon(result_list_1, result_list_2)
print(f"Wilcoxon test statistic: {statistic}")
print(f"Wilcoxon test p-value: {p_value}")

avg_scores = np.zeros((4,2))
std_scores = np.zeros((4,2))
wilco = np.zeros((4))
print(avg_scores)
print("result all", wilcoxon(result_list_1[30:], result_list_2[30:]))
print("result confident", wilcoxon(result_list_1[:30], result_list_2[:30]))
for i in range(len(clust)*len(exps)):
    num = 15
    statistic, p_value = wilcoxon(result_list_1[i*15:i*15+15], result_list_2[i*15:i*15+15])
    print(f"Wilcoxon test statistic: {statistic}\n"
      f"Wilcoxon test p-value: {p_value}")
    wilco[i] = p_value
    # remove np.nan values and get the mean
    avg_scores[i,0] = np.nanmean(result_list_1[i*15:i*15+15])
    avg_scores[i,1] = np.nanmean(result_list_2[i*15:i*15+15])
    std_scores[i,0] = np.nanstd(result_list_1[i*15:i*15+15])
    std_scores[i,1] = np.nanstd(result_list_2[i*15:i*15+15])
    print(avg_scores[i])
print(eval_metric)
#print(avg_scores)
#print(wilco)
# create a bar plot of average scores
plt.figure()
plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
#plt.rcParams["font.size"] = 20
x = np.arange(4)
bars1 = plt.bar(x, avg_scores[:,0], yerr=std_scores[:,0], width=0.4, label="StrucTFactor", color="lightskyblue")
bars2 = plt.bar(x+0.4, avg_scores[:,1], yerr=std_scores[:,1], width=0.4, label="DeepTFactor", color="grey")
plt.xticks(x+0.2, ["confidentAF 30%", "confidentAF 100%", "allAF 30%", "allAF 100%"], rotation=0)
plt.xlabel("Experiment")
plt.ylabel(f"Average {eval_metric}")
plt.legend(loc="lower right")
# Adding p-values
for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
    # Check if the p-value is significant
    if wilco[i] < 1:
        if i%2 == 1:
            # Calculate line coordinates
            line_x = [bar1.get_x() + bar1.get_width() / 2, bar1.get_x() + bar1.get_width() / 2,
                      bar2.get_x() + bar2.get_width() / 2]
            line_y = [bar1.get_height() + 0.03, bar1.get_height() + 0.05,
                      bar1.get_height() + 0.05]

            # Plot the connecting line
            plt.plot(line_x, line_y, color='gray')

            # Add p-value text above the horizontal line
            plt.text((bar1.get_x() + bar2.get_x() + bar2.get_width()) / 2, 
                     bar1.get_height() + 0.055, f"{wilco[i]:.7f}", ha='center', va='bottom')
            # Calculate line coordinates for bar2
            line2_x = [bar2.get_x() + bar2.get_width() / 2, bar2.get_x() + bar2.get_width() / 2]
            line2_y = [bar2.get_height() + 0.03, bar1.get_height() + 0.05]

            # Plot the connecting line for bar2
            plt.plot(line2_x, line2_y, color='gray')
        else:
            # Calculate line coordinates
            line_x = [bar1.get_x() + bar1.get_width() / 2, bar1.get_x() + bar1.get_width() / 2,
                      bar2.get_x() + bar2.get_width() / 2]
            line_y = [bar1.get_height() + 0.07, bar1.get_height() + 0.09,
                      bar1.get_height() + 0.09]

            # Plot the connecting line
            plt.plot(line_x, line_y, color='gray')

            # Add p-value text above the horizontal line
            plt.text((bar1.get_x() + bar2.get_x() + bar2.get_width()) / 2, 
                     bar1.get_height() + 0.095, f"p={wilco[i]:.7f}", ha='center', va='bottom')
            # Calculate line coordinates for bar2
            line2_x = [bar2.get_x() + bar2.get_width() / 2, bar2.get_x() + bar2.get_width() / 2]
            line2_y = [bar2.get_height() + 0.07, bar1.get_height() + 0.09]

            # Plot the connecting line for bar2
            plt.plot(line2_x, line2_y, color='gray')


plt.tight_layout()

plt.savefig(f"../plots/avg_scores_with_pvalue_{eval_metric}.png")

exit()
