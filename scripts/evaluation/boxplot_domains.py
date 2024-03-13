import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import mannwhitneyu
plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["font.size"] = 12
# Read the data
df = pd.read_csv('d1_domain_results.csv')
domains_struct = []
domains_seq = []
no_domains_struct = []
no_domains_seq = []
print(len(df))
for idx, p in df.iterrows():
    # p.domain is a numpy array wit 0 and 1, tell me all position numbers that are 1
    domain_list_pre = p.domain.lstrip("[ ").rstrip(" ]").split()
    domain_list = np.array([float(i) for i in domain_list_pre])
    nums = np.where(domain_list == 1.)
    if len(nums[0]) != 0:
        
        secondary_list_pre = p.secondary.lstrip("[ ").rstrip(" ]").split()
        secondary_list = [float(i) for i in secondary_list_pre]

        no_spatial_list_pre =  p.no_spatial
        no_spatial_list_pre = no_spatial_list_pre.lstrip("[ ").rstrip(" ]").split()
        no_spatial_list = [float(i) for i in no_spatial_list_pre]

        get_domain_score = np.sum(secondary_list[int(nums[0][0]):int(nums[0][-1])+1])
        get_no_spatial_score = np.sum(no_spatial_list[int(nums[0][0]):int(nums[0][-1])+1])

        # get score for all but not domain
        get_no_domain_score = np.sum(secondary_list) - get_domain_score
        get_no_no_spatial_score = np.sum(no_spatial_list) - get_no_spatial_score

        domains_struct.append(get_domain_score)
        domains_seq.append(get_no_spatial_score)
        no_domains_struct.append(get_no_domain_score)
        no_domains_seq.append(get_no_no_spatial_score)


print(np.mean(domains_struct), np.mean(domains_seq), np.mean(no_domains_struct), np.mean(no_domains_seq))
print(len(domains_struct), len(domains_seq), len(no_domains_struct), len(no_domains_seq))
print("start")

combine_struct = [domains_struct, no_domains_struct]
combine_seq = [domains_seq, no_domains_seq]

# Positioning the boxes
positions = [1, 2, 3.5, 4.5]

# Create figure and axis and fix figsize
fig, ax = plt.subplots(figsize=(7.5,4))
#fig, ax = plt.subplots()

# Create boxplot
boxplot = ax.boxplot(combine_struct + combine_seq, positions=positions, patch_artist=True, widths=0.7, medianprops=dict(color='black'))

# Set colors 
colors = ['pink', 'white', 'pink', 'white']
for patch, color in zip(boxplot['boxes'], colors):
    patch.set_facecolor(color)

# Set xticks
ax.set_xticks([1.5, 4])
ax.set_xticklabels(['StrucTFactor', 'DeepTFactor'])

# Add labels and title
ax.set_ylabel("Impact score for all TFs with\ninformation of the binding domain", labelpad=10)
ax.set_xlabel("Method", labelpad=10)
plt.tight_layout()
if not os.path.exists("../plots"):
    os.makedirs("../plots")
plt.savefig("../plots/domain_boxplot.png")
print(np.median(domains_struct), np.median(domains_seq), np.median(no_domains_struct), np.median(no_domains_seq))
print(np.mean(domains_struct), np.mean(domains_seq), np.mean(no_domains_struct), np.mean(no_domains_seq))

print(mannwhitneyu(domains_struct,  no_domains_struct))