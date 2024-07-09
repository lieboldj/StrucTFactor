import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import mannwhitneyu

show_details = False

plt.rcParams["font.family"] = "DejaVu Sans"
#plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["font.size"] = 12
# Read the data
df = pd.read_csv('d_rl_nr_3_domain.csv')

domains_struct = []
domains_seq = []
no_domains_struct = []
no_domains_seq = []

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

combine_struct = [domains_struct, no_domains_struct]
combine_seq = [domains_seq, no_domains_seq]

# Positioning the boxes
positions = [1, 2, 3.5, 4.5]

# Create figure and axis and fix figsize
fig, ax = plt.subplots(figsize=(7.5,4))
#fig, ax = plt.subplots()

box_width = 0.7
# Create boxplot
boxplot = ax.boxplot(combine_struct + combine_seq, positions=positions, patch_artist=True, widths=box_width, medianprops=dict(color='black'))

# Set colors 
colors = ['pink', 'white', 'pink', 'white']
for patch, color in zip(boxplot['boxes'], colors):
    patch.set_facecolor(color)

# Set xticks
ax.set_xticks([1.5, 4])
ax.set_xticklabels(['StrucTFactor', 'DeepTFactor'])

if show_details:
    # Add horizontal lines at the medians between the pairs of boxes
    median1 = np.median(domains_struct)
    median2 = np.median(no_domains_struct)
    median3 = np.median(domains_seq)
    median4 = np.median(no_domains_seq)

    ax.plot([positions[0]+box_width/2, positions[1]-box_width/2], [median1, median1], color='b', linestyle='dotted', linewidth=1)
    ax.plot([positions[0]+box_width/2, positions[1]-box_width/2], [median2, median2], color='b', linestyle='dotted', linewidth=1)
    ax.plot([(positions[0]+positions[1])/2, (positions[0]+positions[1])/2], [median1, median2], color='b', linestyle='dotted', linewidth=1)
    ax.text((positions[0]+positions[1])/2+0.15, median1, f"{median1-median2:.2f}", color='black', fontsize=10, ha='center', va='bottom')


    ax.plot([positions[2]+box_width/2, positions[3]-box_width/2], [median3, median3], color='orange', linestyle='dotted', linewidth=1)
    ax.plot([positions[2]+box_width/2, positions[3]-box_width/2], [median4, median4], color='orange', linestyle='dotted', linewidth=1)
    ax.plot([(positions[2]+positions[3])/2, (positions[2]+positions[3])/2], [median3, median4], color='orange', linestyle='dotted', linewidth=1)
    ax.text((positions[2]+positions[3])/2+0.15, median3, f"{median3-median4:.2f}", color='black', fontsize=10, ha='center', va='bottom')

# Add labels and title
#ax.set_ylabel("Impact score for all TFs with\ninformation of the binding domain", labelpad=10)
ax.set_ylabel("Integrated gradient score", labelpad=10)
ax.set_xlabel("Method", labelpad=10)
plt.tight_layout()
if not os.path.exists("../plots/Paper"):
    os.makedirs("../plots/Paper")
plt.savefig("../plots/Paper/domain_boxplot.pdf")
#print(np.median(domains_struct), np.median(domains_seq), np.median(no_domains_struct), np.median(no_domains_seq))
#print(np.mean(domains_struct), np.mean(domains_seq), np.mean(no_domains_struct), np.mean(no_domains_seq))

print(mannwhitneyu(domains_struct,  no_domains_struct), mannwhitneyu(domains_seq, no_domains_seq))
print(mannwhitneyu(domains_struct,  domains_seq), mannwhitneyu(no_domains_struct, no_domains_seq))