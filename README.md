# Transcription factor prediction using protein 3D secondary structures
![Image](StrucTFactorOverview.png)

## Abstract
Motivation: Transcription factors (TFs) are DNA-binding proteins that regulate gene expression. Traditional methods predict a protein as a TF if the protein contains any DNA-binding domains (DBDs) of known TFs. However, this approach fails to identify a novel TF that does not contain any known DBDs. Recently proposed TF prediction methods do not rely on DBDs. Such methods use features of protein sequences to train a machine learning model, and then use the trained model to predict whether a protein is a TF or not. Because 3-dimensional (3D) structure of a protein captures more information than its sequence, using 3D protein structures will likely allow for more accurate prediction of novel TFs. 

Results: We propose a deep learning-based TF prediction method (_StrucTFactor_), which is the first method to utilize 3D secondary structural information of proteins. We compare StrucTFactor with recent state-of-the-art TF prediction methods based on ∼525 000 proteins across 12 datasets, capturing different aspects of data bias (including sequence redundancy) possibly influencing a method’s performance. We find that StrucTFactor significantly (_p_-value < 0.001) outperforms the existing TF prediction methods, improving the performance over its closest competitor by up to 17% based on Matthews correlation coefficient.

## Software / Hardware prerequisites
Before you begin, ensure that you have the following prerequisites installed on your system:
- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- `pip` package manager
- `python` >=3.6.12

Models have been trained on a single Tesla T4 GPU.

## Installation Steps
Given CUDA version 12.2, the following installation guide was tested.

### 1. Clone the repository
```bash
git clone https://github.com/lieboldj/StrucTFactor.git
```

### 2. Create a conda environment
```bash
conda env create -f stf.yml
conda activate stf
```
If you want to benchmark with DeepReg, please install tensorflow.

## Predict TF/non-TF for a pdb file
Given a pdb file, you run the following to predict whether this protein is a TF: 
```bash
cd strucTFactor
python predictTFwithStrucTFactor.py -i <pdb_file>
```
The result will be printed in your terminal and written in _output.csv_.

## Reproducing results in paper
### Option 1: Download complete dataset:
Download it via:
```bash
python download_data.py
```

Extract via terminal:
```bash
unzip pre-trained_models.zip
unzip datasets.zip
```
To move the folders to the right directories:
```bash
mkdir -p benchmark/DeepReg/results
mv results/AF* benchmark/DeepReg/results
```

### Option 2: Create datasets by running the following script:
```bash
cd scripts/data_preparation
./create_datasets.sh
```
We recommand to download the datasets, because the runtime is >3 hours (including DSSP for all AlphaFold structures and analysis of the dataset).
This procedure includes the download of uniprot_sprot.dat.gz and all steps to create the datasets (see scripts/data_preparation).

### Run our experiments:
After you got or created all data, you can easily execute our method and reproduce the results:
```bash
cd strucTFactor
./run_cross_val.sh
```
Here, you can execute the programm of your choice (```./run_all_data.sh```, ```./run_cross_val.sh```, ```./run_explain_AI.sh```). Our five-fold cross validation is executed with ```./run_cross_val.sh```. 

## License
This work is licensed under a Creative Commons Attribution 4.0 International Public License.

## Reference
Liebold J., Neuhaus F., Geiser J., Kurtz S., Baumbach J., Newaz K. (2024). Transcription factor prediction using protein 3D secondary structures.


*Corresponding author:*

Khalique Newaz, khalique.newaz@uni-hamburg.de
