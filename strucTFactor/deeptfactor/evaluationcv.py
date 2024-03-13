import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve

# evaluation function with MCC, ROC-AUC, AUPRC, F1, Accuracy, Sensitivity, Specificity, Precision, Recall 
def evaluate_model(seq_ids, labels, scores, needed, output_dir, fold):

    cutoff = 0.5
    eval_list = []
    if fold == 0:
        createOrAdd = 'w'
    else:
        createOrAdd = 'a'
    with open(f'{output_dir}/prediction_result.tsv', createOrAdd) as fp:
        fp.write(f'Fold {fold}; Time for run: {needed} \n')
        fp.write('sequence_ID\tprediction\tscore\texp_result\n')
        for (seq_id,  y, score) in zip(seq_ids, labels, scores):
            if score > cutoff:
                tf = 1
            else:
                tf = 0
            tmp_str = f'{seq_id}\t{tf}\t{score:0.4f}\t{y}\n'
            fp.write(tmp_str)
            eval_list.append([seq_id, tf, y])
        fp.write("\n\n")
    #calculate the accuracy, sensitivity, specificity and recall of prediction and groundtruth

    df = pd.DataFrame(eval_list, columns=['sequence_ID', 'prediction', 'exp_result'])
    df = df.astype({'prediction': 'int32', 'exp_result': 'int32'})
    TP = df[(df['prediction'] == 1) & (df['exp_result'] == 1)].shape[0]
    TN = df[(df['prediction'] == 0) & (df['exp_result'] == 0)].shape[0]
    FP = df[(df['prediction'] == 1) & (df['exp_result'] == 0)].shape[0]
    FN = df[(df['prediction'] == 0) & (df['exp_result'] == 1)].shape[0]
    print(f"Fold: {fold}")
    print(f'TP: {TP}\tFP: {FP}\n TN: {TN}\tFN: {FN}')
    # except to avoid errors in local minima runs:
    if  (TP * TN * FP * FN) == 0:
        print("One of the values is 0, so the accuracy, sensitivity, specificity and recall can't be calculated")
        labels = df['exp_result'].values
        score = df['prediction'].values
        #plot the ROC curve and calculate the AUC-ROC
        fpr, tpr, thresholds = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                    lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC and AUC - Fold: {fold}')
        plt.legend(loc="lower right")
        plt.show()
        plt.savefig(f'{output_dir}/ROC_F{fold}.png')
        with open(f'{output_dir}/result_info.txt', createOrAdd) as fp:
            fp.write(f'Fold: {fold}\n')
            fp.write("One of the values is 0, so the accuracy, sensitivity, specificity and recall can't be calculated\n")
            fp.write(f'TP: {TP}\tFP: {FP}\nTN: {TN}\tFN: {FN}\n')
            fp.write(f'ROC curve (area = {roc_auc:0.4f})\n')
            fp.write("\n\n")

        return np.array([TP, TN, FP, FN, roc_auc, 0, 0])
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    recall = TP / (TP + FP)
    print(f'accuracy: {accuracy:0.4f}, sensitivity: {sensitivity:0.4f}, specificity: {specificity:0.4f}, recall: {recall:0.4f}')
    MCC = (TP*TN - FP*FN)/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))

    labels = df['exp_result'].values
    score = df['prediction'].values
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
                lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC and AUC - Fold: {fold}')
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig(f'{output_dir}/ROC_F{fold}.png')
    AUPRC = average_precision_score(labels, scores)
    with open(f'{output_dir}/result_info.txt', createOrAdd) as fp:
        fp.write(f'Fold: {fold}\n')
        fp.write(f'accuracy: {accuracy:0.4f}, sensitivity: {sensitivity:0.4f}, specificity: {specificity:0.4f}, recall: {recall:0.4f}\n')
        fp.write(f'TP: {TP}\tFP: {FP}\nTN: {TN}\tFN: {FN}\n')
        fp.write(f'ROC curve (area = {roc_auc:0.4f})\n')
        fp.write(f'AUPRC (AP): {AUPRC:0.4f}\n')
        fp.write(f'MCC: {MCC:0.4f}\n')
        fp.write("\n\n")

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    plt.figure()
    lw = 2
    plt.plot(recall, precision, color='darkorange',
                lw=lw, label='Ave. Precision score %0.2f' % AUPRC)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig(f'{output_dir}/PRC.png')



    if True:
        print("Evaluation complete")
        return np.array([TP, TN, FP, FN, roc_auc, AUPRC, MCC])


    fpr, tpr, thresholds = metrics.roc_curve(test_loader[1], scores)  
    roc_auc = metrics.auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
                lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')    
    plt.legend(loc="lower right")
    plt.show()


