# -*- coding: utf-8 -*-
##############################################################################################
# Code adapted from the DeepReg repository Inference script and ScratchModel Jupyter notebook
##############################################################################################
import numpy as np
import os
os.environ["OMP_NUM_THREADS"] = '20'
import sys

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
#import tensorflow.keras.preprocessing.text as kpt
from tensorflow.keras.preprocessing.text import Tokenizer
from Bio import SeqIO
import pickle
import re
import math
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Concatenate, concatenate, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, MaxPooling3D, GlobalMaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import plot_model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.layers import TimeDistributed


#%%

class Attention_A2(Layer):
    
    def __init__(self, return_sequences = True):
        self.return_sequences = return_sequences
        super(Attention_A2,self).__init__()
 
    def build(self,input_shape):
        self.W=self.add_weight(name='attention_weight', shape=(input_shape[-1],1), 
                                initializer='random_normal', trainable=True)
        self.b=self.add_weight(name='attention_bias', shape=(input_shape[1],1), 
                                initializer='zeros', trainable=True)        
        super(Attention_A2, self).build(input_shape)
 
    def call(self,x):
        # Alignment scores. Pass them through tanh function
        e = K.tanh(K.dot(x,self.W)+self.b)
        # Remove dimension of size 1
        e = K.squeeze(e, axis=-1)   
        # Compute the weights
        alpha = K.softmax(e)
        # Reshape to tensorFlow format
        alpha = K.expand_dims(alpha, axis=-1)
        # Compute the context vector
        context = x * alpha
        context = K.sum(context, axis=1)
        return context
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'return_sequences': self.return_sequences 
        })
        return config
    
class CNN_LSTM:

    def loading(lr=0.001):
        aa_shape = 21
        input_shape = (1000, aa_shape, 1)
        ###################### FIRST CNN #####################################
        model_1 = Sequential()
        
        model_1.add(Conv2D(128, (4, aa_shape), input_shape = input_shape, name='conv2d'))
        model_1.add(BatchNormalization(name='batch_normalization'))
        model_1.add(Activation('relu', name='activation'))
        model_1.add(Dropout(0.2))
        
        model_1.add(Conv2D(128, (4, 1), name='conv2d_1'))
        model_1.add(BatchNormalization(name='batch_normalization_1'))
        model_1.add(Activation('relu', name='activation_1'))
        model_1.add(Dropout(0.5))
        
        model_1.add(Conv2D(128, (16, 1), name='conv2d_2'))
        model_1.add(BatchNormalization(name='batch_normalization_2'))
        model_1.add(Activation('relu', name='activation_2'))
        model_1.add(Dropout(0.7))
        
        ###################### SECOND CNN #####################################
        
        model_2 = Sequential()
        
        model_2.add(Conv2D(128, (12, aa_shape), input_shape = input_shape, name='conv2d_3'))
        model_2.add(BatchNormalization(name='batch_normalization_3'))
        model_2.add(Activation('relu', name='activation_3'))
        model_2.add(Dropout(0.2))
        
        model_2.add(Conv2D(128, (8, 1), name='conv2d_4'))
        model_2.add(BatchNormalization(name='batch_normalization_4'))
        model_2.add(Activation('relu', name='activation_4'))
        model_2.add(Dropout(0.5))
        
        model_2.add(Conv2D(128, (4, 1), name='conv2d_5'))
        model_2.add(BatchNormalization(name='batch_normalization_5'))
        model_2.add(Activation('relu', name='activation_5'))
        model_2.add(Dropout(0.7))
        
        ###################### THIRD CNN #####################################
        
        model_3 = Sequential()
        
        model_3.add(Conv2D(128, (16, aa_shape), input_shape = input_shape, name='conv2d_6'))
        model_3.add(BatchNormalization(name='batch_normalization_6'))
        model_3.add(Activation('relu', name='activation_6'))
        model_3.add(Dropout(0.2))
        
        model_3.add(Conv2D(128, (4, 1), name='conv2d_7'))
        model_3.add(BatchNormalization(name='batch_normalization_7'))
        model_3.add(Activation('relu', name='activation_7'))
        model_3.add(Dropout(0.5))
        
        model_3.add(Conv2D(128, (4, 1), name='conv2d_8'))
        model_3.add(BatchNormalization(name='batch_normalization_8'))
        model_3.add(Activation('relu', name='activation_8'))
        model_3.add(Dropout(0.7))
        
        ###################### FOURTH CNN #####################################
        
        model_4 = Sequential()
        
        model_4.add(Conv2D(128, (6, aa_shape), input_shape = input_shape, name='conv2d_9'))
        model_4.add(BatchNormalization(name='batch_normalization_9'))
        model_4.add(Activation('relu', name='activation_9'))
        model_4.add(Dropout(0.2))
        
        model_4.add(Conv2D(128, (6, 1), name='conv2d_10'))
        model_4.add(BatchNormalization(name='batch_normalization_10'))
        model_4.add(Activation('relu', name='activation_10'))
        model_4.add(Dropout(0.5))
        
        model_4.add(Conv2D(128, (12, 1), name='conv2d_11'))
        model_4.add(BatchNormalization(name='batch_normalization_11'))
        model_4.add(Activation('relu', name='activation_11'))
        model_4.add(Dropout(0.7))
        
        ###################### CONCATENATION AND LSTM #################################

        merge = Concatenate(name='concatenate')([model_1.output, model_2.output, model_3.output, model_4.output])
        #merge = Concatenate()([model_1.output, model_2.output])
        
        merge_1 = Conv2D(512, (1, 1), name='conv2d_12')(merge)
        merge_2 = BatchNormalization(name='batch_normalization_12')(merge_1)
        merge_3 = Activation('relu', name='activation_12')(merge_2)
        merge_4 = MaxPooling2D(pool_size = (2, 1), name='max_pooling2d')(merge_3)        
        
        # https://stackoverflow.com/questions/52936132/4d-input-in-lstm-layer-in-keras
        
        lstm_1 = TimeDistributed(Flatten(input_shape = (1,512), name='time_distributed'))(merge_4)
        #lstm_2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, name='bidirectional'))(lstm_1)
        #output = tf.keras.layers.Dense(1, kernel_regularizer = 'L2', activation = 'sigmoid', name='dense')(lstm_2)
        
        #lstm_1 = TimeDistributed(Flatten(input_shape = (1,512)))(merge_4)
        lstm_2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512, return_sequences = True, name='bidirectional'))(lstm_1)
        attn_1 = Attention_A2()(lstm_2)
        output = tf.keras.layers.Dense(1, kernel_regularizer = 'l2', activation = 'sigmoid')(attn_1)

        ###################### MODEL #####################################
        
        model = Model(inputs = [model_1.input, model_2.input, model_3.input, model_4.input], outputs = output)
        
        #model = Model(inputs = [model_1.input, model_2.input], outputs = output)
        opt = keras.optimizers.Adam(learning_rate = lr)
        #opt = keras.optimizers.SGD(learning_rate = initial_learning_rate)
        model.compile(loss = 'binary_crossentropy',
                      optimizer = opt,
                      metrics = ['accuracy', 'AUC', 'Precision','Recall', 'TrueNegatives', 'TruePositives', 'FalseNegatives','FalsePositives'])        
        return model
    
def step_decay(epoch):
    initial_lrate = 0.1
    drop = 0.5
    epochs_drop = 8.0
    lrate = initial_lrate * math.pow(drop,  
           math.floor((1+epoch)/epochs_drop))
    return lrate


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.lr = []
 
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.lr.append(step_decay(len(self.losses)))

class Data_inference():

    def __init__(self, data_path):
        self.data_path = data_path

    ##%%
    def loading(self, file_path):
        
        def read_fasta_data(fasta_file, len_criteria=1000):
            result = []
            seq_ids = []
            labels = []
            fp = open(fasta_file, 'r')
            for seq_record in SeqIO.parse(fp, 'fasta'):
                seq = seq_record.seq
                seq_id = seq_record.id
                tmp_str = re.findall("TF=[0-1]", seq_record.description)
                # implementation of length requirement and padding
                # get labels from StrucTFactor dataset files
                if len(tmp_str) != 0:
                    label = int(tmp_str[-1].split("=")[1])     
                else: 
                    label = 0
                if len(seq) <= len_criteria:
                    seq += '_' * (len_criteria-len(seq))
                    result.append(str(seq))
                    seq_ids.append(seq_id)
                    labels.append(label)
            fp.close()
            return np.asarray(result), np.asarray(seq_ids), np.asarray(labels)

        seq_test , seq_test_ids, test_labels = read_fasta_data(file_path)
        return seq_test, seq_test_ids, test_labels

    ##%%
    def cleaning(self, seq_test):

        def amino_count(sequences):
            amino_input = set()
            amino2count = {}

            for seq in sequences:
                seq_1 = list(seq)
                for amino in seq_1:
                    if amino not in amino_input:
                        amino_input.add(amino)
                        amino2count[amino] = 1
                    else:
                        amino2count[amino] += 1
            return amino2count

        def cleaning_sequence(sequence):

            clean_seq_tfs = []
            for seq in sequence:
                seq_1 = list(seq)
                if not any(amino in seq_1 for amino in ('B','O','U','Z')):
                    clean_seq_tfs.append(seq)
            return clean_seq_tfs

        clean_seq_test = cleaning_sequence(seq_test)

        clean_amino2count_test = amino_count(clean_seq_test)
        print('Aminoacids present in testing dataset after cleaning: ', clean_amino2count_test)
        print('Total of aminoacids in dataset: ', len(clean_amino2count_test))
        return clean_amino2count_test, clean_seq_test

    ##%%
    def counting(self, clean_amino2count_test):

        def histograms(amino2count_tf):

            import pandas as pd

            amino_tf = pd.DataFrame(list(amino2count_tf.items()) , columns = ['Amino', 'Values'])
            amino_tf = amino_tf.sort_values(by = ['Amino'], ascending = True)
            ax = amino_tf.plot.bar(x = 'Amino', y = 'Values', rot = 0, title = 'Aminoacids in test sequence')

            return amino_tf

        clean_amino_test = histograms(clean_amino2count_test)
        amino_names = list(clean_amino_test['Amino'])
        print('Aminoacids present in dataset: ', amino_names)
        print('Number of different aminoacids: ', len(amino_names))
        return amino_names


    ##%%
    def tokens(self, clean_seq_test):

        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)

        def get_sequences(tokenizer, database):
            sequences_token = tokenizer.texts_to_sequences(database)
            sequences_padded = pad_sequences(sequences_token, truncating = 'post',
                                             padding = 'post', maxlen = 1000)
            sequences_encoded = to_categorical(sequences_padded, num_classes = 21)
            return sequences_encoded

        padded_seq_test = get_sequences(tokenizer, clean_seq_test)

        return padded_seq_test


    ##%%
    def inference(self, padded_seq_test):
        model_loaded = load_model('saved_model.h5') # example model from authors of DeepReg
        #model_loaded.summary()

        print('Testing sequence shape:', padded_seq_test.shape, type(padded_seq_test))

        import tensorflow as tf
        padded_seq_test = tf.expand_dims(padded_seq_test, -1)
        print('Expanded testing sequence shape:', padded_seq_test.shape, type(padded_seq_test))

        pred = model_loaded.predict([padded_seq_test, padded_seq_test, padded_seq_test, padded_seq_test])
        predictions = np.array((pred > 0.5).astype(np.uint8))
        positions = np.where(pred > 0.5)
        print('Total sequences: ', len(predictions))
        print('Predicted sequences with TF: ', np.sum(predictions))
        print('Predicted sequences with noTF: ', np.sum(1 - predictions))

        return predictions, positions, pred


def get_tfs(valors, indices, map):
    outfile = open(data_path + "predictions.txt", 'a')

    for reg in indices[0]:
        name_protein = map[reg]
        outfile.write(name_protein + "\t" + str(valors[reg])+"\n")

    outfile.close()

# use our evaluation strategy to evaluate the model
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
            #print(seq_id, tf, score, y)
            score = score[0]
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
            fp.write(f'accuracy: {np.nan}, sensitivity: {np.nan}, specificity: {np.nan}, recall: {np.nan}\n')
            fp.write(f'TP: {TP}\tFP: {FP}\nTN: {TN}\tFN: {FN}\n')
            fp.write(f'ROC curve (area = {roc_auc:0.4f})\n')
            fp.write(f'AUPRC (AP): {np.nan}\n')
            fp.write(f'MCC: {np.nan}\n')
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

##%% MAIN

if __name__ == "__main__":
    # ensure that if no GPU is available, the code runs on limited number of CPU kernels
    tf.config.threading.set_inter_op_parallelism_threads(1) 
    tf.config.threading.set_intra_op_parallelism_threads(1)

    data_path = "../../data/"

    exp = sys.argv[1]
    fold_in = int(sys.argv[3])

    file_path = data_path + exp

    train = int(sys.argv[2])

    data_inference = Data_inference(data_path)

    seq_test, map, labels = data_inference.loading(file_path)

    clean_amino2count_test, clean_seq_test = data_inference.cleaning(seq_test)
    #amino_names = data_inference.counting(clean_amino2count_test)
    padded_seq_test = data_inference.tokens(clean_seq_test)
    indexes = np.arange(len(clean_seq_test))
    k = 5
    batch_size = int(sys.argv[4])#128
    epochs = int(sys.argv[5])#50
    learning_rate = float(sys.argv[6])#0.001

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    contbl = np.zeros(7)
    for fold, [train_vali_idx, x_test] in enumerate(skf.split(indexes, labels)):
        print(f"Fold: {fold}")
        if fold != fold_in:
            continue
        if train:
            
            X_train, X_valid, y_train, y_valid = train_test_split(padded_seq_test[train_vali_idx], labels[train_vali_idx], test_size=.1, random_state=42, stratify=labels[train_vali_idx])
    
            loss_history = LossHistory()
            lrate = tf.keras.callbacks.LearningRateScheduler(step_decay)

            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                                          patience=5, min_lr=0.00001, verbose = 1)

            estop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1)
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                f'results/{exp[:-6]}_{batch_size}_{epochs}_{learning_rate}/fold{fold}.keras',
                save_weights_only=False,
                monitor='val_accuracy',
                mode='max',
                verbose=1,
                save_best_only=True)
            #model = load_model('saved_model.h5')
            model = CNN_LSTM.loading(lr=learning_rate)

            # Reinitialize all the weights of the model
            #for ix, layer in enumerate(model.layers):
            #    if hasattr(model.layers[ix], 'kernel_initializer') and \
            #            hasattr(model.layers[ix], 'bias_initializer'):
            #        weight_initializer = model.layers[ix].kernel_initializer
            #        bias_initializer = model.layers[ix].bias_initializer

            #        old_weights, old_biases = model.layers[ix].get_weights()

            #        model.layers[ix].set_weights([
            #            weight_initializer(shape=old_weights.shape),
            #            bias_initializer(shape=old_biases.shape)])
            expanded_X_train = tf.expand_dims(X_train, -1)
            expanded_X_valid = tf.expand_dims(X_valid, -1)

            h = model.fit([expanded_X_train,expanded_X_train,expanded_X_train,expanded_X_train], tf.expand_dims(y_train, -1),
                        batch_size = batch_size,
                        epochs = epochs,
                        verbose = 1,
                        validation_data = ([expanded_X_valid,expanded_X_valid,expanded_X_valid,expanded_X_valid], tf.expand_dims(y_valid, -1)),
                        callbacks=[reduce_lr, estop, model_checkpoint_callback]
                        #callbacks = [model_checkpoint_callback, estop]
                        )
            print("Test")
            model = load_model(f'results/{exp[:-6]}_{batch_size}_{epochs}_{learning_rate}/fold{fold}.keras')

            # predict
            predictions = model.predict([tf.expand_dims(padded_seq_test[x_test], -1),tf.expand_dims(padded_seq_test[x_test], -1),tf.expand_dims(padded_seq_test[x_test], -1),tf.expand_dims(padded_seq_test[x_test], -1)])

            contbl = np.add(contbl, evaluate_model(map[x_test], labels[x_test], predictions, 0, f'./results/{exp[:-6]}_{batch_size}_{epochs}_{learning_rate}/', fold))
            print("tested")
    # temporary mean over the folds (seperate aggregated used)
    contbl = contbl / k
    TP = contbl[0]
    TN = contbl[1]
    FP = contbl[2]
    FN = contbl[3]
    roc_auc = contbl[4]
    AUPRC = contbl[5]
    MCC_middle = contbl[6]
    print("All Folds together:")
    print(f'TP: {TP}\tFP: {FP}\n TN: {TN}\tFN: {FN}')
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    recall = TP / (TP + FP)
    print(f'accuracy: {accuracy:0.4f}, sensitivity: {sensitivity:0.4f}, specificity: {specificity:0.4f}, recall: {recall:0.4f}')


    #with open(f'results/{exp[:-6]}/result_info.txt', 'a') as fp:
    #    fp.write(f'Fold: all; Time for run: 0 \n')
    #    fp.write(f'accuracy: {accuracy:0.4f}, sensitivity: {sensitivity:0.4f}, specificity: {specificity:0.4f}, recall: {recall:0.4f}\n')
    #    fp.write(f'TP: {TP}\tFP: {FP}\nTN: {TN}\tFN: {FN}\n')
    #    fp.write(f'ROC curve (area = {roc_auc:0.4f})\n')
    #    fp.write(f'AUPRC (AP): {AUPRC:0.4f})\n')
    #    fp.write(f'MCC: {MCC_middle}\n')
    #    fp.write("\n\n")
