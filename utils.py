import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import sklearn as sk
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist, squareform
import h2o
from h2o.estimators import H2ODeepLearningEstimator
from sklearn.impute import SimpleImputer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
from sklearn import metrics
from copy import deepcopy
import pyreadr
import requests
from time import time
from math import ceil
from statsmodels.stats.weightstats import ttest_ind
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.model_selection import KFold

DATA_FOLDER = 'data'
RES_DATA_FOLDER = os.path.join(DATA_FOLDER, 'res')
TEST_DATA_FOLDER = os.path.join(DATA_FOLDER, 'final_test_data')
TEST_TCGA_DATA_FOLDER = os.path.join(DATA_FOLDER, 'TCGA_test_data')
SIM_DATA_FOLDER = os.path.join(DATA_FOLDER, 'similarity_data')
RAW_DATA_FOLDER = os.path.join(DATA_FOLDER, 'raw_data')
RAW_BOTH_DATA_FOLDER = os.path.join(DATA_FOLDER, 'CTRP_GDSC_Data')
DRUG_DATA_FOLDER = os.path.join(DATA_FOLDER, 'drug_data')

NEW_RAW_DATA_FOLDER = os.path.join(DATA_FOLDER, 'new_raw_data')
GDSC_RAW_DATA_FOLDER = os.path.join(DATA_FOLDER, 'GDSC_data')
CCLE_RAW_DATA_FOLDER = os.path.join(DATA_FOLDER, 'CCLE_raw')

CTRP_FOLDER = os.path.join(DATA_FOLDER, 'CTRP')
GDSC_FOLDER = os.path.join(DATA_FOLDER, 'GDSC')
CCLE_FOLDER = os.path.join(DATA_FOLDER, 'CCLE')

MODEL_FOLDER = os.path.join(DATA_FOLDER, 'model')

CTRP_EXPERIMENT_FILE = os.path.join(CTRP_FOLDER, 'v20.meta.per_experiment.txt')
CTRP_COMPOUND_FILE = os.path.join(CTRP_FOLDER, 'v20.meta.per_compound.txt')
CTRP_CELLLINE_FILE = os.path.join(CTRP_FOLDER, 'v20.meta.per_cell_line.txt')
CTRP_AUC_FILE = os.path.join(CTRP_FOLDER, 'v20.data.curves_post_qc.txt')

GDSC_AUC_FILE = os.path.join(GDSC_FOLDER, 'GDSC2_fitted_dose_response.csv')
GDSC_cnv_data_FILE = os.path.join(GDSC_FOLDER, 'cnv_abs_copy_number_picnic_20191101.csv')
GDSC_methy_data_FILE = os.path.join(GDSC_FOLDER, 'F2_METH_CELL_DATA.txt')
GDSC_methy_sampleIds_FILE = os.path.join(GDSC_FOLDER, 'methSampleId_2_cosmicIds.xlsx')
GDSC_exp_data_FILE = os.path.join(GDSC_FOLDER, 'Cell_line_RMA_proc_basalExp.txt')
GDSC_exp_sampleIds_FILE = os.path.join(GDSC_FOLDER, 'E-MTAB-3610.sdrf.txt')
GDSC_mut_data_FILE = os.path.join(GDSC_FOLDER, 'mutations_all_20230202.csv')
GDSC_SCREENING_DATA_FOLDER = os.path.join(GDSC_RAW_DATA_FOLDER, 'drug_screening_matrix_GDSC.tsv')
CCLE_SCREENING_DATA_FOLDER = os.path.join(CCLE_RAW_DATA_FOLDER, 'drug_screening_matrix_ccle.tsv')
BOTH_SCREENING_DATA_FOLDER = os.path.join(RAW_BOTH_DATA_FOLDER, 'drug_screening_matrix_gdsc_ctrp.tsv')

CCLE_mut_data_FILE = os.path.join(CCLE_FOLDER, 'CCLE_mutations.csv')

TABLE_RESULTS_FILE = os.path.join(DATA_FOLDER, 'drug_screening_table.tsv')
MATRIX_RESULTS_FILE = os.path.join(DATA_FOLDER, 'drug_screening_matrix.tsv')

MODEL_FILE = os.path.join(MODEL_FOLDER, 'trained_model_V1_EMDP.sav')
TEST_FILE = os.path.join(TEST_DATA_FOLDER, 'test.gzip')
RESULT_FILE = os.path.join(RES_DATA_FOLDER, 'result.tsv')

TCGA_DATA_FOLDER = os.path.join(DATA_FOLDER, 'TCGA_test_data')
TCGA_SCREENING_DATA = os.path.join(TCGA_DATA_FOLDER, 'TCGA_screening_matrix.tsv')

BUILD_SIM_MATRICES = True  # Make this variable True to build similarity matrices from raw data
SIM_KERNEL = {'cell_CN': ('euclidean', 0.001), 'cell_exp': ('euclidean', 0.01), 'cell_methy': ('euclidean', 0.1),
              'cell_mut': ('jaccard', 1), 'drug_DT': ('jaccard', 1), 'drug_comp': ('euclidean', 0.001),
              'drug_desc': ('euclidean', 0.001), 'drug_finger': ('euclidean', 0.001)}
SAVE_MODEL = False  # Change it to True to save the trained model
VARIATIONAL_AUTOENCODERS = False
# DATA_MODALITIES=['cell_CN','cell_exp','cell_methy','cell_mut','drug_comp','drug_DT'] # Change this list to only consider specific data modalities
DATA_MODALITIES = ['cell_mut', 'drug_desc', 'drug_finger']
RANDOM_SEED = 42  # Must be used wherever can be used


def data_modalities_abbreviation():
    abb = []
    if 'cell_CN' in DATA_MODALITIES:
        abb.append('C')
    if 'cell_exp' in DATA_MODALITIES:
        abb.append('E')
    if 'cell_mut' in DATA_MODALITIES:
        abb.append('M')
    if 'cell_methy' in DATA_MODALITIES:
        abb.append('T')
    if 'drug_DT' in DATA_MODALITIES:
        abb.append('D')
    if 'drug_comp' in DATA_MODALITIES:
        abb.append('P')
    return ''.join(abb)


""" TRAIN_INTEGRATION_METHOD used for each cell's and drug_data's data definitions: 
SIMILARITY: A kernel based integration method in which based on the similarity of each cell's data with the training cell's
data the input features for the multi layer perceptron (MLP) is constructed. The similarity function used could be different for
each data modality (euclidean, jaccard,l1_norm, or ...)

AUTO_ENCODER_V1: In this version of integrating multi-omics, for each data modality an autoencoder is trained to reduce the
dimension of the features and finally a concatenation of each autoencoder's latent space builds up the input layer of the MLP.

AUTO_ENCODER_V2: In this version of integrating multi-omics data, we train a big autoencoder which reduces the dimension of 
all the different data modalities features at the same time to a smaller feature space. This version of integrating could
take a lot of memory and time to integrate the data and might be computationally expensive.

AUTO_ENCODER_V3: IN this version of integrating multi-omics data, we train an autoencoder for all the modalities kinda same as 
the autoencoder version 2 but with this difference that the encoder and decoder layers are separate from each other and 
just the latent layer is shared among different data modalities.
"""
