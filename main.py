from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler

from DeepDRA import DeepDRA, train, test
from data_loader import RawDataLoader
from evaluation import Evaluation
from utils import *
import random
import torch
import numpy as np
import pandas as pd


def train_DeepDRA(x_cell_train, x_cell_test, x_drug_train, x_drug_test, y_train, y_test, cell_sizes, drug_sizes,device):
    """

    Train and evaluate the DeepDRA model.

    Parameters:
    - X_cell_train (pd.DataFrame): Training data for the cell modality.
    - X_cell_test (pd.DataFrame): Test data for the cell modality.
    - X_drug_train (pd.DataFrame): Training data for the drug modality.
    - X_drug_test (pd.DataFrame): Test data for the drug modality.
    - y_train (pd.Series): Training labels.
    - y_test (pd.Series): Test labels.
    - cell_sizes (list): Sizes of the cell modality features.
    - drug_sizes (list): Sizes of the drug modality features.

    Returns:
    - result: Evaluation result on the test set.
    """

    # Step 1: Define the batch size for training
    batch_size = 64

    # Step 2: Instantiate the combined model
    ae_latent_dim = 50
    mlp_input_dim = 2 * ae_latent_dim
    mlp_output_dim = 1
    num_epochs = 25
    model = DeepDRA(cell_sizes, drug_sizes, ae_latent_dim, ae_latent_dim, mlp_input_dim, mlp_output_dim)
    model.to(device)
    # Step 3: Convert your training data to PyTorch tensors
    x_cell_train_tensor = torch.Tensor(x_cell_train.values)
    x_drug_train_tensor = torch.Tensor(x_drug_train.values)
    x_cell_train_tensor = torch.nn.functional.normalize(x_cell_train_tensor, dim=0)
    x_drug_train_tensor = torch.nn.functional.normalize(x_drug_train_tensor, dim=0)
    y_train_tensor = torch.Tensor(y_train)
    y_train_tensor = y_train_tensor.unsqueeze(1)

    x_cell_train_tensor.to(device)
    x_drug_train_tensor.to(device)
    y_train_tensor.to(device)
    # Compute class weights
    classes = [0, 1]  # Assuming binary classification
    class_weights = torch.tensor(compute_class_weight(class_weight='balanced', classes=classes, y=y_train),
                                 dtype=torch.float32)


    x_cell_train_tensor, x_cell_val_tensor, x_drug_train_tensor, x_drug_val_tensor, y_train_tensor, y_val_tensor = train_test_split(
        x_cell_train_tensor, x_drug_train_tensor, y_train_tensor, test_size=0.1,
        random_state=RANDOM_SEED,
        shuffle=True)

    # Step 4: Create a TensorDataset with the input features and target labels
    train_dataset = TensorDataset(x_cell_train_tensor, x_drug_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(x_cell_val_tensor, x_drug_val_tensor, y_val_tensor)

    # Step 5: Create the train_loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # Step 6: Train the model
    train(model, train_loader, val_loader, num_epochs,class_weights)

    # Step 7: Save the trained model
    torch.save(model, 'DeepDRA.pth')

    # Step 8: Load the saved model
    model = torch.load('DeepDRA.pth')

    # Step 9: Convert your test data to PyTorch tensors
    x_cell_test_tensor = torch.Tensor(x_cell_test.values)
    x_drug_test_tensor = torch.Tensor(x_drug_test.values)
    y_test_tensor = torch.Tensor(y_test)

    # normalize data
    x_cell_test_tensor = torch.nn.functional.normalize(x_cell_test_tensor, dim=0)
    x_drug_test_tensor = torch.nn.functional.normalize(x_drug_test_tensor, dim=0)

    # Step 10: Create a TensorDataset with the input features and target labels for testing
    test_dataset = TensorDataset(x_cell_test_tensor, x_drug_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=len(x_cell_test))

    # Step 11: Test the model
    return test(model, test_loader)


def run(k, is_test=False):
    """
    Run the training and evaluation process k times.

    Parameters:
    - k (int): Number of times to run the process.
    - is_test (bool): If True, run on test data; otherwise, perform train-validation split.

    Returns:
    - history (dict): Dictionary containing evaluation metrics for each run.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Step 1: Initialize a dictionary to store evaluation metrics
    history = {'AUC': [], 'AUPRC': [], "Accuracy": [], "Precision": [], "Recall": [], "F1 score": []}

    # Step 2: Load training data
    train_data, train_drug_screen = RawDataLoader.load_data(data_modalities=DATA_MODALITIES,
                                                            raw_file_directory=RAW_BOTH_DATA_FOLDER,
                                                            screen_file_directory=BOTH_SCREENING_DATA_FOLDER,
                                                            sep="\t")

    # Step 3: Load test data if applicable
    if is_test:
        test_data, test_drug_screen = RawDataLoader.load_data(data_modalities=DATA_MODALITIES,
                                                              raw_file_directory=CCLE_RAW_DATA_FOLDER,
                                                              screen_file_directory=CCLE_SCREENING_DATA_FOLDER,
                                                              sep="\t")
        train_data, test_data = RawDataLoader.data_features_intersect(train_data, test_data)
        X_cell_test, X_drug_test, y_test, cell_sizes, drug_sizes = RawDataLoader.prepare_input_data(test_data,
                                                                                                    test_drug_screen)

    # Step 4: Prepare input data for training
    X_cell_train, X_drug_train, y_train, cell_sizes, drug_sizes = RawDataLoader.prepare_input_data(train_data,
                                                                                                   train_drug_screen)

    rus = RandomUnderSampler(sampling_strategy="majority", random_state=RANDOM_SEED)
    dataset = pd.concat([X_cell_train, X_drug_train], axis=1)
    dataset.index = X_cell_train.index
    dataset, y_train = rus.fit_resample(dataset, y_train)
    X_cell_train = dataset.iloc[:, :sum(cell_sizes)]
    X_drug_train = dataset.iloc[:, sum(cell_sizes):]

    # Step 5: Loop over k runs
    for i in range(k):
        print('Run {}'.format(i))

        # Step 6: If is_test is True, perform random under-sampling on the training data
        if is_test:

            # Step 7: Train and evaluate the DeepDRA model on test data
            results = train_DeepDRA(X_cell_train, X_cell_test, X_drug_train, X_drug_test, y_train, y_test, cell_sizes,
                                    drug_sizes, device)
        else:
            # Step 8: Split the data into training and validation sets
            X_cell_train, X_cell_test, X_drug_train, X_drug_test, y_train, y_test = train_test_split(X_cell_train,
                                                                                                     X_drug_train, y_train,
                                                                                                     test_size=0.2,
                                                                                                     random_state=RANDOM_SEED,
                                                                                                     shuffle=True)
            # Step 9: Train and evaluate the DeepDRA model on the split data
            results = train_DeepDRA(X_cell_train, X_cell_test, X_drug_train, X_drug_test, y_train, y_test, cell_sizes,
                                    drug_sizes, device)

        # Step 10: Add results to the history dictionary
        Evaluation.add_results(history, results)

    # Step 11: Display final results
    Evaluation.show_final_results(history)
    return history


if __name__ == '__main__':
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    run(10, is_test=False)
