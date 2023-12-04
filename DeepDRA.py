import torch.nn as nn
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from autoencoder import Autoencoder
from evaluation import Evaluation
from mlp import MLP
from utils import *


class DeepDRA(nn.Module):
    """
    DeepDRA (Deep Drug Response Anticipation) is a neural network model composed of two autoencoders for cell and drug modalities
    and an MLP for integrating the encoded features and making predictions.

    Parameters:
    - cell_modality_sizes (list): Sizes of the cell modality features.
    - drug_modality_sizes (list): Sizes of the drug modality features.
    - cell_ae_latent_dim (int): Latent dimension for the cell autoencoder.
    - drug_ae_latent_dim (int): Latent dimension for the drug autoencoder.
    - mlp_input_dim (int): Input dimension for the MLP.
    - mlp_output_dim (int): Output dimension for the MLP.
    """

    def __init__(self, cell_modality_sizes, drug_modality_sizes, cell_ae_latent_dim, drug_ae_latent_dim, mlp_input_dim,
                 mlp_output_dim):
        super(DeepDRA, self).__init__()

        # Initialize cell and drug autoencoders
        self.cell_autoencoder = Autoencoder(sum(cell_modality_sizes), cell_ae_latent_dim)
        self.drug_autoencoder = Autoencoder(sum(drug_modality_sizes), drug_ae_latent_dim)

        # Store modality sizes
        self.cell_modality_sizes = cell_modality_sizes
        self.drug_modality_sizes = drug_modality_sizes

        # Initialize MLP
        self.mlp = MLP(mlp_input_dim, mlp_output_dim)

    def forward(self, cell_x, drug_x):
        """
        Forward pass of the DeepDRA model.

        Parameters:
        - cell_x (torch.Tensor): Input tensor for cell modality.
        - drug_x (torch.Tensor): Input tensor for drug modality.

        Returns:
        - cell_decoded (torch.Tensor): Decoded tensor for the cell modality.
        - drug_decoded (torch.Tensor): Decoded tensor for the drug modality.
        - mlp_output (torch.Tensor): Output tensor from the MLP.
        """
        # Encode and decode cell modality
        cell_encoded = self.cell_autoencoder.encoder(cell_x)
        cell_decoded = self.cell_autoencoder.decoder(cell_encoded)

        # Encode and decode drug modality
        drug_encoded = self.drug_autoencoder.encoder(drug_x)
        drug_decoded = self.drug_autoencoder.decoder(drug_encoded)

        # Concatenate encoded cell and drug features and pass through MLP
        mlp_output = self.mlp(torch.cat((cell_encoded, drug_encoded), 1))

        return cell_decoded, drug_decoded, mlp_output

    def compute_l1_loss(self, w):
        """
        Computes L1 regularization loss.

        Parameters:
        - w (torch.Tensor): Input tensor.

        Returns:
        - loss (torch.Tensor): L1 regularization loss.
        """
        return torch.abs(w).sum()

    def compute_l2_loss(self, w):
        """
        Computes L2 regularization loss.

        Parameters:
        - w (torch.Tensor): Input tensor.

        Returns:
        - loss (torch.Tensor): L2 regularization loss.
        """
        return torch.square(w).sum()


def train(model, train_loader, num_epochs):
    """
    Trains the DeepDRA (Deep Drug Response Anticipation) model.

    Parameters:
    - model (DeepDRA): The DeepDRA model to be trained.
    - train_loader (DataLoader): DataLoader for the training dataset.
    - num_epochs (int): Number of training epochs.
    """
    autoencoder_loss_fn = nn.MSELoss()
    mlp_loss_fn = nn.BCELoss()

    mlp_optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = lr_scheduler.ReduceLROnPlateau(mlp_optimizer, mode='min', factor=0.8, patience=5, verbose=True)

    for epoch in range(num_epochs):
        for batch_idx, (cell_data, drug_data, target) in enumerate(train_loader):
            mlp_optimizer.zero_grad()

            # Forward pass
            cell_decoded_output, drug_decoded_output, mlp_output = model(cell_data, drug_data)

            # Compute losses
            cell_ae_loss = autoencoder_loss_fn(cell_decoded_output, cell_data)
            drug_ae_loss = autoencoder_loss_fn(drug_decoded_output, drug_data)
            mlp_loss = mlp_loss_fn(mlp_output, target)

            # Total loss is the sum of autoencoder losses and MLP loss
            total_loss = drug_ae_loss + cell_ae_loss + mlp_loss

            # Backward pass and optimization
            total_loss.backward()
            mlp_optimizer.step()

            # Print progress
            if batch_idx % 200 == 0:
                print('Epoch [{}/{}], Total Loss: {:.4f}'.format(
                    epoch + 1, num_epochs, total_loss.item()))

        # Learning rate scheduler step
        scheduler.step(total_loss)

    # Save the trained model
    torch.save(model.state_dict(), MODEL_FOLDER + 'DeepDRA.pth')


def test(model, test_loader, reverse=False):
    """
    Tests the given model on the test dataset using evaluation metrics.

    Parameters:
    - model: The trained model to be evaluated.
    - test_loader: DataLoader for the test dataset.
    - reverse (bool): If True, reverse the predictions for evaluation.

    Returns:
    - result: The evaluation result based on the chosen metrics.
    """
    # Set model to evaluation mode
    model.eval()

    # Initialize lists to store predictions and ground truth labels
    all_predictions = []
    all_labels = []

    # Iterate over the test dataset
    for i, (test_cell_loader, test_drug_loader, labels) in enumerate(test_loader):
        # Forward pass through the model
        with torch.no_grad():
            decoded_cell_output, decoded_drug_output, mlp_output = model(test_cell_loader, test_drug_loader)

        # Apply reverse if specified
        predictions = 1 - mlp_output if reverse else mlp_output

        # # Store predictions and ground truth labels
        # all_predictions.extend(predictions.cpu().numpy())
        # all_labels.extend(labels.cpu().numpy())

    # Evaluate the predictions using the specified metrics
    result = Evaluation.evaluate(labels, predictions)

    return result

