import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import itertools
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super(VAE, self).__init__()

        # Encoder: Build progressively smaller layers using hidden_dims
        encoder_layers = []
        for i in range(len(hidden_dims)):
            if i == 0:
                encoder_layers.append(nn.Linear(input_dim, hidden_dims[i]))
            else:
                encoder_layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
            encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)

        # Latent space
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        # Decoder: Build progressively larger layers using reversed hidden_dims
        decoder_layers = []
        reversed_hidden_dims = hidden_dims[::-1]
        for i in range(len(reversed_hidden_dims)):
            if i == 0:
                decoder_layers.append(nn.Linear(latent_dim, reversed_hidden_dims[i]))
            else:
                decoder_layers.append(nn.Linear(reversed_hidden_dims[i - 1], reversed_hidden_dims[i]))
            decoder_layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*decoder_layers)

        # Final output layer
        self.fc_output = nn.Linear(reversed_hidden_dims[-1], input_dim)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder(z)
        return torch.sigmoid(self.fc_output(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Define the loss function
def loss_function(recon_x, x, mu, logvar):
    recon_loss = nn.MSELoss()(recon_x, x)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + kl_div

def test(model, test_loader, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            data = data[0].to(device)
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            test_loss += loss.item()
    return test_loss / len(test_loader.dataset)

# Train the model
def train(model, train_loader, optimizer, device):
    model.train()
    train_loss = 0
    for data in train_loader:
        data = data[0].to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    return train_loss / len(train_loader.dataset)

def reconstruction_error(recon_x, x):
    return nn.MSELoss()(recon_x, x)

def test_reconstruction_error(model, test_loader, device):
    model.eval()
    total_error = 0
    num_samples = 0

    with torch.no_grad():
        for data in test_loader:
            data = data[0].to(device)  # Assuming DataLoader returns (data, labels)
            recon_batch, mu, logvar = model(data)
            error = reconstruction_error(recon_batch, data)
            total_error += error.item() * data.size(0)
            num_samples += data.size(0)

    return total_error / num_samples  # Mean reconstruction error


def plot_loss(train_losses, save_path="training_loss.png"):
    plt.figure(figsize=(16, 12))
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', linestyle='-', color='b',
             label="Loss")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)

def extract_latent_representations(vae, data_loader, device):
    vae.eval()
    latent_vectors = []

    for batch in data_loader:
        batch = batch[0].to(device)
        z = vae.encoder(batch)  # Extract latent mean
        latent_vectors.append(z.detach().cpu().numpy())

    return np.concatenate(latent_vectors, axis=0)

# Main script
def main():
    # Parameters
    input_dim = 418 #64 #450
    hidden_dim_grid = [[512]]
    latent_dim_grid = [16] #8, 16
    batch_size_grid = [128]
    epochs_grid = [50]
    learning_rate_grid = [1e-6]
    real_data_path = r"E:\FDI\OneDrive_1_2-16-2025\IEEE118Normal_Corrected.csv"

    syn_data_path = r"E:\FDI\SourceCode\PL_Class_FDI_NotScaled.csv"

    selected_columns = [f'PL{i}' for i in range(1, 65)]
    grid = itertools.product(hidden_dim_grid, latent_dim_grid, batch_size_grid, epochs_grid, learning_rate_grid)
    with open(r"E:\FDI\SourceCode\result\vae_output.txt", "a") as file:
        for hidden_dim, latent_dim, batch_size, epochs, learning_rate in grid:
            print(
                f"Running: hidden_dim={hidden_dim}, latent_dim={latent_dim}, batch_size={batch_size}, epochs={epochs}, learning_rate={learning_rate}", file=file, flush=True)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            columns_to_drop = ['QG12', 'QG19', 'QG11', 'QG34', 'QG45', 'QG2', 'QG47', 'QG33',
                               'QG30', 'QG10', 'QG46', 'QG9', 'QG20', 'QG44', 'QG3', 'QG24',
                               'QG37', 'QG18', 'QG14', 'QG43', 'QG15', 'QG39', 'QG26', 'QG29',
                               'QG35', 'QG22', 'QG51', 'QG54', 'QG42', 'QG31', 'QG6', 'QG21',
                               'QG48', 'QG17', 'QG5', 'QG25', 'QG7', 'QG28', 'QG38', 'QG8',
                               'QG16', 'QG13', 'QG1', 'QG23', 'QG4', 'QG49', 'QG50', 'QG52',
                               'QG40', 'QG41', 'QG27', 'Label', 'QG32', 'QG36', 'QG53']

            # Load and preprocess data
            data = pd.read_csv(real_data_path).drop(columns=columns_to_drop, errors='ignore')#.to_numpy()
            print(data.shape)
            print(data.columns)
            # Independent variables
            vgm_columns = [col for col in data.columns if 'VGM' in col]
            pg_columns = [col for col in data.columns if 'PG' in col]
            pl_columns = [col for col in data.columns if 'PL' in col]
            ql_columns = [col for col in data.columns if 'QL' in col]

            # Dependent variables
            vlm_columns = [col for col in data.columns if 'VLM' in col]
            vla_columns = [col for col in data.columns if 'VLA' in col]
            vga_columns = [col for col in data.columns if 'VGA' in col]
            selected_columns = [vlm_columns + vla_columns + vga_columns + vgm_columns + pg_columns + pl_columns + ql_columns]
            data = data[selected_columns].to_numpy()
            # data = pd.read_csv(r"E:\FDI\OneDrive_1_1-15-2025\118\Datasets\IEEE118NormalWithPd_Qd.csv")[selected_columns].to_numpy()
            # scaler = MinMaxScaler()
            # data_scaled = scaler.fit_transform(data.values)

            X_test_syn = pd.read_csv(syn_data_path)[selected_columns].to_numpy()
            # X_test_syn = pd.read_csv(r"E:\FDI\OneDrive_1_1-15-2025\118\Datasets\PL_class.csv")[selected_columns].to_numpy()
            random_indices = np.random.choice(X_test_syn.shape[0], size=7001, replace=False)
            X_test_syn = X_test_syn[random_indices,:473]
            print(X_test_syn.shape)
            print(X_test_syn.columns)

            # Split data into training and testing sets (80:20)
            X_train, X_test = train_test_split(data, test_size=0.2, random_state=42)
            X_val, X_test = train_test_split(X_test, test_size=0.5, random_state=42)

            train_tensor = torch.tensor(X_train, dtype=torch.float32)
            val_tensor = torch.tensor(X_val, dtype=torch.float32)

            # Create DataLoader
            train_loader = DataLoader(TensorDataset(train_tensor), batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(TensorDataset(val_tensor), batch_size=batch_size, shuffle=False)

            # Model, optimizer
            model = VAE(input_dim, hidden_dim, latent_dim).to(device)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            best_val_loss = float("inf")
            patience = 5  # Stop if validation loss doesn't improve for 5 epochs
            no_improve_epochs = 0

            train_losses = []
            val_losses = []
            # Training loop
            for epoch in range(epochs):
                # print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
                # Train model
                train_loss = train(model, train_loader, optimizer, device)
                val_loss = test(model, val_loader, device)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print("Epoch: ", epoch, ", train loss: ", train_loss, ", val_loss: ", val_loss, file=file, flush=True)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improve_epochs = 0  # Reset counter
                    # torch.save(model.state_dict(), "best_vae_model.pth")  # Save best model
                else:
                    no_improve_epochs += 1

                if no_improve_epochs >= patience:
                    print("Early stopping triggered!", file=file, flush=True)
                    break
            torch.save(model.state_dict(), r"E:\FDI\SourceCode\result\vae_pl.pth")
            print(train_losses)
            print(val_losses)
            epochs = list(range(1, len(train_losses) + 1))

            plt.plot(epochs, train_losses, marker='o', linestyle='-', color='b', label="Training Loss")
            plt.xlabel("Epochs")
            plt.ylabel("Loss Value")  # Ensures y-axis correctly represents loss
            plt.title("Training Loss Curve")
            plt.legend()
            plt.grid(True)
            plt.savefig(r"E:\FDI\SourceCode\result\training_loss.png", dpi=300, bbox_inches="tight")


            plt.plot(epochs, val_losses, marker='o', linestyle='-', color='b', label="Validation Loss")
            plt.xlabel("Epochs")
            plt.ylabel("Loss Value")  # Ensures y-axis correctly represents loss
            plt.title("Validation Loss Curve")
            plt.legend()
            plt.grid(True)
            plt.savefig(r"E:\FDI\SourceCode\result\validating_loss.png", dpi=300, bbox_inches="tight")

            # torch.load("vae_whole.pth")

            test_tensor = torch.tensor(X_test, dtype=torch.float32)
            test_loader = DataLoader(TensorDataset(test_tensor), batch_size=batch_size, shuffle=False)
            test_error = test_reconstruction_error(model, test_loader, device)
            print("Final real test Recon error", round(test_error, 3), file=file, flush=True)

            test_tensor_syn = torch.tensor(X_test_syn, dtype=torch.float32)
            test_loader_syn = DataLoader(TensorDataset(test_tensor_syn), batch_size=batch_size, shuffle=False)
            test_error_syn = test_reconstruction_error(model, test_loader_syn, device)
            test_loss_syn = test(model, test_loader_syn, device)
            print("Final PL syn test Loss", round(test_loss_syn, 3), "\n", file=file, flush=True)
            print("Final PL syn test Recon error", round(test_error_syn, 3), "\n", file=file, flush=True)

            z_real = extract_latent_representations(model, test_loader, device)
            z_synthetic = extract_latent_representations(model, test_loader_syn, device)
            z_combined = np.vstack((z_real, z_synthetic))
            labels = np.array([0] * len(z_real) + [1] * len(z_synthetic))

            print(f"Shape of z_combined: {z_combined.shape}")  # Expected: (num_real + num_synthetic, latent_dim)
            print(f"Shape of labels: {labels.shape}")  # Expected: (num_real + num_synthetic,)

            pca = PCA(n_components=2)
            z_pca = pca.fit_transform(z_combined)

            plt.figure(figsize=(16, 12))
            plt.scatter(z_pca[labels == 0, 0], z_pca[labels == 0, 1], label='Real Data', alpha=0.6)
            plt.scatter(z_pca[labels == 1, 0], z_pca[labels == 1, 1], label='Synthetic Data', alpha=0.6)
            plt.legend()
            plt.title("PCA Projection of Latent Space")
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.savefig(r"E:\FDI\SourceCode\result\pca_latent.png", dpi=300)


            tsne = TSNE(n_components=2, perplexity=30, random_state=42)
            z_tsne = tsne.fit_transform(z_combined)

            plt.figure(figsize=(16, 12))
            plt.scatter(z_tsne[labels == 0, 0], z_tsne[labels == 0, 1], label='Real Data', alpha=0.6)
            plt.scatter(z_tsne[labels == 1, 0], z_tsne[labels == 1, 1], label='Synthetic Data', alpha=0.6)
            plt.legend()
            plt.title("t-SNE Projection of Latent Space")
            plt.xlabel("t-SNE Component 1")
            plt.ylabel("t-SNE Component 2")
            plt.savefig(r"E:\FDI\SourceCode\result\tsne_latent.png", dpi=300)

            # X_test_syn = pd.read_csv(r"E:\FDI\OneDrive_1_1-15-2025\118\Datasets\Combine_class.csv").to_numpy()
            # random_indices = np.random.choice(X_test_syn.shape[0], size=14001, replace=False)
            # X_test_syn = X_test_syn[random_indices,:473]
            # test_tensor_syn = torch.tensor(X_test_syn, dtype=torch.float32)
            # test_loader_syn = DataLoader(TensorDataset(test_tensor_syn), batch_size=batch_size, shuffle=False)
            # test_error_syn = test_reconstruction_error(model, test_loader_syn, device)
            # print("Final combined syn test Loss", round(test_error_syn, 3), file=file, flush=True)


            # X_test_syn = pd.read_csv(r"E:\FDI\OneDrive_1_1-15-2025\118\Datasets\QL_class.csv").to_numpy()
            # random_indices = np.random.choice(X_test_syn.shape[0], size=14001, replace=False)
            # X_test_syn = X_test_syn[random_indices,:473]
            # test_tensor_syn = torch.tensor(X_test_syn, dtype=torch.float32)
            # test_loader_syn = DataLoader(TensorDataset(test_tensor_syn), batch_size=batch_size, shuffle=False)
            # test_error_syn = test_reconstruction_error(model, test_loader_syn, device)
            # print("Final QL syn test Loss", round(test_error_syn, 3), file=file, flush=True)
            #
            # X_test_syn = pd.read_csv(r"E:\FDI\OneDrive_1_1-15-2025\118\Datasets\VGM_class.csv").to_numpy()
            # random_indices = np.random.choice(X_test_syn.shape[0], size=14001, replace=False)
            # X_test_syn = X_test_syn[random_indices,:473]
            # test_tensor_syn = torch.tensor(X_test_syn, dtype=torch.float32)
            # test_loader_syn = DataLoader(TensorDataset(test_tensor_syn), batch_size=batch_size, shuffle=False)
            # test_error_syn = test_reconstruction_error(model, test_loader_syn, device)
            # print("Final VGM syn test Loss", round(test_error_syn, 3), "\n", file=file, flush=True)

    # torch.load("vae_model.pth")
    # # Reconstruct a few samples and visualize
    # model.eval()
    # with torch.no_grad():
    #     sample_data = torch.tensor(X_test[:10], dtype=torch.float32).to(device)
    #     reconstructed, _, _ = model(sample_data)
    #
    # # Show original vs reconstructed
    # print("Original Samples:")
    # print(scaler.inverse_transform(sample_data.cpu().numpy()))
    # print("\nReconstructed Samples:")
    # print(scaler.inverse_transform(reconstructed.cpu().numpy()))


if __name__ == "__main__":
    main()
