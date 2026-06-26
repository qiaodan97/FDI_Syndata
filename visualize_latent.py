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
import torch

# Assuming `vae.encoder` is the encoder part of your VAE
# and `data_loader_real` and `data_loader_synthetic` are your test data loaders

def extract_latent_representations(vae, data_loader, device):
    vae.eval()  # Set VAE to evaluation mode
    latents = []

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            mean, log_var = vae.encoder(batch)
            z = mean
            latents.append(z.cpu().numpy())

    return np.concatenate(latents, axis=0)



# Main script
def main():
    # Parameters
    input_dim = 418 #64 #418
    hidden_dim_grid = [[512, 256]]
    latent_dim_grid = [16] #8, 16
    batch_size_grid = [128]
    epochs_grid = [50]
    learning_rate_grid = [1e-5]

    selected_columns = [f'PL{i}' for i in range(1, 65)]
    grid = itertools.product(hidden_dim_grid, latent_dim_grid, batch_size_grid, epochs_grid, learning_rate_grid)
    with open(r"vae_output.txt", "a") as file:
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
            data = pd.read_csv(r"E:\FDI\OneDrive_1_1-15-2025\118\Datasets\IEEE118NormalWithPd_Qd.csv").drop(columns=columns_to_drop, errors='ignore').to_numpy()
            # data = pd.read_csv(r"E:\FDI\OneDrive_1_1-15-2025\118\Datasets\IEEE118NormalWithPd_Qd.csv")[selected_columns].to_numpy()
            # scaler = MinMaxScaler()
            # data_scaled = scaler.fit_transform(data.values)

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
            print(train_losses)
            print(val_losses)
            epochs = list(range(1, len(train_losses) + 1))

            plt.plot(epochs, train_losses, marker='o', linestyle='-', color='b', label="Training Loss")
            plt.xlabel("Epochs")
            plt.ylabel("Loss Value")  # Ensures y-axis correctly represents loss
            plt.title("Training Loss Curve")
            plt.legend()
            plt.grid(True)
            plt.savefig("training_loss_whole.png", dpi=300, bbox_inches="tight")


            plt.plot(epochs, val_losses, marker='o', linestyle='-', color='b', label="Validation Loss")
            plt.xlabel("Epochs")
            plt.ylabel("Loss Value")  # Ensures y-axis correctly represents loss
            plt.title("Validation Loss Curve")
            plt.legend()
            plt.grid(True)
            plt.savefig("validating_loss_whole.png", dpi=300, bbox_inches="tight")

            # Save the trained model
            # torch.save(model.state_dict(), "vae_model.pth")

            test_tensor = torch.tensor(X_test, dtype=torch.float32)
            test_loader = DataLoader(TensorDataset(test_tensor), batch_size=batch_size, shuffle=False)
            test_error = test_reconstruction_error(model, test_loader, device)
            print("Final real test Recon error", round(test_error, 3), file=file, flush=True)

            X_test_syn = pd.read_csv(r"E:\FDI\OneDrive_1_1-15-2025\118\Datasets\PL_class.csv").to_numpy()
            # X_test_syn = pd.read_csv(r"E:\FDI\OneDrive_1_1-15-2025\118\Datasets\PL_class.csv")[selected_columns].to_numpy()
            random_indices = np.random.choice(X_test_syn.shape[0], size=7001, replace=False)
            X_test_syn = X_test_syn[random_indices,:473]
            test_tensor_syn = torch.tensor(X_test_syn, dtype=torch.float32)
            test_loader_syn = DataLoader(TensorDataset(test_tensor_syn), batch_size=batch_size, shuffle=False)
            test_error_syn = test_reconstruction_error(model, test_loader_syn, device)
            test_loss_syn = test(model, test_loader_syn, device)
            print("Final PL syn test Loss", round(test_loss_syn, 3), "\n", file=file, flush=True)
            print("Final PL syn test Recon error", round(test_error_syn, 3), "\n", file=file, flush=True)

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
