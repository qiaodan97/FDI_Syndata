import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# File paths and column combinations
file_path = r"E:\FDI\OneDrive_1_2-16-2025\IEEE118Normal_Corrected.csv"
syn_file_paths = [
    r"E:\FDI\SourceCode\PL_Class_FDI_NotScaled_lr0.03.csv",
    r"E:\FDI\SourceCode\PL_Class_FDI_NotScaled_lr0.03_1~1.3_100PL.csv",
    r"E:\FDI\SourceCode\PL_Class_FDI_NotScaled_lr0.03_1~1.3_50PL.csv"
]
syn_file_labels = ['PL30only', '100%PL', '50%PL']
cols = ["VLA2", "VGA1", "VLM2"]
pl_col = "PL30"  # The column to be used on the x-axis

# Load the original data
data = pd.read_csv(file_path).sample(500)
data['Dataset'] = 'Original'

# Iterate over synthetic files and columns
for syn_file_path, label in zip(syn_file_paths, syn_file_labels):
    syn_data = pd.read_csv(syn_file_path).sample(500)
    syn_data['Dataset'] = label

    combined_data = pd.concat([data, syn_data], ignore_index=True)

    for col in cols:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=pl_col, y=col, data=combined_data, hue='Dataset')

        # Add labels and title
        plt.xlabel(pl_col)
        plt.ylabel(col)
        plt.title(f'Scatter Plot of {pl_col} vs {col} ({label})')

        # Construct the file name and save the plot
        file_name = f'{pl_col}_{col}_{label}.png'
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
        plt.close()

print("Scatter plots saved successfully.")
