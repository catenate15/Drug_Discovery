import pandas as pd
from visualization import visualize_molecule

def main():
    # Load the activation maps CSV
    activation_df = pd.read_csv('activation_maps.csv')
    
    # Select molecules to visualize, for example, those with a binary prediction of 1 (active)
    molecules_to_visualize = activation_df[activation_df['BINARY_PREDICTION'] == 1]

    # Visualize molecules
    for idx, row in molecules_to_visualize.iterrows():
        smiles = row['SMILES']
        activation_map = row[[f'bit_{i}' for i in range(2048)]].values  # Assuming 2048 bits
        visualize_molecule(smiles, activation_map, f"visualizations/{idx}.png")

if __name__ == "__main__":
    main()


# import pandas as pd
# from visualization import visualize_molecule
# from activation_map import compute_activation_map
# from fingerprint_classifier import FingerprintClassifier
# from fingerprint_data_module import FingerprintDataModule

# def main():
#     # Load the trained model (update with your actual model)
#     model = FingerprintClassifier.load_from_checkpoint('model_checkpoint.ckpt')
    
#     # Load predictions CSV
#     predictions_df = pd.read_csv('predictions.csv')
    
#     # Select molecules to visualize, for example, those with a prediction above a threshold
#     molecules_to_visualize = predictions_df[predictions_df['PREDICTION'] > 0.5]
    
#     # Compute activation maps for selected molecules
#     activation_maps = compute_activation_map(model, molecules_to_visualize)
    
#     # Visualize molecules
#     for idx, row in molecules_to_visualize.iterrows():
#         smiles = row['SMILES']
#         activation_map = activation_maps[smiles]
#         visualize_molecule(activation_map, smiles, f"visualizations/{smiles}.png")

# if __name__ == "__main__":
#     main()
