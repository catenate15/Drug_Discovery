import torch
import argparse
import pandas as pd
from fingerprint_classifier import FingerprintClassifier
from fingerprint_data_module import FingerprintDataModule
from datetime import datetime

def load_model(checkpoint_path):
    model = FingerprintClassifier.load_from_checkpoint(checkpoint_path)
    model.eval()
    return model

def predict_with_ids(model, dataset):
    model.eval()
    predictions = []
    compound_ids = []
    smiles_list = []

    with torch.no_grad():
        for idx in range(len(dataset)):
            fingerprint, _ = dataset[idx]
            compound_id = dataset.df.iloc[idx][dataset.id_column]
            smiles = dataset.df.iloc[idx][dataset.smiles_column]

            fingerprint = fingerprint.unsqueeze(0)
            output = model(fingerprint)
            prediction = torch.sigmoid(output)

            predictions.append(prediction.item())
            compound_ids.append(compound_id)
            smiles_list.append(smiles)

    return compound_ids, smiles_list, predictions

def get_activation_maps(model, dataset):
    model.eval()
    activation_data = []

    for idx in range(len(dataset)):
        fingerprint, _ = dataset[idx]
        fingerprint = fingerprint.unsqueeze(0).float()
        fingerprint.requires_grad = True
        output = model(fingerprint)

        model.zero_grad()
        output.backward(torch.ones_like(output))

        fingerprint_gradients = fingerprint.grad.abs().detach()

        activation_data.append(fingerprint_gradients.squeeze().numpy().tolist())

    return activation_data

def main(file_path):
    filebase = file_path.rsplit('.', 1)[0]
    model_checkpoint_path = "./tb_logs/my_model/version_23/checkpoints/epoch=9-step=140.ckpt"
    model = load_model(model_checkpoint_path)


    data_module = FingerprintDataModule(filebase=filebase, target_column='target')
    data_module.setup()
    val_dataset = data_module.val_dataset

    compound_ids, smiles_list, predictions = predict_with_ids(model, val_dataset)
    activation_maps_data = get_activation_maps(model, val_dataset)

    activation_df = pd.DataFrame({
        'COMPOUND_ID': compound_ids,
        'SMILES': smiles_list,
        'PREDICTION': predictions,
        'BINARY_PREDICTION': [int(pred > 0.5) for pred in predictions]
    })

    activation_map_list = []
    for activation_map in activation_maps_data:
        activation_map_list.append(activation_map)

    # Convert list of activation maps to DataFrame
    activation_map_df = pd.DataFrame(activation_map_list, columns=[f'bit_{i}' for i in range(len(activation_maps_data[0]))])

    # Get current time to use in filename
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_filename = f'activation_maps_{timestamp}.csv'

    # Combine the activation map DataFrame with the activation_df
    combined_df = pd.concat([activation_df, activation_map_df], axis=1)
    combined_df.to_csv(output_filename, index=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate activation maps for a trained model.')
    parser.add_argument('file_path', type=str, help='The file path to the CSV data file.')
    args = parser.parse_args()
    main(args.file_path)

