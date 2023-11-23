from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
import pandas as pd
import io
from PIL import Image

def visualize_molecule(smiles, activation_map, output_file=None):
    mol = Chem.MolFromSmiles(smiles)
    drawer = rdMolDraw2D.MolDraw2DCairo(350, 400)
    opts = drawer.drawOptions()
    
    # Set custom colors for atoms and bonds
    atom_colors = {}
    bond_colors = {}
    for idx, value in enumerate(activation_map):
        if value > 0.5:  # Active
            color = (0, 1, 0, 1)  # Green with full opacity
        else:  # Inactive
            color = (1, 0, 0, 1)  # Red with full opacity
        # Assuming the index is for atoms; if it's for bonds, switch to bond indexing
        atom_colors[idx] = color

    # Draw the molecule
    Chem.rdDepictor.Compute2DCoords(mol)
    drawer.DrawMolecule(mol, highlightAtoms=atom_colors.keys(), highlightAtomColors=atom_colors)
    drawer.FinishDrawing()

    # Save or display the image
    if output_file:
        with open(output_file, "wb") as f:
            f.write(drawer.GetDrawingText())
    else:
        bio = io.BytesIO(drawer.GetDrawingText())
        return Image.open(bio)

# Example usage
activation_maps_df = pd.read_csv('activation_maps.csv')
molecule_index = 0
smiles = activation_maps_df.iloc[molecule_index]['SMILES']
activation_map = activation_maps_df.iloc[molecule_index, 4:].values.astype(float)
visualize_molecule(smiles, activation_map, 'molecule_visualization.png')




# from rdkit import Chem
# import pandas as pd
# from rdkit.Chem import Draw
# import matplotlib.colors as mcolors
# from rdkit.Chem.Draw import rdMolDraw2D

# def visualize_molecule(smiles, activation_map, output_file=None):
#     # Generate a molecule object from SMILES
#     mol = Chem.MolFromSmiles(smiles)
#     if mol is None:
#         raise ValueError(f"Invalid SMILES string: {smiles}")

#     # Create an empty drawing object
#     drawer = rdMolDraw2D.MolDraw2DSVG(300, 300)
#     opts = drawer.drawOptions()

#     # Prepare the color map with red at the low end, white for neutral, and green at the high end
#     cmap = mcolors.LinearSegmentedColormap.from_list("activity_cmap", ["red", "white", "green"])

#     # Normalize the activation map values between 0 and 1
#     norm = mcolors.Normalize(vmin=0, vmax=1)

#     # Convert activation map to colors
#     bond_colors = dict()
#     for bond in mol.GetBonds():
#         idx = bond.GetIdx()
#         # Assuming the activation map has one entry per bond, otherwise adjust as needed
#         value = activation_map[idx]
#         color = cmap(norm(value))
#         # Convert matplotlib color to RGBA tuple that RDKit expects
#         rgba_color = mcolors.to_rgba(color)
#         bond_colors[idx] = rgba_color

#     # Update the drawing options to use the bond colors
#     drawer.drawOptions().highlightBondColors = bond_colors

#     # Draw the molecule
#     drawer.DrawMolecule(mol)
#     drawer.FinishDrawing()

#     # Write to file or return SVG
#     svg = drawer.GetDrawingText().replace('svg:','')
#     if output_file:
#         with open(output_file, "w") as f:
#             f.write(svg)
#     else:
#         from IPython.display import SVG
#         return SVG(svg)

# # Call the function with the desired molecule index and output file
# # Example usage
# def visualize_from_csv(csv_file, molecule_index, output_file=None):
#     # Load the activation maps CSV file
#     activation_maps_df = pd.read_csv(csv_file)

#     # Select a molecule to visualize
#     smiles = activation_maps_df.iloc[molecule_index]['SMILES']
#     activation_map = activation_maps_df.iloc[molecule_index][3:].values.astype(float)  # Skip the first 3 columns (COMPOUND_ID, SMILES, PREDICTION)

#     # Visualize the molecule
#     return visualize_molecule(smiles, activation_map, output_file)

# # Call the function with the desired molecule index and output file
# svg_output = visualize_from_csv('activation_maps.csv', molecule_index=0, output_file='molecule_visualization.svg')

