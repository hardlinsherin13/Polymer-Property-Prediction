import pandas as pd

# Load the dataset
dataset = pd.read_csv("/content/drive/My Drive/polymer_dataset_with_properties_and_smiles.csv")
# Define a function to get SMILES notation and melting temperature for a given polymer name
def get_smiles_and_temperature(polymer_name):
    # Search for the polymer name in the dataset
    polymer_info = dataset[dataset["PolymerName"] == polymer_name]
    if polymer_info.empty:
        return "Polymer name not found", None
    else:
        # Extract SMILES notation and melting temperature
        smiles = polymer_info["SMILES"].iloc[0]
        melting_temp = polymer_info["Melting Temperature (Tm)"].iloc[0]
        return smiles, melting_temp
    # Example usage:
polymer_name = input("Enter polymer name: ")
smiles, melting_temp = get_smiles_and_temperature(polymer_name)

if smiles is not None:
    print("SMILES Notation:", smiles)
    print("Melting Temperature (Tm):", melting_temp, "°C")
else:
    print("Polymer name not found in the dataset.")