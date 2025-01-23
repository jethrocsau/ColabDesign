# Import PyMOL module
import re

from pymol import cmd

# List of original object names
original_names = [
    "Hotspots-Runs6V0R_binder_50-55__seed1",
    "Hotspots-Runs6V0R_binder_100-105__seed1",
    "Hotspots-Runs6V0R_binder_100-105__seed2",
    "Hotspots-Runs6V0R_binder_100-105__seed3",
    "Hotspots-Runs6V0R_binder_150-155__seed1",
    "Hotspots-Runs6V0R_binder_150-155__seed2",
    "Hotspots-Runs6V0R_binder_150-155__seed3",
    "Hotspots-Runs6V0R_binder_200__seed1",
    "Hotspots-Runs6V0R_binder_200__seed2",
    "Hotspots-Runs6V0R_binder_200__seed3",
    "Hotspots-Runs6V0R_binder_None__seed1",
    "Hotspots-Runs6V0R_binder_None__seed2",
    "Hotspots-Runs6V0R_binder_None__seed3"
]

# Function to make a valid PyMOL object name
def make_valid_name(name):
    # Remove invalid characters, keeping alphanumeric and underscores
    return re.sub(r'\W|^(?=\d)', '_', name)

# Loop through the original names and rename them
for original_name in original_names:
    # Extract the binder and seed parts
    parts = original_name.split('__')
    binder_part = parts[0].split('_')[2]  # Extracts the binder part
    seed_part = parts[1]  # This is the seed part

    # Create the new name
    new_name = f"{binder_part}_{seed_part}"
    print("New_name: ", new_name)

    # Make a valid name
    valid_name = make_valid_name(new_name)

    # Rename the object in PyMOL
    cmd.set_name(original_name, valid_name)

    # Create a label at the position of Chain B
    cmd.label(f"{valid_name} and chain B and index 1", f'"{valid_name}"')

# Optional: Set label properties (font, size, etc.)
for original_name in original_names:
    binder_part = original_name.split('__')[0].split('_')[2]
    seed_part = original_name.split('__')[1]
    new_name = f"{binder_part}_{seed_part}"
    valid_name = make_valid_name(new_name)
    cmd.set("label_font_id", 7, f"{valid_name} and chain B and index 1")
    cmd.set("label_size", 20, f"{valid_name} and chain B and index 1")
