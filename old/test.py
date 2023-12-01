import os
import re

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def rename_images(directory_path, prefix):
    # Get the absolute path to the directory
    directory_path = os.path.abspath('dataset\surya')

    # Get a list of all files in the directory
    files = os.listdir('dataset\surya')

    # Filter only files with certain extensions (e.g., '.jpg', '.jpeg', '.png')
    image_files = [file for file in files if file.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Sort the list of image files using natural sort
    image_files.sort(key=natural_sort_key)

    # Iterate over each image file and rename it
    for i, old_name in enumerate(image_files, start=1):
        # Construct the new file name
        new_name = f"{prefix}_{i}{os.path.splitext(old_name)[1]}"

        # Full paths for old and new names
        old_path = os.path.join('dataset\surya', old_name)
        new_path = os.path.join('dataset\surya', new_name)

        # Rename the file
        os.rename(old_path, new_path)
        print(f"Renamed: {old_path} to {new_path}")

# Specify the directory path and prefix
directory_path = r'dataset\lokesh'
prefix = 'surya'

# Call the function to rename images
rename_images(directory_path, prefix)
