import os

def rename_files(directory_path, base_name):
    # Get the list of files in the specified directory
    files = os.listdir(directory_path)

    # Iterate over each file and rename it
    for i, file_name in enumerate(files):
        # Build the new file name with the suffix
        new_name = f"{base_name}_{i+1}{os.path.splitext(file_name)[1]}"

        # Construct the full paths for the old and new names
        old_path = os.path.join(directory_path, file_name)
        new_path = os.path.join(directory_path, new_name)

        # Rename the file
        os.rename(old_path, new_path)

        print(f"Renamed: {file_name} -> {new_name}")

# Specify the directory path where files need to be renamed
directory_path = "dataset\jahnavi"

# Specify the base name for the files
user_given_name = "jahnavi"

# Call the function to rename files
rename_files(directory_path, user_given_name)
