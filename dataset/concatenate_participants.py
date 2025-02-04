import os
import numpy as np
import pandas as pd

def process_participant_files(input_folder, output_folder, version, seq_len=125):
    """
    Reads participant files from input_folder, processes them, and saves 
    accumulated data and labels into .npy files in output_folder.
    
    Args:
        input_folder (str): Path to the folder containing participant files.
        output_folder (str): Path to the folder where output .npy files will be saved.
        version (str): Version string for output files.
        seq_len (int): Sequence length for segmenting the data.
    
    Returns:
        None
    """
    data = []
    labels = []

    # Iterate through files in the input folder
    for file_name in os.listdir(input_folder):
        print(file_name)
        if "catch" in file_name:
            print("skipping....")
            continue

        file_path = os.path.join(input_folder, file_name)

        if not os.path.isfile(file_path):
            continue  # Skip directories or invalid files

        # Load participant file
        try:
            participant_data = np.load(file_path, allow_pickle=True)
            windows, activity_values, user_values = participant_data
            # print(type(windows[0]))
        except Exception as e:
            print(f"Error loading {file_name}: {e}")
            continue

        # Process each window (dataframe) and corresponding labels
        for window, activity, user in zip(windows, activity_values, user_values):
            # Convert window (dataframe) to numpy array
            window_data = window.to_numpy(dtype=np.float32)

            # Ensure window_data length is a multiple of seq_len
            usable_length = (window_data.shape[0] // seq_len) * seq_len
            if usable_length == 0:
                continue  # Skip windows too short for even one sequence
            window_data = window_data[:usable_length, :]

            # Reshape into sequences of (seq_len, features)
            reshaped_data = window_data.reshape(-1, seq_len, window_data.shape[1])

            # Create corresponding labels
            activity_label = np.full((reshaped_data.shape[0], seq_len, 1), activity, dtype=np.int32)
            user_label = np.full((reshaped_data.shape[0], seq_len, 1), user, dtype=np.int32)
            combined_label = np.concatenate((activity_label, user_label), axis=-1)  # Shape: (n_segments, seq_len, 2)

            # Append processed data and labels
            data.append(reshaped_data)
            labels.append(combined_label)

    # Concatenate all data and labels
    if data:
        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0)

        # Save to .npy files
        os.makedirs(output_folder, exist_ok=True)
        np.save(os.path.join(output_folder, f"data_{version}.npy"), data)
        np.save(os.path.join(output_folder, f"label_{version}.npy"), labels)

        print(f"Data and labels saved. Data shape: {data.shape}, Label shape: {labels.shape}")
    else:
        print("No data processed. Check input folder and file formats.")


process_participant_files("/mnt/EA3E453B3E450255/ETH_RSC/3/MT/RealisticHAR/source/data/C24/processed", "./", "25_125", seq_len=125)
