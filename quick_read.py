import numpy as np

# Replace 'file.npy' with the path to your .npy file
# file_path = './dataset/hhar/data_20_120.npy'
# file_path = './dataset/hhar/label_20_120.npy'
# file_path = './dataset/motion/data_20_120.npy'
# file_path = './dataset/motion/label_20_120.npy'
# file_path = './dataset/uci/data_20_120.npy'
# file_path = './dataset/uci/label_20_120.npy'
# file_path = './dataset/shoaib/data_20_120.npy'
file_path = './dataset/shoaib/label_20_120.npy'
try:
    # Load the .npy file
    data = np.load(file_path)
    
    # Check if the loaded data is a NumPy array
    if isinstance(data, np.ndarray):
        print(f"The dimensions of the loaded array are: {data.shape}")
        print(data[5405,:,:])    
    else:
        print("The loaded file does not contain a NumPy array.")
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
