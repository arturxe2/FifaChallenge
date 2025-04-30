## Open and print boxes.npz file
import numpy as np
import os


# Load the boxes.npz file
boxes_path = os.path.join('data', 'boxes.npz')
boxes = np.load(boxes_path, allow_pickle=True)

#npz to dictionary
boxes_dict = {key: boxes[key] for key in boxes.files}
# Print the keys and data of the boxes
print("Boxes keys:", boxes_dict.keys())
print("Boxes data:")
for key in boxes_dict.keys():
    print(f"{key}: {boxes_dict[key]}")

