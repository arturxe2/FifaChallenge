"""
This script converts npz to parquet submission file.

Author: Tianjian Jiang
Date: March 16, 2025
"""
import numpy as np
import pandas as pd

def npz_to_submission(input_file, submission_path, drop_na=False, add_row_id=False):
    """
    Convert an NPZ file (dict with key: sequence name and value: 4D array)
    to a submission file with columns: sequence, frame, person, joint, x, y, z.
    """
    # Load the NPZ file (allow_pickle=True is needed if it was saved as a dict)
    data = np.load(input_file, allow_pickle=True)
    
    rows = []
    # data.files gives you all the keys (sequence names)
    for seq_name in data.files:
        arr = data[seq_name]  # shape: (NUM_FRAMES, NUM_PERSONS, NUM_JOINTS, 3)
        num_frames, num_persons, num_joints, _ = arr.shape
        # Iterate through each index to extract coordinates
        for frame in range(num_frames):
            for person in range(num_persons):
                for joint in range(num_joints):
                    x, y, z = arr[frame, person, joint, :]
                    rows.append([seq_name, frame, person, joint, x, y, z])
    
    # Create a DataFrame and specify column names
    df = pd.DataFrame(rows, columns=["sequence", "frame", "person", "joint", "x", "y", "z"])

    if drop_na:
        df = df.dropna(subset=["x", "y", "z"])
    
    if add_row_id:
        row_id = df["sequence"] + "_" + df["frame"].astype(str) + "_" + df["person"].astype(str) + "_" + df["joint"].astype(str)
        df.insert(0, "row_id", row_id)
    
    df["x"] = df["x"].round(3)
    df["y"] = df["y"].round(3)
    df["z"] = df["z"].round(3)

    # Save to CSV without the index
    # df.to_csv(csv_path, index=False)
    # Save to parquet
    df.to_parquet(submission_path, index=False, compression="snappy")
    print(f"Saved to {submission_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", help="Path to the input NPZ file", default="input_data.npz")
    parser.add_argument("--output", "-o", help="Path to the submission file", default="submission.parquet")
    args = parser.parse_args()

    args.drop_na = True
    args.add_row_id = True
    npz_to_submission(args.input, args.output, drop_na=args.drop_na, add_row_id=args.add_row_id)
