import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

def umeyama_alignment(source, target):
    """
    Computes the similarity transform (scale, rotation, translation) that best aligns
    the source points to the target points (both arrays of shape (N, 3)).
    
    Returns:
      s (float): scale factor
      R (3x3 numpy array): rotation matrix
      t (3-element numpy array): translation vector
    """
    assert source.shape == target.shape, "Source and target must have same shape"
    n, dim = source.shape

    # Compute means of source and target.
    mu_source = np.mean(source, axis=0)
    mu_target = np.mean(target, axis=0)

    # Subtract mean.
    source_centered = source - mu_source
    target_centered = target - mu_target

    # Compute covariance matrix.
    H = source_centered.T @ target_centered / n

    # SVD on the covariance matrix.
    U, D, Vt = np.linalg.svd(H)
    R = U @ Vt

    # Ensure a proper rotation (determinant +1).
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt

    # Compute scale.
    var_source = np.sum(source_centered ** 2) / n
    s = np.sum(D) / var_source

    # Compute translation.
    t = mu_target - s * (R @ mu_source)

    return s, R, t

def align_predictions(pred_poses, gt_poses):
    """
    Given predicted and ground truth poses (each of shape (N, 6), where columns 3:6 are x,y,z),
    computes a similarity transform using Umeyama alignment on the translation components,
    applies it to all predicted poses, and returns the aligned predictions.
    """
    pred_xyz = pred_poses[:, 3:6]
    gt_xyz = gt_poses[:, 3:6]
    
    s, R, t = umeyama_alignment(pred_xyz, gt_xyz)
    pred_xyz_aligned = (s * (R @ pred_xyz.T)).T + t
    
    pred_poses_aligned = np.copy(pred_poses)
    pred_poses_aligned[:, 3:6] = pred_xyz_aligned
    return pred_poses_aligned, s, R, t

def main():
    # File paths – adjust as needed.
    pred_file = "/home/krkavinda/ProjectX-midAir/result/out_Kite_training_sunny_trajectory_0008.txt"
    gt_file = "/media/krkavinda/New Volume/Mid-Air-Dataset/MidAir_processed/Kite_training/sunny/poses/poses_0008.npy"

    # Check file existence.
    if not os.path.exists(pred_file):
        print(f"Predicted file not found: {pred_file}")
        return
    if not os.path.exists(gt_file):
        print(f"Ground truth file not found: {gt_file}")
        return

    # Load predicted poses from text file.
    pred_list = []
    with open(pred_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                # Each line is expected to have six comma-separated numbers.
                pred_list.append([float(x.strip()) for x in line.split(',')])
    pred_poses = np.array(pred_list)  # shape: (N_pred, 6)
    
    # Load ground truth poses.
    gt_poses = np.load(gt_file)       # shape: (N_gt, 6)

    print("Predicted data shape:", pred_poses.shape)
    print("Ground truth data shape:", gt_poses.shape)

    # To compare, we need to use a common number of poses.
    # In your output, ground truth has ~2205 poses while predictions have ~2204.
    n = min(pred_poses.shape[0], gt_poses.shape[0])
    pred_poses = pred_poses[:n]
    gt_poses = gt_poses[:n]

    # Print some initial values for sanity check.
    print("First 5 predicted poses:")
    print(pred_poses[:5])
    print("First 5 ground truth poses:")
    print(gt_poses[:5])

    # Display data ranges for the translation part.
    print("\nData ranges:")
    print(f"Ground truth X: {gt_poses[:,3].min():.2f} to {gt_poses[:,3].max():.2f}")
    print(f"Ground truth Y: {gt_poses[:,4].min():.2f} to {gt_poses[:,4].max():.2f}")
    print(f"Ground truth Z: {gt_poses[:,5].min():.2f} to {gt_poses[:,5].max():.2f}")
    print(f"Predicted X: {pred_poses[:,3].min():.2f} to {pred_poses[:,3].max():.2f}")
    print(f"Predicted Y: {pred_poses[:,4].min():.2f} to {pred_poses[:,4].max():.2f}")
    print(f"Predicted Z: {pred_poses[:,5].min():.2f} to {pred_poses[:,5].max():.2f}")

    # Align the predicted poses to the ground truth using Umeyama alignment.
    pred_aligned, s, R, t = align_predictions(pred_poses, gt_poses)
    print("\nAlignment parameters:")
    print("Scale =", s)
    print("Rotation matrix:\n", R)
    print("Translation vector:", t)

    # Compute the mean absolute trajectory error (ATE) after alignment.
    errors = np.linalg.norm(pred_aligned[:,3:6] - gt_poses[:,3:6], axis=1)
    ate_error = np.mean(errors)
    print("\nMean ATE Error after alignment: {:.4f} meters".format(ate_error))

    # Plot the trajectories.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(gt_poses[:,3], gt_poses[:,4], gt_poses[:,5],
            label="Ground Truth", color='green')
    ax.plot(pred_aligned[:,3], pred_aligned[:,4], pred_aligned[:,5],
            label="Aligned Prediction", color='red')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.title("Trajectory Comparison after Umeyama Alignment")

    # Save the plot.
    out_plot = "/home/krkavinda/ProjectX-midAir/comparison_plots/aligned_trajectory.png"
    os.makedirs(os.path.dirname(out_plot), exist_ok=True)
    plt.savefig(out_plot)
    print("\nSaved aligned trajectory plot to", out_plot)
    plt.show()

if __name__ == "__main__":
    main()
