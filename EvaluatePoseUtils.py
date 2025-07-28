import numpy as np
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import os
from scipy.spatial import cKDTree

def invert_transform(T):
    """Invert a 4x4 homogeneous transformation matrix."""
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv

def get_correction_matrix(aruco_mat, fp_mat):
    T_offset = invert_transform(aruco_mat) @ fp_mat
    return T_offset

def rot_trans_from_mat(matrix):
    rot = matrix[:3, :3]
    trans = (matrix[:3, 3])
    return rot, trans

def calculate_add_error(points, R_est, t_est, R_gt, t_gt):
    transformed_est = (R_est @ points.T).T + t_est
    transformed_gt = (R_gt @ points.T).T + t_gt
    add_error = np.mean(np.linalg.norm(transformed_est - transformed_gt, axis=1))
    return add_error

# ADD-S
def calculate_adds_error(points, R_est, t_est, R_gt, t_gt):
    # Transform model points using estimated and ground truth poses
    transformed_est = (R_est @ points.T).T + t_est
    transformed_gt = (R_gt @ points.T).T + t_gt

    # Build KD-tree for fast nearest neighbor lookup
    gt_kdtree = cKDTree(transformed_gt)

    # Find closest GT point for each estimated point
    distances, _ = gt_kdtree.query(transformed_est, k=1)

    # Compute ADD-S as the mean of nearest neighbor distances
    adds_error = np.mean(distances)
    return adds_error

def mssd(R_est, t_est, R_gt, t_gt, pts, syms):
    """Maximum Symmetry-Aware Surface Distance (MSSD).

    See: http://bop.felk.cvut.cz/challenges/bop-challenge-2019/

    :param R_est: 3x3 ndarray with the estimated rotation matrix.
    :param t_est: 3x1 ndarray with the estimated translation vector.
    :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
    :param t_gt: 3x1 ndarray with the ground-truth translation vector.
    :param pts: nx3 ndarray with 3D model points.
    :param syms: Set of symmetry transformations, each given by a dictionary with:
      - 'R': 3x3 ndarray with the rotation matrix.
      - 't': 3x1 ndarray with the translation vector.
    :return: The calculated error.
    """
    pts_est = misc.transform_pts_Rt(pts, R_est, t_est)
    es = []
    for sym in syms:
        R_gt_sym = R_gt.dot(sym["R"])
        t_gt_sym = R_gt.dot(sym["t"]) + t_gt
        pts_gt_sym = misc.transform_pts_Rt(pts, R_gt_sym, t_gt_sym)
        es.append(np.linalg.norm(pts_est - pts_gt_sym, axis=1).max())
    return min(es)

def compute_accuracy_curve(add_errors, thresholds):
    accuracies = []
    for thresh in thresholds:
        count = np.sum(np.array(add_errors) <= thresh)
        accuracy = np.mean(np.array(add_errors) <= thresh)
        accuracies.append(accuracy)
        print(f"Threshold: {thresh}, Count: {count}, Accuracy: {accuracy}")
    #accuracies = [np.mean(np.array(add_errors) <= thresh) for thresh in thresholds]
    return accuracies

def compute_auc(accuracies, thresholds):
    return auc(thresholds, accuracies)

def compute_accuracy_curve_percent(add_errors, thresholds):
    """
    Compute accuracy curve (recall at different thresholds) given ADD errors and thresholds
    """
    accuracies = []
    for thresh in thresholds:
        # Calculate percentage of frames where ADD error is less than threshold (recall/accuracy)
        accuracy = np.mean(np.array(add_errors) <= thresh) * 100.0  # Convert to percentage
        accuracies.append(accuracy)
    return accuracies

def compute_auc_percent(accuracies, thresholds):
    """
    Compute Area Under the Curve using scikit-learn's auc function
    Returns a value between 0 and 100%
    """
    # Normalize by the threshold range to get area as a percentage of the total possible area
    auc_value = auc(thresholds, accuracies) / (thresholds[-1] - thresholds[0])
    return auc_value  # This is already a percentage (0-100) since accuracies are in percentage


## Errors by individual type

def calculate_rotation_error(points, R_est, R_gt, t_gt):
    # Apply the estimated and ground truth rotations to the points
    transformed_est = (R_est @ points.T).T + t_gt  # Use ground truth translation
    transformed_gt = (R_gt @ points.T).T + t_gt
    rot_error = np.mean(np.linalg.norm(transformed_est - transformed_gt, axis=1))
    return rot_error

def calculate_translation_error(points, t_est, t_gt, R_gt):
    # Apply the ground truth rotation but different translations
    transformed_est = (R_gt @ points.T).T + t_est
    transformed_gt = (R_gt @ points.T).T + t_gt
    trans_error = np.mean(np.linalg.norm(transformed_est - transformed_gt, axis=1))
    return trans_error

def calculate_translation_error_by_axis(points, t_est, t_gt, R_gt):
    # Apply the ground truth rotation
    transformed_est = (R_gt @ points.T).T + t_est
    transformed_gt = (R_gt @ points.T).T + t_gt
    
    # Compute the elementwise differences for each axis
    diff_x = np.mean(np.abs(transformed_est[:, 0] - transformed_gt[:, 0]))
    diff_y = np.mean(np.abs(transformed_est[:, 1] - transformed_gt[:, 1]))
    diff_z = np.mean(np.abs(transformed_est[:, 2] - transformed_gt[:, 2]))
    
    errors_by_axis = {
        'x': diff_x,
        'y': diff_y,
        'z': diff_z
    }
    
    return errors_by_axis

def compute_accuracy_curves(trans_err_lists, thresholds):
    # Accuracies by axis
    accuracies_by_axis = {'x': [], 'y': [], 'z': []}

    # Convert threshold values to numpy array for vectorized operations
    thresholds = np.array(thresholds)

    for axis in ['x', 'y', 'z']:
        accuracies = []
        for thresh in thresholds:
            count = np.sum([err[axis] <= thresh for err in trans_err_lists])
            accuracy = count / len(trans_err_lists)
            accuracies.append(accuracy)
        accuracies_by_axis[axis] = accuracies
    
    return accuracies_by_axis

def plot_accuracy_over_time(rot_err_list, trans_err_list, add_list, threshold):
    #x_values = [d['z'] for d in trans_err_list]
    #plt.hist(add_list, bins=50, alpha=0.75)
    #plt.axvline(x=threshold_10_percent, color='r', linestyle='--', label='10% Diameter Threshold')

    x_values = [x for x in range(len(rot_err_list))]
    #plt.plot(x_values, trans_err_list, label = 'Translation Error')
    tx_values = [d['x'] for d in trans_err_list]
    ty_values = [d['y'] for d in trans_err_list]
    tz_values = [d['z'] for d in trans_err_list]
    plt.plot(x_values, tx_values, label="Just x")
    plt.plot(x_values, ty_values, label="Just y")
    plt.plot(x_values, tz_values, label="Just z")

    plt.plot(x_values, rot_err_list, label='Rotation Error')

    plt.plot(x_values, add_list, label="Combined Error")
    plt.axhline(y=threshold, color='r', linestyle='--')

    plt.xlabel('Frame')
    plt.ylabel('ADD Error')
    plt.legend()
    plt.title('ADD Error over time')
    #plt.title('AUC of accuracies = {}'.format(auc_value))
    plt.show()

def plot_add_by_threshold(add_list):
    plt.hist(add_list, bins=50, alpha=0.75)
    #plt.axvline(x=threshold_10_percent, color='r', linestyle='--', label='10% Diameter Threshold')
    plt.xlabel('ADD Error')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('ADD Error Distribution')
    plt.show()

def calculate_ADD_pointone_percent(diameter, add_list, output_dir):
    # Define thresholds for evaluation (from 0 to 10% of object diameter)
    thresholds = np.linspace(0, 0.1 * diameter, 100)

    # Calculate accuracy curve and AUC
    accuracies = compute_accuracy_curve(add_list, thresholds)
    auc_value = compute_auc(accuracies, thresholds)

    # Calculate accuracy at specific thresholds
    threshold_5_percent = 0.05 * diameter
    threshold_10_percent = 0.1 * diameter
    accuracy_5_percent = np.mean(np.array(add_list) <= threshold_5_percent)
    accuracy_10_percent = np.mean(np.array(add_list) <= threshold_10_percent)

    # Print evaluation results
    print(f"\nEvaluation Results:")
    print(f"Number of frames: {len(add_list)}")
    print(f"Mean ADD error: {np.mean(add_list):.6f}")
    print(f"Median ADD error: {np.median(add_list):.6f}")
    print(f"Min ADD error: {np.min(add_list):.6f}")
    print(f"Max ADD error: {np.max(add_list):.6f}")
    print(f"5% diameter threshold ({threshold_5_percent:.6f}): {accuracy_5_percent*100:.2f}% accuracy")
    print(f"10% diameter threshold ({threshold_10_percent:.6f}): {accuracy_10_percent*100:.2f}% accuracy")
    print(f"AUC value: {auc_value:.6f}")

    # Visualize results
    # 1. Plot accuracy curve
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, accuracies)
    plt.xlabel('ADD Threshold')
    plt.ylabel('Accuracy')
    plt.title(f'ADD Accuracy Curve (AUC: {auc_value:.2f}%, 10% diameter accuracy: {accuracy_10_percent:.2f}%)')
    plt.axvline(x=threshold_10_percent, color='r', linestyle='--', label='10% Diameter')
    plt.grid(True)
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'add_accuracy_curve.png'))
    plt.show()

    # 2. Plot ADD error distribution
    plt.figure(figsize=(10, 6))
    plt.hist(add_list, bins=30, alpha=0.75)
    plt.axvline(x=threshold_5_percent, color='g', linestyle='--', label='5% Diameter')
    plt.axvline(x=threshold_10_percent, color='r', linestyle='--', label='10% Diameter')
    plt.xlabel('ADD Error')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('ADD Error Distribution')
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'add_error_distribution.png'))
    plt.show()

    # 3. Plot ADD error over time
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(add_list)), add_list)
    plt.axhline(y=threshold_5_percent, color='g', linestyle='--', label='5% Diameter')
    plt.axhline(y=threshold_10_percent, color='r', linestyle='--', label='10% Diameter')
    plt.xlabel('Frame')
    plt.ylabel('ADD Error')
    plt.legend()
    plt.title('ADD Error Over Time')
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'add_error_over_time.png'))
    plt.show()
        