"""
This script provides a naive baseline for FIFA Skeletal Tracking Challenge.

Author: Tianjian Jiang
Date: March 16, 2025
"""
from pathlib import Path
import numpy as np
import cv2
import scipy.optimize
import torch
import torch.optim as optim
from tqdm import trange


OPENPOSE_TO_OURS = [0, 2, 5, 3, 6, 4, 7, 9, 12, 10, 13, 11, 14, 22, 19]


def intersection_over_plane(o, d):
    """
    args:
        o: (3,) origin of the ray
        d: (3,) direction of the ray

    returns:
        intersection: (3,) intersection point
    """
    # solve the x and y where z = 0
    t = -o[2] / d[2]
    return o + t * d


def ray_from_xy(xy, K, R, t, k1=0.0, k2=0.0):
    """
    Compute the ray from the camera center through the image point (x, y),
    correcting for radial distortion using coefficients k1 and k2.

    Args:
        xy: (2,) array_like containing pixel coordinates [x, y] in the image.
        K: (3, 3) ndarray representing the camera intrinsic matrix.
        R: (3, 3) ndarray representing the camera rotation matrix.
        t: (3,) ndarray representing the camera translation vector.
        k1: float, the first radial distortion coefficient (default 0).
        k2: float, the second radial distortion coefficient (default 0).

    Returns:
        origin: (3,) ndarray representing the camera center in world coordinates.
        direction: (3,) unit ndarray representing the direction of the ray in world coordinates.
    """
    # Convert the pixel coordinate to homogeneous coordinates.
    p = np.array([xy[0], xy[1], 1.0])

    # Compute the normalized coordinate (distorted) in the camera coordinate system.
    p_norm = np.linalg.inv(K) @ p  # p_norm = [x_d, y_d, 1]
    x_d, y_d = p_norm[0], p_norm[1]

    # Compute the radial distance (squared) in the normalized plane.
    r2 = x_d**2 + y_d**2
    # Compute the distortion factor.
    factor = 1 + k1 * r2 + k2 * (r2**2)

    # Correct the distorted normalized coordinates.
    x_undist = x_d / factor
    y_undist = y_d / factor

    # Construct the undistorted direction in camera coordinates (z = 1).
    d_cam = np.array([x_undist, y_undist, 1.0])

    # Transform the direction to world coordinates.
    direction = R.T @ d_cam
    direction = direction / np.linalg.norm(direction)

    # The camera center in world coordinates is given by -R^T t.
    origin = -R.T @ t
    return origin, direction


def estimate_rt(K1, K2, R1, t1, boxes1, boxes2, dist_coeffs1=None, dist_coeffs2=None):
    """
    Using SVD directly to estimate the rotation.

    args:
        K1: (3, 3) - Camera intrinsics matrix for the first camera
        K2: (3, 3) - Camera intrinsics matrix for the second camera
        R1: (3, 3) - Rotation matrix of the first camera
        t1: (3,) - Translation vector of the first camera
        boxes1: (N, 4) - Bounding boxes of the first camera
        boxes2: (N, 4) - Bounding boxes of the second camera
        dist_coeffs1: (5,) - Distortion coefficients for the first camera
        dist_coeffs2: (5,) - Distortion coefficients for the second camera

    returns:
        R2: (3, 3) - Estimated rotation matrix of the second camera
        t2: (3,) - Estimated translation vector of the second camera
    """
    # Compute box centers
    valid_indices = []
    for i in range(len(boxes1)):
        if not np.any(np.isnan(boxes1[i])) and not np.any(np.isnan(boxes2[i])):
            valid_indices.append(i)

    if len(valid_indices) < 3:
        raise ValueError("Need at least 3 valid correspondences to estimate rotation reliably")

    # Extract valid boxes
    valid_boxes1 = boxes1[valid_indices]
    valid_boxes2 = boxes2[valid_indices]

    # we calculate the aspect ratio of the boxes
    ar1 = valid_boxes1[:, 2:] - valid_boxes1[:, :2]
    ar1 = ar1[:, 0] / ar1[:, 1]
    ar2 = valid_boxes2[:, 2:] - valid_boxes2[:, :2]
    ar2 = ar2[:, 0] / ar2[:, 1]
    aspect_ratios = ar1 / ar2

    # sort by aspect ratio
    valid_indices = np.argsort(np.abs(aspect_ratios - 1))[: max(len(valid_indices) // 4, 5)]
    valid_boxes1 = valid_boxes1[valid_indices]
    valid_boxes2 = valid_boxes2[valid_indices]

    # we pick all the 4 corners of boxes
    centers1_pixel = np.array(
        [
            [
                [box[0], box[1]],
                [box[0], box[3]],
                [box[2], box[1]],
                [box[2], box[3]],
            ]
            for box in valid_boxes1
        ]
    ).reshape(-1, 2)

    centers2_pixel = np.array(
        [
            [
                [box[0], box[1]],
                [box[0], box[3]],
                [box[2], box[1]],
                [box[2], box[3]],
            ]
            for box in valid_boxes2
        ]
    ).reshape(-1, 2)

    # Apply distortion correction if coefficients are provided
    if dist_coeffs1 is not None:
        # Ensure dist_coeffs1 is in the right format
        if len(dist_coeffs1) == 2:  # If only k1, k2 provided
            dist_coeffs1 = np.array([dist_coeffs1[0], dist_coeffs1[1], 0, 0, 0])
        centers1_pixel = cv2.undistortPoints(centers1_pixel.reshape(-1, 1, 2), K1, dist_coeffs1, P=K1).reshape(-1, 2)

    if dist_coeffs2 is not None:
        # Ensure dist_coeffs2 is in the right format
        if len(dist_coeffs2) == 2:  # If only k1, k2 provided
            dist_coeffs2 = np.array([dist_coeffs2[0], dist_coeffs2[1], 0, 0, 0])
        centers2_pixel = cv2.undistortPoints(centers2_pixel.reshape(-1, 1, 2), K2, dist_coeffs2, P=K2).reshape(-1, 2)

    # Convert to homogeneous coordinates
    centers1 = np.column_stack((centers1_pixel, np.ones(centers1_pixel.shape[0])))
    centers2 = np.column_stack((centers2_pixel, np.ones(centers2_pixel.shape[0])))

    # Convert centers to normalized camera coordinates
    normalized_centers1 = np.array([np.linalg.inv(K1) @ center for center in centers1])
    normalized_centers2 = np.array([np.linalg.inv(K2) @ center for center in centers2])

    # Normalize vectors
    normalized_centers1 = normalized_centers1 / np.linalg.norm(normalized_centers1, axis=1, keepdims=True)
    normalized_centers2 = normalized_centers2 / np.linalg.norm(normalized_centers2, axis=1, keepdims=True)

    # Compute correlation matrix
    H = normalized_centers2.T @ normalized_centers1

    # SVD of correlation matrix
    U, _, Vt = np.linalg.svd(H)

    # Compute rotation matrix
    R_rel = U @ Vt

    # Ensure proper rotation matrix (det=1)
    if np.linalg.det(R_rel) < 0:
        # Flip the last column of U
        U[:, 2] = -U[:, 2]
        R_rel = U @ Vt

    # Calculate t2 using the constraint that the camera center doesn't change
    R2 = R_rel @ R1
    t2 = R2 @ np.linalg.inv(R1) @ t1
    return R2, t2


def find_closest_orthogonal_matrix(A: np.ndarray) -> np.ndarray:
    """find closest orthogonal matrix in terms of matrix norm"""
    u, _, vh = np.linalg.svd(A)
    return u @ vh


def extract_lane_lines_mask(image):
    """extract lane lines from the image using adaptive thresholding and masking
    args:
        image: (H, W, 3) - BGR image
    returns:
        mask: (H, W) - binary mask of the lane lines
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    lightness = image[:, :, 1]
    threshold = cv2.adaptiveThreshold(lightness, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, -10)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    black = gray < 10
    threshold[black] = 0
    mask = threshold
    return mask


def make_dist_map(mask):
    """
    Create a distance map from a binary mask

    args:
        mask: (H, W) - binary mask

    returns:
        dist: (H, W) - distance map
    """
    mask_inv = (1 - (mask > 0)).astype(np.uint8)
    dist = cv2.distanceTransform(mask_inv, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return dist


def refine_camera_rotation(mask, pts_3d, K, R_init, C, dist_coeffs=None):
    """
    Refine camera rotation matrix R to align projected 3D points with a mask
    using direct optimization of the rotation matrix with orthogonality enforcement

    args:
        mask: (H, W) - binary mask of the lane lines
        pts_3d: (N, 3) - 3D points in world space
        K: (3, 3) - Camera intrinsic matrix
        R_init: (3, 3) - Initial rotation matrix
        C: (3,) - Camera center
        dist_coeffs: (5,) - Distortion coefficients

    returns:
        R_refined: (3, 3) - Refined rotation matrix
    """
    # Ensure all inputs are numpy arrays
    if dist_coeffs is None:
        dist_coeffs = np.zeros(5, dtype=np.float32)

    # Create distance map from mask
    dist_map = make_dist_map(mask)
    H, W = mask.shape[:2]

    # Define objective function for optimization
    def objective_function(r_flat):
        R_raw = r_flat.reshape(3, 3)
        R = find_closest_orthogonal_matrix(R_raw)
        t = -R @ C

        pts_2d, _ = cv2.projectPoints(pts_3d, cv2.Rodrigues(R)[0], t, K, dist_coeffs)
        pts_2d = pts_2d.squeeze(axis=1)
        pts_2d = pts_2d.clip([-500, -500], [W + 500, H + 500])

        xs = np.round(pts_2d[:, 0]).astype(np.int32)
        ys = np.round(pts_2d[:, 1]).astype(np.int32)

        valid_mask = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
        xs_valid = xs[valid_mask]
        ys_valid = ys[valid_mask]

        if len(xs_valid) == 0:
            distances = np.sqrt(H**2 + W**2)  # Return large value if no valid points
        else:
            distances = dist_map[ys_valid, xs_valid]
            distances = distances.mean()
        return distances

    r_flat = R_init.flatten()
    epsilon = 0.1  # Adjust epsilon as needed
    lower_bounds = r_flat - epsilon
    upper_bounds = r_flat + epsilon
    result = scipy.optimize.least_squares(
        objective_function,
        r_flat,
        bounds=(lower_bounds, upper_bounds),
    )

    # Reshape result to 3x3 matrix and enforce orthogonality
    R_refined_raw = result.x.reshape(3, 3)
    R_refined = find_closest_orthogonal_matrix(R_refined_raw)
    return R_refined


def project_points_th(obj_pts, R, C, K, k):
    """Projects 3D points onto 2D image plane using camera intrinsics and distortion.

    args:
        obj_pts: (N, 3) - 3D points in world space
        R: (3, 3) - Rotation matrix
        C: (3,) - Camera center
        K: (3, 3) - Camera intrinsic matrix
        k: (5,) - Distortion coefficients

    returns:
        img_pts: (N, 2) - Projected 2D points
    """

    # Transform world points to camera coordinates
    pts_c = (R @ ((obj_pts - C).unsqueeze(-1))).squeeze(-1)

    # Normalize to get image plane coordinates
    img_pts = pts_c[:, :2] / pts_c[:, 2:]

    # Compute radial distortion
    r2 = (img_pts**2).sum(dim=-1, keepdim=True)
    r2 = torch.clamp(r2, 0, 0.5 / min(max(torch.abs(k).max().item(), 1.0), 1.0))
    p = torch.arange(1, k.shape[-1] + 1, device=k.device)
    img_pts = img_pts * (torch.ones_like(r2) + (k * r2.pow(p)).sum(-1, keepdim=True))

    # Apply intrinsics K
    img_pts_h = torch.cat([img_pts, torch.ones_like(img_pts[:, :1])], dim=-1)  # Homogeneous coords
    img_pts = (K @ img_pts_h.unsqueeze(-1)).squeeze(-1)[:, :2]  # Convert back to 2D

    return img_pts


def minimize_reprojection_error(pts_3d, pts_2d, R, C, K, k, iterations=10):
    """
    Optimize 3D points to minimize reprojection error.

    args:
        pts_3d: (N, 3)  - Initial 3D points (learnable)
        pts_2d: (N, 2)  - Corresponding 2D points
        R: (3, 3)       - Rotation matrix (fixed)
        C: (3,)         - Camera center (fixed)
        K: (3, 3)       - Camera intrinsic matrix (fixed)
        k: (5,)         - Distortion coefficients (fixed)
        lr: float       - Learning rate for Adam optimizer
        iterations: int - Number of optimization steps

    returns:
        t: (N, 3) - Optimized translation
    """
    # Convert 3D points to learnable parameters
    # pts_3d = torch.nn.Parameter(pts_3d.clone().detach().requires_grad_(True))
    t = torch.nn.Parameter(torch.zeros_like(pts_3d).clone().detach().requires_grad_(True))
    offset = torch.tensor([1, 1, 0.2], dtype=pts_3d.dtype, device=pts_3d.device)
    lower_bounds = t - offset
    upper_bounds = t + offset

    # check if there are any NaN values
    assert not torch.isnan(pts_3d).any()
    assert not torch.isnan(pts_2d).any()

    def closure():
        optimizer.zero_grad()
        projected_pts = project_points_th(pts_3d + t, R, C, K, k)
        loss = torch.nn.functional.mse_loss(projected_pts, pts_2d)
        loss.backward()
        return loss

    optimizer = optim.LBFGS([t], line_search_fn="strong_wolfe")
    for _ in range(iterations):
        optimizer.step(closure)
        with torch.no_grad():
            t.copy_(torch.clamp(t, lower_bounds, upper_bounds))

    return t.detach()


def fine_tune_translation(predictions, skels_2d, cameras, Rt, boxes):
    """wrapper function to fine-tune the translation of the 3D predictions to minimize reprojection error"""
    NUM_PERSONS = predictions.shape[0]
    mid_hip_3d = predictions[:, :, [7, 8]].mean(axis=-2, keepdims=False)
    mid_hip_2d = skels_2d[:, :, [7, 8]].mean(axis=-2, keepdims=False).transpose(1, 0, 2)
    camera_params = {
        "K": cameras["K"][None].repeat(NUM_PERSONS, axis=0),
        "R": np.array([k[0] for k in Rt])[None].repeat(NUM_PERSONS, axis=0),
        "t": np.array([k[1] for k in Rt])[None].repeat(NUM_PERSONS, axis=0),
        "k": cameras["k"][None, ..., :2].repeat(NUM_PERSONS, axis=0),
    }
    camera_params["C"] = -(camera_params["t"][..., None, :] @ camera_params["R"]).squeeze(axis=-2)
    valid = ~np.isnan(boxes).any(axis=-1).transpose(1, 0)
    traj_3d = minimize_reprojection_error(
        pts_3d=torch.tensor(mid_hip_3d[valid], dtype=torch.float32).to("cuda"),
        pts_2d=torch.tensor(mid_hip_2d[valid], dtype=torch.float32).to("cuda"),
        R=torch.tensor(camera_params["R"][valid], dtype=torch.float32).to("cuda"),
        C=torch.tensor(camera_params["C"][valid], dtype=torch.float32).to("cuda"),
        K=torch.tensor(camera_params["K"][valid], dtype=torch.float32).to("cuda"),
        k=torch.tensor(camera_params["k"][valid], dtype=torch.float32).to("cuda"),
    )
    return traj_3d, valid


def baseline(boxes, cameras, skels_3d, skels_2d, image_dir):
    """a naive baseline that uses the bounding boxes to estimate the camera pose
    1. estimate the camera pose using the bounding boxes
    2. periodically refine the camera pose using lane lines
    3. project the 3D skeletons to the 2D image plane and optimize the translation to minimize reprojection error
    """
    NUM_FRAMES, NUM_PERSONS, _ = boxes.shape
    predictions = np.zeros((NUM_PERSONS, NUM_FRAMES, 15, 3))
    predictions.fill(np.nan)
    pitch_points = np.loadtxt("data/pitch_points.txt")

    Rt = []
    STEPS = max(NUM_FRAMES // 500, 1)
    for frame in trange(NUM_FRAMES, desc=f"{image_dir.stem}"):
        if frame == 0:
            R, t = cameras["R"][0], cameras["t"][0]
        else:
            R, t = estimate_rt(
                K1=cameras["K"][frame - 1],
                K2=cameras["K"][frame],
                R1=Rt[-1][0],
                t1=Rt[-1][1],
                boxes1=boxes[frame - 1],
                boxes2=boxes[frame],
                dist_coeffs1=cameras["k"][frame - 1],
                dist_coeffs2=cameras["k"][frame],
            )
            if frame % STEPS == 0:
                image_path = image_dir / f"{frame:05d}.jpg"
                mask = extract_lane_lines_mask(cv2.imread(str(image_path)))
                R_refined = refine_camera_rotation(
                    mask=mask,
                    pts_3d=pitch_points,
                    K=cameras["K"][frame],
                    R_init=R,
                    C=-R.T @ t,
                    dist_coeffs=cameras["k"][frame],
                )
                t = R_refined @ R.T @ t
                R = R_refined
        Rt.append((R, t))

        for person in range(NUM_PERSONS):
            box = boxes[frame, person]
            if np.isnan(box).any():
                continue
            _, br = box[:2], box[2:]
            x, y = br
            K = cameras["K"][frame]
            k = cameras["k"][frame]
            R, t = Rt[-1]
            o, d = ray_from_xy((x, y), K, R, t, k[0], k[1])
            intersection = intersection_over_plane(o, d)

            skel_3d = skels_3d[frame, person]
            # convert from camera space to world space
            skel_3d = skel_3d @ R
            skel_3d = skel_3d - skel_3d[[-1]] + intersection
            predictions[person, frame] = skel_3d

    # fine-tune the translation to minimize reprojection error
    traj_3d, valid = fine_tune_translation(predictions, skels_2d, cameras, Rt, boxes)
    predictions[valid] = predictions[valid] + traj_3d.cpu().numpy()[:, None, :]
    return predictions


def main(root):
    solutions = {}
    skels_2d = np.load(root / "skel_2d_4dhuman.npz")
    skels_3d = np.load(root / "skel_3d_4dhuman.npz")
    boxes = np.load(root / "boxes.npz")
    for cam_path in root.glob("cameras/*.npz"):
        sequence_name = cam_path.stem
        img_dir = root / "images" / sequence_name
        cam = dict(np.load(cam_path))
        solutions[sequence_name] = baseline(
            cameras=cam,
            boxes=boxes[sequence_name],
            skels_3d=skels_3d[sequence_name][:, :, OPENPOSE_TO_OURS],
            skels_2d=skels_2d[sequence_name][:, :, OPENPOSE_TO_OURS],
            image_dir=img_dir,
        )
    np.savez_compressed("dummy-solution.npz", **solutions)


if __name__ == "__main__":
    main(Path("data/"))
