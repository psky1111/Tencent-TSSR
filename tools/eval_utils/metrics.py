import torch
import trimesh
import numpy as np
import point_cloud_utils as pcu
from scipy.spatial import cKDTree
from pathlib import Path
import os
import shutil

from src.utils.common import init_logger
from src.utils.mesh import remove_degeneration_elements


logger = init_logger()

    
def coords_to_mesh(mesh_coords, scale_factor=1.0):
    mesh_coords = mesh_coords[~torch.isnan(mesh_coords[:, 0, 0])]   # nvalid_face x 3 x 3
    vertices = mesh_coords.reshape(-1, 3) * scale_factor
    vertices_index = np.arange(len(vertices))   # 0, 1, ..., 3 x face
    triangles = vertices_index.reshape(-1, 3)

    vertices = np.asarray(vertices.float().cpu())
    faces = np.asarray(triangles)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    valid_faces = mesh.area_faces > 1e-10
    mesh.update_faces(valid_faces)
    mesh = remove_degeneration_elements(mesh)
    return mesh


def calc_mesh_iou(mesh1, mesh2):
    bounds1 = mesh1.bounds  # shape: (2, 3)
    bounds2 = mesh2.bounds  # shape: (2, 3)
    if bounds2 is None:
        return 0.0
    
    intersect_min = np.maximum(bounds1[0], bounds2[0])
    intersect_max = np.minimum(bounds1[1], bounds2[1])
    
    intersect_whd = np.maximum(intersect_max - intersect_min, 0.0)
    intersection_vol = np.prod(intersect_whd)
    
    vol1 = np.prod(bounds1[1] - bounds1[0])
    vol2 = np.prod(bounds2[1] - bounds2[0])

    union_vol = vol1 + vol2 - intersection_vol
    
    if union_vol == 0:
        return 0.0
    else:
        return intersection_vol / union_vol


def apply_normalize(mesh):
    """
    normalize mesh to [-1, 1]
    """
    bbox = mesh.bounds
    center = (bbox[1] + bbox[0]) / 2
    scale = (bbox[1] - bbox[0]).max()

    mesh.apply_translation(-center)
    mesh.apply_scale(1 / scale * 2 * 0.95)

    return mesh


def compute_hausdorff_and_chamfer_distance(mesh1, mesh2, pc_num=10000):
    try:
        points1, _ = mesh1.sample(pc_num, return_index=True)
        points2, _ = mesh2.sample(pc_num, return_index=True)
        hausdorff_dist = pcu.hausdorff_distance(points1, points2)
        chamfer_dist = pcu.chamfer_distance(points1, points2)
        return hausdorff_dist, chamfer_dist
    except BaseException as e:
        logger.error(f"metric error, detail is {e}")
        return 1000, 1000


from glob import glob

def evaluate_comprehensive_1nna(root_path):

    base_root = Path(root_path)
    datasets = ['obj', 'shapenet', 'think']
    face_ranges = ['0k-2k_faces', '2k-4k_faces', '4k-8k_faces']
    
    results = {}

    for ds in datasets:
        for fr in face_ranges:
            current_path = base_root / ds / "unknown" /fr 
            if not current_path.exists():
                continue
                
            print(f"--- Evaluating Dataset: {ds} | Range: {fr} ---")
            
            # Use the core 1-NNA logic from the previous step
            # Note: root.glob("**/sample_*") will find nested samples in your path
            score = calculate_1nna_rebuttal(current_path)
            
            if score is not None:
                results[f"{ds}_{fr}"] = score
                
    return results

def calculate_1nna_rebuttal(root_path, pc_num=10000):
    """
    Revised for specific deep path structure:
    .../sample_72/meshtron_step_checkpoint-7000/seed_123/generated_mesh_199.ply
    """
    root = Path(root_path)
    # Search for all sample directories (e.g., sample_0, sample_1, ...)
    sample_dirs = sorted(list(root.glob("**/sample_*")), key=lambda x: x.name)
    
    gen_points = []
    gt_points = []
    valid_samples = 0

    logger.info(f"Found {len(sample_dirs)} sample directories. Starting sampling...")

    for sample in sample_dirs:
        # 1. Locate the Generated file 
        # Looking for generated_mesh_199.ply inside the seed/checkpoint folders
        gen_matches = list(sample.glob("**/generated_mesh_199.ply"))
        
        # 2. Locate the GT file
        # Assuming GT is in a peer 'gt' folder or inside the sample folder
        # Adjust '**/gt/*.ply' if your GT has a specific name
        gt_matches = list(sample.glob("**/gt_mesh.ply"))

        if not gen_matches or not gt_matches:
            # logger.warning(f"Missing pair for {sample.name}")
            continue

        try:
            # Load and sample GT
            gt_mesh = trimesh.load(gt_matches[0])
            gt_pts = gt_mesh.sample(pc_num) if len(gt_mesh.faces) > 0 else gt_mesh.vertices
            if len(gt_pts) < pc_num:
                gt_pts = gt_pts[np.random.choice(len(gt_pts), pc_num, replace=True)]
            else:
                gt_pts = gt_pts[:pc_num] # Ensure exact count

            # Load and sample Generated
            gen_mesh = trimesh.load(gen_matches[0])
            gen_pts = gen_mesh.sample(pc_num) if len(gen_mesh.faces) > 0 else gen_mesh.vertices
            if len(gen_pts) < pc_num:
                gen_pts = gen_pts[np.random.choice(len(gen_pts), pc_num, replace=True)]
            else:
                gen_pts = gen_pts[:pc_num]

            gt_points.append(gt_pts)
            gen_points.append(gen_pts)
            valid_samples += 1

        except Exception as e:
            logger.error(f"Error loading {sample.name}: {e}")

    if valid_samples == 0:
        logger.error("No valid GT/Generated pairs found.")
        return None

    logger.info(f"Successfully sampled {valid_samples} pairs.")

    # Stack: [N_gen, pc_num, 3] + [N_gt, pc_num, 3]
    all_points = np.concatenate([np.array(gen_points), np.array(gt_points)], axis=0)
    total = valid_samples * 2

    # Compute Distance Matrix
    dist_matrix = np.zeros((total, total))
    for i in range(total):
        for j in range(i + 1, total):
            # Chamfer Distance is used for 1-NNA in 3D generation tasks
            d = pcu.chamfer_distance(all_points[i], all_points[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
            
    np.fill_diagonal(dist_matrix, np.inf)

    # 1-NN Accuracy logic
    labels = np.concatenate([np.zeros(valid_samples), np.ones(valid_samples)])
    nn_indices = np.argmin(dist_matrix, axis=1)
    nn_labels = labels[nn_indices]
    
    accuracy = np.mean(labels == nn_labels)
    logger.info(f"1-NNA Score: {accuracy:.4f} (Ideal: 0.5)")
    return accuracy

def normalize_mesh_cdf1(mesh):

    mesh_normalized = mesh.copy()
    centroid = mesh_normalized.centroid
    mesh_normalized.apply_translation(-centroid)
    max_distance = np.max(np.linalg.norm(mesh_normalized.vertices, axis=1))
    if max_distance > 1e-8:
        mesh_normalized.apply_scale(1.0 / max_distance)
    return mesh_normalized

def compute_f1_score(gt_mesh, pred_mesh, threshold=0.001, num_samples=10000):

    gt_mesh = gt_mesh
    pred_mesh = pred_mesh
    try:
        # Ensure meshes have vertices
        if len(gt_mesh.vertices) == 0 or len(pred_mesh.vertices) == 0:
            return 0.0

        gt_points, _ = trimesh.sample.sample_surface(gt_mesh, num_samples)
        pred_points, _ = trimesh.sample.sample_surface(pred_mesh, num_samples)

        # Create KD-Trees for efficient nearest neighbor search
        gt_tree = cKDTree(gt_points)
        pred_tree = cKDTree(pred_points)

        # 1. Precision: For each predicted point, find its distance to the nearest ground truth point.
        dist_pred_to_gt, _ = gt_tree.query(pred_points, k=1)
        precision = np.mean((dist_pred_to_gt < threshold).astype(float))

        # 2. Recall: For each ground truth point, find its distance to the nearest predicted point.
        dist_gt_to_pred, _ = pred_tree.query(gt_points, k=1)
        recall = np.mean((dist_gt_to_pred < threshold).astype(float))
        
        # 3. F1 Score
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
            
        return f1

    except Exception as e:
        print(f"Error calculating F1 score: {e}")
        return 0.0
    
def compute_chamfer_l2(gt_mesh, pred_mesh, num_samples=10000):

    gt_mesh = normalize_mesh_cdf1(gt_mesh)
    pred_mesh = normalize_mesh_cdf1(pred_mesh)
    try:
        # Ensure meshes have vertices before proceeding
        if len(gt_mesh.vertices) == 0 or len(pred_mesh.vertices) == 0:
            print("Warning: One or both meshes have 0 vertices. Returning CD=1.0")
            return 1.0

        # Sample points from the surfaces
        gt_points, _ = trimesh.sample.sample_surface(gt_mesh, num_samples)
        pred_points, _ = trimesh.sample.sample_surface(pred_mesh, num_samples)

        # Create KD-Trees for efficient nearest neighbor search
        gt_tree = cKDTree(gt_points)
        pred_tree = cKDTree(pred_points)

        # Find the nearest neighbor distances
        # Distance from predicted points to the nearest ground truth point
        dist_pred_to_gt, _ = gt_tree.query(pred_points, k=1)
        
        # Distance from ground truth points to the nearest predicted point
        dist_gt_to_pred, _ = pred_tree.query(gt_points, k=1)

        # Square the distances (L2 squared)
        dist_pred_to_gt_sq = dist_pred_to_gt**2
        dist_gt_to_pred_sq = dist_gt_to_pred**2

        # Calculate the mean of the squared distances
        mean_sq_dist_pred_to_gt = np.mean(dist_pred_to_gt_sq)
        mean_sq_dist_gt_to_pred = np.mean(dist_gt_to_pred_sq)
        
        # The Chamfer Distance is the sum of these two mean values
        chamfer_dist = mean_sq_dist_pred_to_gt + mean_sq_dist_gt_to_pred
        
        return chamfer_dist * 1000

    except Exception as e:
        print(f"Error calculating Chamfer Distance: {e}")
        return 1.0 # Return a large error value


def compute_normal_consistency(gt_mesh, pred_mesh, num_samples=10000):
    """
    Computes the Normal Consistency (NC) between two meshes.

    ROBUST IMPLEMENTATION with sanity checks.
    """
    try:
        # Sanity check: ensure meshes have faces and normals
        if not hasattr(gt_mesh, 'face_normals') or not hasattr(pred_mesh, 'face_normals'):
            print("Warning: One or both meshes do not have face normals. Returning 0.")
            return 0.0
        if len(gt_mesh.face_normals) == 0 or len(pred_mesh.face_normals) == 0:
            print("Warning: One or both meshes have 0 faces. Returning 0.")
            return 0.0

        # Sample points from the predicted mesh
        pred_points, face_indices_pred = trimesh.sample.sample_surface(pred_mesh, num_samples)
        normals_pred = pred_mesh.face_normals[face_indices_pred]
        
        # Find the closest points on the ground truth mesh and their corresponding face indices
        _, _, face_indices_gt = gt_mesh.nearest.on_surface(pred_points)
        normals_gt = gt_mesh.face_normals[face_indices_gt]
        
        # Calculate the absolute dot product (cosine similarity)
        consistency = np.abs(np.sum(normals_pred * normals_gt, axis=1))
        
        return np.mean(consistency)

    except Exception as e:
        print(f"Error calculating Normal Consistency: {e}")
        return 0.0
    
def mesh_collect(input_dir, output_root):
    # Create output subdirectories
    gen_output_dir = os.path.join(output_root, "generated")
    gt_output_dir = os.path.join(output_root, "gt")
    os.makedirs(gen_output_dir, exist_ok=True)
    os.makedirs(gt_output_dir, exist_ok=True)

    path_parts = Path(input_dir).parts
    unique_name = f"{path_parts[-3]}_{path_parts[-1]}" 

    found_gen = False
    found_gt = False

    # Iterate through files in the input directory
    for filename in os.listdir(input_dir):
        # Match generated_mesh_XXX.ply
        if filename.startswith("generated_mesh_199") and filename.endswith(".ply"):
            src_path = os.path.join(input_dir, filename)
            dst_path = os.path.join(gen_output_dir, f"{unique_name}.ply")
            shutil.copy2(src_path, dst_path)
            found_gen = True
            
        # Match gt_mesh.ply
        elif filename == "gt_mesh.ply":
            src_path = os.path.join(input_dir, filename)
            dst_path = os.path.join(gt_output_dir, f"{unique_name}.ply")
            shutil.copy2(src_path, dst_path)
            found_gt = True

    if found_gen and found_gt:
        logger.info(f"Successfully collected pair: {unique_name}")
    else:
        logger.warning(f"Missing files in {input_dir}.")

def compute_metrics(gt_coords, gen_coords, pc_num=10000):

    gt_mesh = coords_to_mesh(gt_coords, scale_factor=2.0) 
    gen_mesh = coords_to_mesh(gen_coords, scale_factor=2.0)
    
    results = {}
    bbox_iou = calc_mesh_iou(gt_mesh, gen_mesh)
    results['bbox_iou'] = bbox_iou
    
    hausdorff_dist, chamfer_dist = compute_hausdorff_and_chamfer_distance(gt_mesh, gen_mesh, pc_num)
    results['hausdorff_dist'], results['chamfer_dist'] = hausdorff_dist, chamfer_dist
    norm_consistency = compute_normal_consistency(gt_mesh, gen_mesh, pc_num)
    f1_score = compute_f1_score(gt_mesh, gen_mesh, pc_num)
    cdl2 = compute_chamfer_l2(gt_mesh, gen_mesh, pc_num)
    
    norm_gt_mesh = apply_normalize(gt_mesh)
    norm_gen_mesh = apply_normalize(gen_mesh)
    norm_hausdorff_dist, norm_chamfer_dist = compute_hausdorff_and_chamfer_distance(norm_gt_mesh, norm_gen_mesh, pc_num)

    results['norm_hausdorff_dist'], results['norm_chamfer_dist'] = norm_hausdorff_dist, norm_chamfer_dist
    results['norm_consistency'], results['f1_score'] = norm_consistency, f1_score
    results['chamfer_dist_l2'] = cdl2
    
    return results
