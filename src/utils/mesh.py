import dataclasses
import torch
import trimesh
import numpy as np
from collections import deque,defaultdict
from einops import rearrange
from typing import Dict, List

from src.utils.ply_helper import write_ply

from itertools import combinations


FACE_COND_FACTOR = 512


def normalize_mesh(vertices: np.ndarray, scale_range: tuple=(-1.0, 1.0)) -> np.ndarray:
    lower, upper = scale_range
    scale_per_axis = (vertices.max(0) - vertices.min(0)).max()
    center_xyz = 0.5 * (vertices.max(0) + vertices.min(0))
    normalized_xyz = (vertices - center_xyz) / scale_per_axis   # scaled into range (-0.5, 0.5)
    vertices = normalized_xyz * (upper - lower)
    return vertices

def quantize_vertices(vertices, cube_reso):

    vertices = vertices * 0.5
    assert abs(abs(vertices).max() - 0.5) < 1e-3  # [-0.5, 0.5]
    vertices = (vertices + 0.5) * cube_reso  # [0, num_tokens]
    vertices -= 0.5  # for evenly distributed, [-0.5, num_tokens -0.5] will be round to 0 or num_tokens (-1)
    vertices_quantized_ = np.clip(vertices.round(), 0, cube_reso - 1).astype(int)  # [0, num_tokens -1]
    return vertices_quantized_

def remove_degeneration_elements(mesh: trimesh.Trimesh):
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    return mesh
    
def sort_mesh(mesh: trimesh.Trimesh):
    sort_inds = np.lexsort(mesh.vertices.T)
    vertices = mesh.vertices[sort_inds]
    sort_res = np.argsort(sort_inds)
    faces = [sort_res[f] for f in mesh.faces]

    faces  = [np.roll(sub_arr,-np.argmin(sub_arr)) for sub_arr in faces]

    faces = sorted(faces, key=lambda face: (face[0], face[1], face[2]))
    return vertices, faces

    
def quantize_mesh(vertices, faces, cube_reso=128):
    vertices_quantized_ = quantize_vertices(vertices, cube_reso)

    cur_mesh = trimesh.Trimesh(vertices=vertices_quantized_, faces=faces)
    cur_mesh = remove_degeneration_elements(cur_mesh)
    vertices, faces = sort_mesh(cur_mesh)
    traversed_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    return traversed_mesh.vertices, traversed_mesh.faces, traversed_mesh.copy()


def resort_input_ids(input_ids, bos_token_id):

    assert len(input_ids) % 9 == 0 and len(input_ids) // 9 > 1
    raw_device = input_ids.device
    bos_ids = None
    if input_ids[0] == bos_token_id:
        bos_ids, input_ids = input_ids[:9], input_ids[9:]
    input_ids = input_ids.cpu()
    
    faces = rearrange(input_ids, '(f c) -> f c', c=9)
    repeat_indices = torch.all(faces == faces[-1], dim=-1)
    if repeat_indices.sum() >= 10:
        faces =faces[repeat_indices.logical_not()]
    
    vertices = rearrange(faces, 'f (v c) -> (f v) c', c=3)
    vertices = torch.flip(vertices, dims=(-1,))  
    faces = torch.arange(vertices.shape[0]).reshape(-1, 3)
    cur_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    valid_faces = cur_mesh.area_faces > 1e-10
    cur_mesh.update_faces(valid_faces)
    cur_mesh = remove_degeneration_elements(cur_mesh)
    
    vertices, faces = sort_mesh(cur_mesh)
    
    vertices = torch.from_numpy(vertices).flip(dims=(-1,))
    faces = torch.tensor(faces).long().clip(0)
    final_vertices = vertices[faces]
    sorted_ids = rearrange(final_vertices, 'f v c -> (f v c)', c=3).to(device=raw_device, dtype=torch.long)
    
    if bos_ids is not None:
        sorted_ids = torch.concat([bos_ids, sorted_ids])
    
    return sorted_ids


def y_up_coord_to_z_up(points):
    rotate_m = torch.tensor([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ], dtype=points.dtype)
    transfer_p = torch.matmul(rotate_m, rearrange(points, '... n c -> ... c n'))
    return rearrange(transfer_p, '... c n -> ... n c')


def z_up_coord_to_y_up(points):
    rotate_m = torch.tensor([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0]
    ], dtype=points.dtype)
    transfer_p = torch.matmul(rotate_m, rearrange(points, '... n c -> ... c n'))
    return rearrange(transfer_p, '... c n -> ... n c')


def post_process_mesh(mesh_coords, filename: str):
    mesh_coords = mesh_coords[~torch.isnan(mesh_coords[:, 0, 0])]   # nvalid_face x 3 x 3
    vertices = mesh_coords.reshape(-1, 3)
    vertices = z_up_coord_to_y_up(vertices)
    vertices_index = np.arange(len(vertices))   # 0, 1, ..., 3 x face
    triangles = vertices_index.reshape(-1, 3)
    write_ply(
        np.asarray(vertices.float().cpu()),
        None,
        np.asarray(triangles),
        filename
    )
    return vertices 


@dataclasses.dataclass
class GenerationCtx:
    out: torch.Tensor = None
    all_current_face: List[tuple] = None
    all_points: List[torch.Tensor] = None
    all_points_count: List[torch.Tensor] = None
    all_faces: List[torch.Tensor] = None
    current_scores: torch.Tensor = None
    end_beam_indices: List = None

