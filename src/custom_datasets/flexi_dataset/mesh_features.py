import dataclasses
import fpsample
import random
import torch
import trimesh
import math
import numpy as np

from src.utils.common import init_logger
from src.utils.mesh import normalize_mesh, quantize_mesh, y_up_coord_to_z_up, FACE_COND_FACTOR


logger = init_logger()


@dataclasses.dataclass
class MeshMeta:
    model_id: str
    raw_obj: str = None
    watertight_geometry_info: str = None
    render_images: dict = None
    face_info: dict = None
    category: str = None
    category_path_en: str = None


def set_seed(seed=None):
    random.seed(seed)
    np.random.seed(seed)


class MeshFeatures:
    def __init__(self, dataset_args, project_args, mesh_geometry: trimesh.Geometry = None, mesh_meta=None, seed=None, 
                 max_triangles=64000, mode='train', augment=False, tokenizer=None,deepmesh=None, **kwargs):
        self.dataset_args = dataset_args
        self.project_args = project_args
        self.mesh_geometry = mesh_geometry
        self.is_invalid_mesh = False
        self.mesh_meta = mesh_meta
        self.max_triangles = max_triangles
        self.pad_id = -1
        self.tokenizer = tokenizer
        self.deepmesh = deepmesh

        self.augment = ((mode == 'train') and augment)
        
        set_seed(seed)

        if mesh_geometry is None:
            self.is_invalid_mesh = True
        else:
            vertices, faces = self.mesh_geometry.vertices, self.mesh_geometry.faces
            if self.dataset_args.convert_y_up_to_z_up:
                vertices = y_up_coord_to_z_up(torch.from_numpy(vertices)).numpy()
            
            try:
                quantized_mesh = self._quantize_mesh(vertices, faces)
                self._make_surface_points_features(quantized_mesh)
            except BaseException as e:
                self.is_invalid_mesh = True
                import traceback
                logger.error(f"_quantize_mesh error: {e}, traceback \n{traceback.format_exc()}, mesh_meta is {self.mesh_meta}")

        set_seed()

    def _make_surface_points_features(self, quantized_mesh):
        n_samples = self.dataset_args.n_samples
        
        norm_quantized_mesh = trimesh.Trimesh(vertices=normalize_mesh(quantized_mesh.vertices, scale_range=(-0.95, 0.95)), 
                                              faces=quantized_mesh.faces, process=False)
        points, face_idx = norm_quantized_mesh.sample(n_samples + 10000, return_index=True)
        normals = norm_quantized_mesh.face_normals[face_idx]
        pc_normal = np.concatenate([points, normals], axis=-1, dtype=np.float16)
        all_surface_points = torch.from_numpy(pc_normal)  # torch.FloatTensor, shape: (n, 6)

        if all_surface_points.isnan().any():
            right_point_indices = torch.where(torch.logical_not(all_surface_points.sum(dim=-1).isnan()))[0]
            if len(right_point_indices) >= n_samples:
                all_surface_points = all_surface_points[right_point_indices]

        if self.dataset_args.sampling_strategy == "random":

            perm = torch.randperm(all_surface_points.shape[0])
            indices = perm[:n_samples]
        elif self.dataset_args.sampling_strategy == "fps":
            indices = fpsample.bucket_fps_kdline_sampling(pc=all_surface_points[:, :3], n_samples=n_samples, h=5)
        else:
            raise NotImplementedError(f"sampling strategy {self.dataset_args.sampling_strategy} not implemented")
        
        self.point_cond = all_surface_points[indices]

    def _quantize_mesh(self, vertices, faces):
        if self.augment:
            vertices = self.augment_mesh(vertices)
        normalized_vertices = normalize_mesh(vertices)
        quantize_vertices, quantize_faces, quantized_mesh = quantize_mesh(
            vertices=normalized_vertices, 
            faces=faces, 
            cube_reso=self.project_args.tokenizer.params.n_discrete_size
        )
        

        quantize_vertices = np.flip(quantize_vertices, axis=-1)


        max_vertices = self.max_triangles * 3
        vertices = np.ones((max_vertices, 3), dtype=np.float64) * self.pad_id
        faces = np.ones((self.max_triangles, 3), dtype=np.int64) * self.pad_id

        if len(quantize_vertices) <= max_vertices:
            vertices[:len(quantize_vertices)] = quantize_vertices
            faces[:len(quantize_faces)] = quantize_faces
        else:
            vertices[:max_vertices] = quantize_vertices[:max_vertices]
            faces[:self.max_triangles] = quantize_faces[:self.max_triangles]
            logger.warning(f"self.max_triangles: {self.max_triangles}, len(quantize_faces): {len(quantize_faces)}, mesh_meta: {self.mesh_meta}")
        self.connect_map = faces # assume batch size =1 only for training
        if self.deepmesh:
            self.seq_tokens = self.tokenizer.tokenize(trimesh.Trimesh(vertices=quantize_vertices, faces=quantize_faces))
        #self.remove_face_index = create_hole_on_mesh(trimesh.Trimesh(vertices=quantize_vertices, faces=quantize_faces))
        self.remove_face_index = self.connect_map
        self.vertices = torch.from_numpy(vertices).clone()
        self.faces = torch.from_numpy(faces).clone()
        face_num = len(quantize_faces)
        if torch.isnan(self.vertices).any():
            self.is_invalid_mesh = True
        if torch.isnan(self.faces).any():
            self.is_invalid_mesh = True
        self.face_count_id = torch.tensor(math.ceil(face_num / FACE_COND_FACTOR), dtype=torch.long)
        if self.face_count_id is None:
            self.is_invalid_mesh = True
        

        gt_vertices = vertices.copy()
        # gt_vertices[:len(quantize_vertices)] = normalize_mesh(vertices[:len(quantize_vertices)])
        gt_vertices = gt_vertices[faces.clip(0).astype(np.int64)]  # shape: (n_face, 3, 3)
        gt_vertices[faces[:, 0] == self.pad_id] = float('nan')
        self.gt_vertices = torch.from_numpy(gt_vertices)

        if face_num < 100: 
            self.is_invalid_mesh = True

        return quantized_mesh
    
    @staticmethod
    def augment_mesh(vertices, scale_min=0.95, scale_max=1.05):
        scaled_v = normalize_mesh(vertices)  # shape: (n, 3)
        for i in range(3):
            scale = random.uniform(scale_min, scale_max)
            scaled_v[:, i] *= scale
        return scaled_v
