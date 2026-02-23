#no cos s3 logic
import json
import os
import random
import torch
import trimesh
import pyassimp
import numpy as np
from typing import List

from src.custom_datasets.flexi_dataset.base_dataset import BaseDataset
from src.custom_datasets.flexi_dataset.bucket_weights import scaled_category_weights
from src.custom_datasets.flexi_dataset.mesh_features import MeshFeatures, MeshMeta
from src.utils.common import init_logger, import_module_or_data
def gameai_cos_sign_reduce(self):
    return (self.__class__, (self._username, self._password))

logger = init_logger()



class MeshDataset(BaseDataset):
    def __init__(self, dataset_args, project_args, debug_mode=False, mode='train', batch_size=2, dtype=torch.float16, 
                 device=None, default_obj_key='obj_url', tokenizer=None,deepmesh=False, **kwargs):
        super().__init__(dataset_args, project_args, debug_mode, mode, batch_size, dtype, device)
        
        if mode == "train":  
            sample_info_list = dataset_args.train_sample_info_list if not debug_mode else dataset_args.val_sample_info_list
        elif mode == "val": 
            sample_info_list = dataset_args.val_sample_info_list
        elif mode == "test": 
            sample_info_list = dataset_args.eval_sample_info_list
        else:
            raise NotImplementedError(f"Unknown mode {mode}")
        
        self.all_mesh_meta = []
        self.face_bucket = {}
        self.invalid_mesh_record = set()
        self.max_triangles = 0
        face_bucket_interval = dataset_args.face_bucket_interval
        self.tokenizer = tokenizer
        self.deepmesh = deepmesh
        
        model_id_set = set()
        
        self.default_obj_key = default_obj_key
        for sample_info in sample_info_list:
            item_type = list(sample_info.keys())[0]
            info = sample_info[item_type]

            for sample_urls_file in info.index_file_list:
                with open(sample_urls_file) as f:
                    metadata = json.load(f)
                for model_info in metadata:
                    model_id = f"item_{item_type}_raw_{model_info['model_id']}"

                    obj_files = model_info['obj_file']
                    try:
                        raw_obj = obj_files[self.default_obj_key]
                    except:
                        raw_obj = obj_files['separate_by_loose_origin_url']
                   
                    mesh_meta = MeshMeta(
                        model_id=model_id,
                        raw_obj=raw_obj,
                        face_info=model_info['face_info'],
                        category=model_info['category'],
                        category_path_en=model_info['category_path_en'],
                    )
                    try:
                        triangle_face_cnt = mesh_meta.face_info.get('face_cnt',2000) 
                    except:
                        triangle_face_cnt = 2000

                    if triangle_face_cnt is None:
                        logger.warning(f"{model_id} has no face_cnt atrribute, mesh_meta is {mesh_meta}")
                        self.invalid_mesh_record.add(model_id)
                        continue

                    need_face_range = info.get('need_face_range', None)
                    if need_face_range is not None and (triangle_face_cnt < need_face_range[0] or triangle_face_cnt > need_face_range[1]):
                        continue
                    
                    need_category = info.get('need_category', None)
                    if need_category is not None and mesh_meta.category not in need_category:
                        continue
                    
                    model_id_set.add(model_id)
                    self.max_triangles = max(self.max_triangles, triangle_face_cnt)
                    
                    category = mesh_meta.category if mesh_meta.category is not None else 'unknown'
                    face_bucket_id = triangle_face_cnt // face_bucket_interval

                    face_bucket_key = (face_bucket_id * face_bucket_interval, (face_bucket_id + 1) * face_bucket_interval)
                    self.face_bucket.setdefault(face_bucket_key, {}).setdefault(category, []).append(mesh_meta)
                    self.all_mesh_meta.append(mesh_meta)
        
        self.bucket_num_info = {}
        self.category_num_in_each_bucket = {}
        for k in sorted(self.face_bucket.keys()):
            face_bucket_sample_num = 0
            for category in sorted(self.face_bucket[k].keys()):
                meta_list = self.face_bucket[k][category]
                self.category_num_in_each_bucket.setdefault(k, {})[category] = len(meta_list)
                face_bucket_sample_num += len(meta_list)
            self.bucket_num_info[k] = face_bucket_sample_num
        
        bucket_weights_method = import_module_or_data(dataset_args.bucket_weights_method)
        self.bucket_weights = bucket_weights_method(bucket_num_info=self.bucket_num_info, project_args=project_args)
        
        self.category_weights_in_each_bucket = scaled_category_weights(self.category_num_in_each_bucket,
                                                                       category_scale_conf=dataset_args.get('category_scale_conf', None))

        self.rank = kwargs.get('rank', None)
        self.world_size = kwargs.get('world_size', None)
        
        if self.rank == 0:
            logger.info(f"mode: {mode}, all_mesh_meta num is {len(self.all_mesh_meta)}")
            logger.info(f"bucket_num_info is {self.bucket_num_info},\n")
            logger.info(f"bucket_weights is {self.bucket_weights},\n")
            
            display_bucket_num = min(10, len(self.bucket_weights))
            display_indices = np.random.choice(list(range(len(self.bucket_weights))), size=display_bucket_num, replace=False)
            for idx in display_indices:
                face_bucket = list(self.bucket_weights.keys())[idx]
                logger.info(f"bucket {face_bucket}, category_num in this bucket is {self.category_num_in_each_bucket[face_bucket]},\n")
                logger.info(f"bucket {face_bucket}, category_weights in this bucket {self.category_weights_in_each_bucket[face_bucket]},\n")
            
            if len(model_id_set) != len(self.all_mesh_meta):
                logger.warning(f"unique model num: {len(model_id_set)}, mesh_meta num {len(self.all_mesh_meta)}")
        
        del model_id_set
        
        self.required_keys = ['point_cond', 'vertices', 'faces', 'gt_vertices', 'face_count_id']
        if deepmesh:
            self.required_keys.append('seq_tokens')
        self.augment_prob = dataset_args.get('augment_prob', 0.5)
        self.s3 = None
        
    def __len__(self):
        return len(self.all_mesh_meta)
    def process_fbx(self,fbx_file):
        meshes = []
        with pyassimp.load(fbx_file, file_type="fbx") as scene:
            for mesh_data in scene.meshes:
                vertices = mesh_data.vertices
                faces = mesh_data.faces
                if faces.shape[0] >= 10 and vertices.shape[0] >= 10 and faces.shape[1] == 3:
                    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                    meshes.append(mesh)

        if len(meshes) > 1:
            combined_mesh = trimesh.util.concatenate(meshes)
        else:
            combined_mesh = meshes[0] if meshes else None
        return combined_mesh, meshes
    def make_features_for_meta(self, mesh_meta, max_triangles, seed=None, augment=False):
        if mesh_meta.raw_obj.endswith('.fbx'):
            raw_obj, _ = self.process_fbx(mesh_meta.raw_obj)
        else:
            raw_obj = trimesh.load(mesh_meta.raw_obj, force='mesh', file_type='obj') 

        mesh_features = MeshFeatures(
            dataset_args=self.dataset_args,
            project_args=self.project_args,
            mesh_geometry=raw_obj,
            mesh_meta=mesh_meta,
            seed=seed,
            mode=self.mode,
            max_triangles=max_triangles,
            augment=augment,
            deepmesh=self.deepmesh,
            tokenizer=self.tokenizer,
        )
        return mesh_features
        
    def __getitem__(self, index):

        bucket_key = random.choices(population=list(self.bucket_weights.keys()), weights=list(self.bucket_weights.values()), k=1)[0]
        bucket_results = self.face_bucket[bucket_key]
        category_weights = self.category_weights_in_each_bucket[bucket_key]
        
        example = {k: [] for k in self.required_keys}
        
        batch_idx = 0
        while batch_idx < self.batch_size:
            category = random.choices(population=list(category_weights.keys()), weights=list(category_weights.values()), k=1)[0]
            category_metas = bucket_results[category]
            
            mesh_meta = random.choice(category_metas)
            if mesh_meta.model_id in self.invalid_mesh_record:
                logger.warning(f"{mesh_meta.model_id} is invalid_mesh, skip it")
                continue
            
            mesh_features = self.make_features_for_meta(mesh_meta, max_triangles=bucket_key[1] + 1000,
                                                        augment=random.random() < self.augment_prob)
            
            if mesh_features.is_invalid_mesh:
                logger.error(f"{mesh_meta.model_id} is invalid_mesh, skip it")
                self.invalid_mesh_record.add(mesh_meta.model_id)
                continue
            
            if mesh_features.point_cond.isnan().any():
                logger.error(f"{mesh_meta.model_id} still has nan data, skip it")
                continue
            
            for k, v in example.items():
                v.append(getattr(mesh_features, k))
            batch_idx += 1
        return example
    
    def random_sample(self, seed=None, batch_size=2, mesh_metas: List[MeshMeta] = None, **kwargs):
        if mesh_metas is None:
            np.random.seed(seed)
            mesh_metas = np.random.choice(self.all_mesh_meta, size=batch_size, replace=False)
            np.random.seed()
        
        
        example = {k: [] for k in self.required_keys}
        sample_info = []
        for mesh_meta in mesh_metas:
            mesh_features = self.make_features_for_meta( mesh_meta, max_triangles=self.max_triangles + 1000, seed=seed)
            for k, v in example.items():
                v.append(getattr(mesh_features, k))
            sample_info.append({
                'model_id': mesh_meta.model_id,
                'raw_obj': mesh_meta.raw_obj,
                'face_info': mesh_meta.face_info,
                'category': mesh_meta.category,
                'category_path_en': mesh_meta.category_path_en,
            })
        batch_data = collate_fn_for_collected_batch([example])
        return batch_data, sample_info      


def collate_fn_for_collected_batch(examples):
    batch_example = examples[0]
    final_batch_data = {}
    for k, batch_feature_list in batch_example.items():
        if k in ['point_cond', 'vertices', 'faces', 'gt_vertices', 'face_count_id', 'seq_tokens']:
            batch_tensor = torch.stack(batch_feature_list, dim=0).detach()
        else:
            batch_tensor = batch_feature_list

        if batch_tensor is not None:
            final_batch_data[k] = batch_tensor
    return final_batch_data


def mesh_bucket_collate_fn(batch_list):
    if not batch_list:
        return {}
    all_keys = set()
    for item in batch_list:
        if isinstance(item, dict):
            all_keys.update(item.keys())
    
    result = {}
    for key in all_keys:
        values = []
        for item in batch_list:
            if isinstance(item, dict) and key in item:
                val = item[key]
                if isinstance(val, list):
                    values.extend(val)
                else:
                    values.append(val)
        
        if values and isinstance(values[0], torch.Tensor):
            if key in ['vertices', 'gt_vertices', 'point_cond']:
                values = [v.float() if v.dtype not in [torch.float16, torch.float32, torch.float64] else v for v in values]
            
            try:
                result[key] = torch.stack(values, dim=0)
            except RuntimeError as e:
                logger.warning(f"{key}: {e}, shapes: {[v.shape for v in values]}")
                result[key] = values
        else:
            result[key] = values
    
    return result
