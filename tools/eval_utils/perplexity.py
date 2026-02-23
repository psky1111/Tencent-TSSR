import os
import time
import torch
import numpy as np
from torch import nn, Tensor
from collections import defaultdict, OrderedDict
from src.utils.ply_helper import write_ply
from accelerate.utils import set_seed

import matplotlib.pyplot as plt  
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  
from matplotlib.animation import FuncAnimation  
from src.utils.misc import SmoothedValue
from tools.eval_utils.visualization import create_animation, create_multi_animation
from tools.eval_utils.metrics import compute_hausdorff_and_chamfer_distance
import random
import trimesh

def perplexity(neg_log_likelihood: list) -> Tensor:
    # gather per-sequence log likelihood for perplexity
    nll_chunk = torch.cat(neg_log_likelihood, dim=0)
    return torch.exp(nll_chunk.mean())



def post_process_mesh(mesh_coords: Tensor, filename: str):
    mesh_coords = mesh_coords[~torch.isnan(mesh_coords[:, 0, 0])]   # nvalid_face x 3 x 3
    vertices = mesh_coords.reshape(-1, 3)
    vertices_index = np.arange(len(vertices))   # 0, 1, ..., 3 x face
    triangles = vertices_index.reshape(-1, 3)
    write_ply(
        np.asarray(vertices.float().cpu()),
        None,
        np.asarray(triangles),
        filename
    )
    return vertices 

def coords2vfs(mesh_coords):
    mesh_coords = mesh_coords[~torch.isnan(mesh_coords[:, 0, 0])]   # nvalid_face x 3 x 3
    vertices = mesh_coords.reshape(-1, 3)
    vertices_index = np.arange(len(vertices))   # 0, 1, ..., 3 x face
    triangles = vertices_index.reshape(-1, 3)

    vertices = np.asarray(vertices.float().cpu())
    faces = np.asarray(triangles)
    return vertices, faces

def compute_metrics(gt_coords, gen_coords):
    gt_vts, gt_fts = coords2vfs(gt_coords)
    gen_vts, gen_fts = coords2vfs(gen_coords)

    hd_dist, cf_dist = compute_hausdorff_and_chamfer_distance(gt_vts, gt_fts, gen_vts, gen_fts, pc_num=2000)
    return hd_dist, cf_dist

def process_mesh_gen_animate(mesh_coords: Tensor, filename: str, batch_data=None, max_frames=100):
    gen_vts, gen_fts = coords2vfs(mesh_coords)

    if batch_data is not None:
        gt_vts, gt_fts = coords2vfs(batch_data['gt_vertices'])
        
        gt_pts = batch_data['point_cond']
        gt_pts = np.asarray(gt_pts.cpu())

        sampled_points = gt_pts

        anim = create_multi_animation(gen_vts, gen_fts, gt_vts, gt_fts, sampled_points, max_frames=max_frames)
    else:
        anim = create_animation(gen_vts, gen_fts)

    anim.save(filename, writer='pillow')  
    return anim



def post_process_trimesh(mesh_coords: Tensor, filename: str, clean=True):
    mesh_coords = mesh_coords[~torch.isnan(mesh_coords[:, 0, 0])]   # nvalid_face x 3 x 3
    vertices = mesh_coords.reshape(-1, 3)
    vertices_index = np.arange(len(vertices))   # 0, 1, ..., 3 x face
    triangles = vertices_index.reshape(-1, 3)

    vertices = np.asarray(vertices.cpu())
    triangles = np.asarray(triangles)
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)

    if clean:
        mesh.merge_vertices()
        mesh.update_faces(mesh.unique_faces())
        mesh.fix_normals()

        print(f'[INFO] cleaned vertices: {mesh.vertices.shape[0]}, faces: {mesh.faces.shape[0]}')

    mesh.export(filename)



@torch.no_grad()
def evaluate(
    args,
    curr_epoch,
    model,
    accelerator,
    dataset_loader,
    logger,
    curr_train_iter=-1,
    curr_dataset_name=None,
    point_encoder=None,
    writer=None,
):
    
    model.eval()
    point_encoder.eval()
    net_device = next(model.parameters()).device
    num_batches = len(dataset_loader)
    
    ### parse evaluation status
    if hasattr(dataset_loader.dataset, "dataset_name"):
        dataset_name = dataset_loader.dataset.dataset_name
    else:
        dataset_name = "default"
    task_name_prefix = dataset_name + '_'

    time_delta = SmoothedValue(window_size=10)

    accelerator.wait_for_everyone()
    
    epoch_str = f"[{curr_epoch}/{args.max_epoch}]" if curr_epoch > 0 else ""
    
    if accelerator.is_main_process:
        logger.info("==" * 10)
        logger.info(f"Evaluate Epoch [{curr_epoch}/{args.max_epoch}]")
        logger.info("==" * 10)

    # seed_torch()
    set_seed(accelerator.process_index)

    # create storage directory
    storage_dir = os.path.join(args.checkpoint_dir, task_name_prefix + 'visualization', f'epoch{curr_epoch}')
    if accelerator.is_main_process:
        os.makedirs(storage_dir, exist_ok = True)
    accelerator.wait_for_everyone()

    ### calculate perplexity
    neg_log_likelihood = []

    ### calculate hf and cd distance
    hf_distances, cd_distances = [], []
    for curr_iter, batch_data_label in enumerate(dataset_loader):
        
        curr_time = time.time()
        
        # forward pass to calculate per-sequence negative log likelihood
        with accelerator.autocast():    
            outputs = model(batch_data_label, is_eval=True, point_encoder=point_encoder)
        # [(batch,), (batch,), ...]
        neg_log_likelihood.append(outputs['neg_log_likelihood'])

        ### log status
        time_delta.update(time.time() - curr_time)
        
        if accelerator.is_main_process and curr_iter % args.log_every == 0:
            mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            moving_average_ppl = perplexity(neg_log_likelihood)
            logger.info(
                '; '.join(
                    (
                        f"Evaluate {epoch_str}",
                        f"Batch [{curr_iter}/{num_batches}]",
                        f"perplexity: {moving_average_ppl:0.4f}",
                        f"Evaluating on iter: {curr_train_iter}",
                        f"Iter time {time_delta.avg:0.2f}",
                        f"Mem {mem_mb:0.2f}MB",
                    )
                )
            )
        
        if curr_iter < 1:
        # if accelerator.is_main_process and curr_iter < 1:
            # visualization stage
            with accelerator.autocast():   
                outputs = model(
                    data_dict=batch_data_label,
                    is_eval=True, 
                    is_generate=True,
                    num_return_sequences=args.batchsize_per_gpu,
                    point_encoder=point_encoder,
                )
        
            generated_meshes = outputs["recon_faces"]   # nsample x nf x 3 x 3
        
            for sample_idx in range(args.batchsize_per_gpu):
                # store the generated meshes
                post_process_mesh(
                    generated_meshes[sample_idx],
                    os.path.join(
                        storage_dir, 
                        '_'.join(
                            (   f'{curr_epoch}',
                                f'{accelerator.process_index:04d}',
                                f'{curr_iter:04d}',
                                f'{sample_idx:04d}.ply',
                            )
                        )
                    )
                )

                hf_dist, cd_dist = compute_metrics(generated_meshes[sample_idx], batch_data_label['gt_vertices'][sample_idx])
                hf_distances.append(hf_dist)
                cd_distances.append(cd_dist)

                process_mesh_gen_animate(
                    generated_meshes[sample_idx],
                    os.path.join(
                        storage_dir, 
                        '_'.join(
                            (   
                                f'{curr_epoch}',
                                f'{accelerator.process_index:04d}',
                                f'{curr_iter:04d}',
                                f'{sample_idx:04d}.gif',
                            )
                        )
                    ),
                    batch_data={
                        'gt_vertices': batch_data_label['gt_vertices'][sample_idx],
                        'point_cond': batch_data_label['point_cond'][sample_idx],
                    }
                    # batch_data_label
                )
        ### end of an iteration
        
    ### end of a round

    hf_metrics = torch.Tensor(hf_distances).to(net_device)
    cd_metrics = torch.Tensor(cd_distances).to(net_device)

    all_hf_metrics, all_cd_metrics = accelerator.gather(hf_metrics), accelerator.gather(cd_metrics)

    if accelerator.is_main_process:
        # mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        moving_average_ppl = perplexity(neg_log_likelihood)
        avg_hf_dist = torch.mean(all_hf_metrics).item()
        avg_cd_dist = torch.mean(all_cd_metrics).item()
        logger.info(
            '; '.join(
                (
                    f"Evaluate {epoch_str}",
                    f"Hansdorff distance: {avg_hf_dist:0.4f}",
                    f"Chamfer distance: {avg_cd_dist:0.4f}",
                )
            )
        )

    quantitative = {
        task_name_prefix + 'ppl': perplexity(neg_log_likelihood).item(),
        task_name_prefix + 'batch_chamfer_distance':  torch.mean(all_cd_metrics).item(),
        task_name_prefix + 'batch_hansdoff_distance':  torch.mean(all_hf_metrics).item(),
    }
    if accelerator.is_main_process:
        for k, v in quantitative.items():
            writer.add_scalar(k, v, global_step=curr_epoch)
    accelerator.wait_for_everyone()
    
    return {}, quantitative

@torch.no_grad()
def review_mesh(
    args,
    curr_epoch,
    accelerator,
    batch_data_label,
    logger,
    curr_train_iter=-1,
):
    storage_dir = os.path.join(args.checkpoint_dir, 'training_mesh_review', f'epoch{curr_epoch}')
    if accelerator.is_main_process:
        os.makedirs(storage_dir, exist_ok = True)
    generated_meshes = batch_data_label["gt_vertices"]   # nsample x nf x 3 x 3

    accelerator.wait_for_everyone()
    for sample_idx in range(args.batchsize_per_gpu):
        # store the generated meshes
        post_process_mesh(
            generated_meshes[sample_idx],
            os.path.join(
                storage_dir, 
                '_'.join(
                    (   f'{curr_epoch}',
                        f'{accelerator.process_index:04d}',
                        f'{curr_train_iter:04d}',
                        f'{sample_idx:04d}.ply',
                    )
                )
            )
        )

        process_mesh_gen_animate(
            generated_meshes[sample_idx],
            os.path.join(
                storage_dir, 
                '_'.join(
                    (   
                        f'{curr_epoch}',
                        f'{accelerator.process_index:04d}',
                        f'{curr_train_iter:04d}',
                        f'{sample_idx:04d}.gif',
                    )
                )
            ),
            batch_data={
                'gt_vertices': batch_data_label['origin_gt_vertices'][sample_idx],
                'point_cond': batch_data_label['point_cond'][sample_idx],
            },
            max_frames=100,
            # batch_data_label
        )

    accelerator.wait_for_everyone()
    return 
    

import time
@torch.no_grad()
def evaluate_sample(
    args,
    curr_epoch,
    model,
    accelerator,
    dataset_loader,
    logger,
    curr_train_iter=-1,
    sampling_method='sampling_debug'
):
    
    model.eval()
    net_device = next(model.parameters()).device
    num_batches = len(dataset_loader)
    
    ### parse evaluation status
    if hasattr(dataset_loader.dataset, "dataset_name"):
        dataset_name = dataset_loader.dataset.dataset_name
    else:
        dataset_name = "default"
    task_name_prefix = dataset_name + '_'

    time_delta = SmoothedValue(window_size=10)
    
    accelerator.wait_for_everyone()
    
    ### do sampling every evaluation
    set_seed(accelerator.process_index)
    
    # create storage directory
    storage_dir = os.path.join(args.checkpoint_dir, sampling_method, f'epoch{curr_epoch}')
    if accelerator.is_main_process:
        os.makedirs(storage_dir, exist_ok = True)
    accelerator.wait_for_everyone()
    
    # just sample one round for checking training status
    round_time = 2
    if accelerator.is_main_process:
        logger.info(f'sampling {round_time} rounds')

    start = time.time()

    for round_idx in range(round_time):
        
        
        # logger.log_messages(f'sampling {round_time} rounds')
        outputs = model(
            data_dict=dict(),
            is_eval=True, 
            is_generate=True,
            num_return_sequences=args.batchsize_per_gpu,
        )

        if accelerator.is_main_process:
            time_cost = time.time() - start
            start = time.time()
            logger.log_messages(f'sampling time cost: {time_cost}')
        # print(outputs)
    
        generated_meshes = outputs["recon_faces"]   # nsample x nf x 3 x 3
    
        for sample_idx in range(args.batchsize_per_gpu):
            # store the generated meshes
            post_process_mesh(
                generated_meshes[sample_idx],
                os.path.join(
                    storage_dir, 
                    '_'.join(
                        (   
                            f'{curr_epoch}',
                            f'{accelerator.process_index:04d}',
                            f'{round_idx:04d}',
                            f'{sample_idx:04d}.ply',
                        )
                    )
                )
            )

            process_mesh_gen_animate(
                generated_meshes[sample_idx],
                os.path.join(
                    storage_dir, 
                    '_'.join(
                        (   
                            f'{curr_epoch}',
                            f'{accelerator.process_index:04d}',
                            f'{round_idx:04d}',
                            f'{sample_idx:04d}.gif',
                        )
                    )
                )
            )
    
    accelerator.wait_for_everyone()
    
    return {}, None
