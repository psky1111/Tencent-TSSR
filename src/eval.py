import copy
import dataclasses
import json
import logging
import os
import torch
import trimesh
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import ProjectConfiguration, InitProcessGroupKwargs
from accelerate.logging import get_logger
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from torch.utils.tensorboard import SummaryWriter

from src.utils.config import load_config, parse_args, get_output_dir
from src.utils.common import import_module_or_data, pack_code
from src.custom_datasets.flexi_dataset.mesh_features import MeshMeta
from tools.eval_utils.perplexity import post_process_mesh

class TrainModuleCollection(torch.nn.Module):
    def __init__(self, point_encoder=None, meshtron_net=None):
        super().__init__()
        self.point_encoder = point_encoder
        self.meshtron_net = meshtron_net

logger = get_logger(__name__, log_level='INFO')


@dataclasses.dataclass
class EvalTask:
    eval_metas: List[MeshMeta]
    eval_id: str
    global_task_idx: int = 0
    complexity: int = 0


def main():
    args = copy.deepcopy(parse_args())
    # 初始化配置和handler
    eval_config = load_config(args.eval_config)
    output_dir = get_output_dir(eval_config.output_dir, eval_config)
    if args.debug_mode:
        output_dir = output_dir + "_debug"
    print(f"output dir {output_dir}")

    # 设置accelerator
    pg_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=3600 * 24))
    accelerator_project_config = ProjectConfiguration(project_dir=output_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=None,
        project_config=accelerator_project_config,
        kwargs_handlers=[pg_kwargs],
        dataloader_config=DataLoaderConfiguration(dispatch_batches=args.dispatch_batches)
    )
    
    # 确保多进程的 create_time 一致
    create_time_list = [datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S')]
    #torch.distributed.broadcast_object_list(create_time_list, src=0)
    create_time = create_time_list[0]

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    # 设置tensorboard路径
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        #pack_code(output_dir)

    upload_all_results = eval_config.get("upload_all_results", False)  # 决定是否每个进程都保存结果
    board_log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(board_log_dir, exist_ok=True)
    writer = SummaryWriter(board_log_dir) if accelerator.is_main_process or upload_all_results else None

    # 设置浮点数精度
    weight_dtype = torch.bfloat16
    if accelerator.mixed_precision == 'fp16':
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == 'bf16':
        weight_dtype = torch.bfloat16

    # 目前只做一阶段的评估，后续有需求再支持多阶段 pipeline 评估
    assert len(eval_config.eval_models) == 1
    model_name = list(eval_config.eval_models.keys())[0]
    config = eval_config.eval_models[model_name]
    checkpoint_dirs = [os.path.join(config.model_save_dir, name) for name in config.checkpoint_names]
    # 逐 checkpoint 评估
    for checkpoint_dir in checkpoint_dirs:
        eval_impl(eval_config, accelerator, weight_dtype, output_dir, args, writer, checkpoint_dir, create_time)


def eval_impl(eval_config, accelerator, weight_dtype, output_dir, args, writer, checkpoint_dir, create_time):
    model_name = list(eval_config.eval_models.keys())[0]
    config = eval_config.eval_models[model_name]
    
    # 将需要评估的 checkpoint_dir 更新到 config 上
    model_config = copy.deepcopy(load_config(config.model_config))
    model_config.model.pretrained_model_path = checkpoint_dir
    checkpoint_name = os.path.basename(checkpoint_dir)
    dataset_config = load_config(model_config.data.dataset_config)
    
    logger.info(f"model config: {model_config}")
    logger.info(f"dataset config: {dataset_config}")
    
    # 加载模型，构建推理 pipeline
    model_handler_class = import_module_or_data(model_config.model_handler)
    model_handler = model_handler_class(
        model_config=model_config,
        dataset_config=dataset_config,
        accelerator=accelerator,
        logger=logger,
        weight_dtype=weight_dtype,
        output_dir=output_dir,
        debug_mode=args.debug_mode
    )
    model_handler.build_modules()
    model_handler.set_module_performance()
    model_handler.set_module_attributes(TrainModuleCollection(), eval_mode=True)
    pipeline = model_handler.init_pipeline()

    # 准备评估 dataset
    task_list = []
    eval_dataset_dict = {}
    for eval_sample_info in eval_config.detailed_eval_samples:
        # 分品类构造对应的 dataset
        item_type = list(eval_sample_info.keys())[0]
        copied_dataset_config = copy.deepcopy(dataset_config)
        copied_dataset_config.eval_sample_info_list = [eval_sample_info]
        test_dataset = model_handler.build_dataset(mode="test", dataset_config=copied_dataset_config)
        eval_dataset_dict[item_type] = test_dataset
        logger.info(f"item_type is {item_type}, total {len(test_dataset.all_mesh_meta)} eval objects")

        # 构建评估任务集合，准备分发给不同进程
        infer_bsz = config.infer_config.batch_size
        for idx in range(0, len(test_dataset.all_mesh_meta), infer_bsz):
            eval_metas = test_dataset.all_mesh_meta[idx: idx + infer_bsz]
            
            # 以 gt 面数作为任务复杂度
            complexity = 0
            for meta in eval_metas:
                gt_face_cnt = meta.face_info['face_cnt']
                complexity += gt_face_cnt
            
            for seed in eval_config.get("seeds", [123]):
                task_list.append(EvalTask(
                    eval_metas=eval_metas,
                    eval_id=(model_name, checkpoint_name, item_type, seed),
                    global_task_idx=idx,
                    complexity=complexity,
                ))
    
    # accelerator.split_between_processes 的切分逻辑为简单顺序切分，需要保证所有进程的的 input 序列一致，这里
    # 提前按照任务复杂度排序，保证负载均衡
    sorted_task_list = sorted(task_list, key=lambda x: x.complexity)
    all_process_tasks = {process_idx: [] for process_idx in range(accelerator.num_processes)}
    for i, task in enumerate(sorted_task_list):
        process_idx = i % accelerator.num_processes
        all_process_tasks[process_idx].append(task)
    
    new_task_list = []
    for process_idx in range(accelerator.num_processes):
        new_task_list.extend(all_process_tasks[process_idx])
    task_list = new_task_list
    
    logger.info(f"total {len(task_list)} eval tasks", main_process_only=False)

    accelerator.wait_for_everyone()
    
    local_results_and_metrics = {}
    face_buckets=((0, 2000), (2000, 4000), (4000, 8000), (8000, 16000), (16000, 32000), (32000, 64000))
    with accelerator.split_between_processes(task_list, apply_padding=False) as local_task_list:
        logger.info(f"process idx: {accelerator.process_index}, num tasks: {len(local_task_list)}", in_order=True, main_process_only=False)
        
        for idx, task in enumerate(local_task_list):
            # 使用对应 dataset 获取 batch 数据
            model_name, checkpoint_name, item_type, seed = task.eval_id
            current_dataset = eval_dataset_dict[item_type]
            batch_data, sample_info = current_dataset.random_sample(seed=seed, mesh_metas=task.eval_metas)
            # 进行 pipeline 推理，获得结果
            generator = torch.Generator(device=pipeline.device).manual_seed(seed)
            result_dict, metric_dict = model_handler.predict_sample(
                batch_data=batch_data, 
                pipeline=pipeline, 
                generator=generator,
                **config.infer_config.pipeline_params,
            )
            result_dict['sample_info'] = sample_info
            local_results_and_metrics.setdefault(task.eval_id, []).append((result_dict, metric_dict))
            # 如果当前进程存在 writer, 则保存结果
            if writer is not None:
                for result_name, result in result_dict.items():
                    for sample_idx, sample_meta in enumerate(task.eval_metas):
                        # 按品类与面数建立文件夹，保存 sample 结果
                        category = sample_meta.category if sample_meta.category is not None else 'unknown'
                        face_num = sample_meta.face_info['face_cnt']
                        bucket_name = 'unknown_faces'
                        for bucket in face_buckets:
                            if bucket[0] <= face_num and face_num < bucket[1]:
                                bucket_name = f"{bucket[0] // 1000}k-{bucket[1] // 1000}k_faces"
                                break

                        global_sample_idx = task.global_task_idx + sample_idx
                        save_dir = os.path.join(writer.log_dir, create_time, item_type, category, bucket_name, f"sample_{global_sample_idx}", 
                                                f"{model_name}_step_{checkpoint_name}", f"seed_{seed}")
                        os.makedirs(save_dir, exist_ok=True)
                        
                        # 生成结果路径下保存 mesh 的原始品类、指标等信息
                        if result_name == 'sample_info':
                            curr_sample_info = result[sample_idx]
                            for metric_name, value_list in metric_dict.items():
                                curr_sample_info[metric_name] = value_list[sample_idx]
                            with open(os.path.join(save_dir, 'sample_info.json'), 'w') as f:
                                json.dump(curr_sample_info, f, indent=4, ensure_ascii=False)
                            continue
                        
                        if result['visualize'] == 'mesh':
                            post_process_mesh(result['data'][sample_idx], filename=os.path.join(save_dir, f"{result_name}.ply"))
                        elif result['visualize'] == 'point_cloud':
                            pointscloud = trimesh.points.PointCloud(vertices=result['data'][sample_idx])
                            pointscloud.export(os.path.join(save_dir, f"{result_name}.ply"))
            logger.info(f"rank: {accelerator.process_index}, finished {idx + 1}/{len(local_task_list)} tasks", main_process_only=False)

        # 保存当前进程结果到本地
        torch.save(local_results_and_metrics, os.path.join(output_dir, f"all_res-{accelerator.process_index}.pt"))
        logger.info(f"rank: {accelerator.process_index}, finished all tasks", main_process_only=False)
        accelerator.wait_for_everyone()
        
        # 对所有进程的指标结果进行汇总
        if accelerator.is_main_process:
            all_results_from_all_processes = {}
            for idx in range(accelerator.num_processes):
                data = torch.load(os.path.join(output_dir, f"all_res-{idx}.pt"), weights_only=False)
                for eval_id, results_list in data.items():
                    all_results_from_all_processes.setdefault(eval_id, []).extend(results_list)

            for eval_id, results_list in all_results_from_all_processes.items():
                logger.info(f"eval_id is {eval_id}")
                final_metric_res = {}
                for record in results_list:
                    metric: Dict[str, List[torch.Tensor]] = record[1]
                    for metric_name in metric:
                        final_metric_res.setdefault(metric_name, []).extend(metric[metric_name])

                # 获得某个 (checkpoint_name, item_type, seed) 的指标结果集合后，计算该类评估任务的平均指标结果
                model_name, checkpoint_name, item_type, seed = eval_id
                group_key = ",".join([str(c) for c in eval_id])
                checkpoint_step = int(checkpoint_name.split("-")[-1])

                hparams = {
                    'model_name': model_name,
                    'checkpoint_name': checkpoint_name,
                    'item_type': item_type,
                    'seed': str(seed),
                }
                metric_values = {}
                for metric_name, metric_value in final_metric_res.items():
                    avg_metric = sum(metric_value) / len(metric_value)
                    logger.info(f"{metric_name}: {avg_metric}, len {len(metric_value)}")
                    writer.add_scalar(f"{group_key}/{metric_name}", avg_metric, global_step=checkpoint_step)
                    metric_values.update({metric_name: avg_metric})
                writer.add_hparams(hparams, metric_values, run_name=group_key, global_step=checkpoint_step)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()
