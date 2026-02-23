import json
import os
import torch
import trimesh
from typing import Any, Dict, List

from src.custom_pipelines.meshtron_pipeline import MeshtronPipeline
from src.model_handlers.base_handler import BaseHandler
from src.utils.common import (
    import_module_or_data,
    calc_index_range,
    save_nll_curve,
    CROSS_ENTROPY_IGNORE_IDX,
    get_num_params,
)
from tools.eval_utils.perplexity import post_process_mesh


class MeshtronHandlerV3(BaseHandler):
    def build_modules(self):
        self.tokenizer = self.build_hf_model(model_name='tokenizer', specific_model_conf=self.config.tokenizer)
      
        self.meshtron_net = self.build_hf_model(model_name='meshtron_net', 
                                                specific_model_conf=self.config.model.meshtron_net,
                                                vocab_size=self.tokenizer.vocab_size,
                                                n_sliding_triangles=self.tokenizer.n_sliding_triangles,
                                                )
        self.point_encoder = self.build_hf_model(model_name='point_encoder', 
                                                 specific_model_conf=self.config.model.point_encoder)
        self.meshtron_net.scheduler = self.meshtron_net.scheduler.to(device=self.device)
 
        meshtron_param_count = get_num_params(self.meshtron_net)
        self.logger.info(f"meshtron parameters size: {meshtron_param_count}")

    def set_module_performance(self):
        super().set_module_performance()
        
        if self.config.trainer.gradient_checkpointing:
            self.point_encoder.enable_gradient_checkpointing()
            self.meshtron_net.enable_gradient_checkpointing()
            self.logger.info('enable checkpointing')

    def build_dataset(self, mode='train', dataset_config=None):
        assert mode in ["train", "val", "test"]
        dataset_config = self.dataset_config if dataset_config is None else dataset_config
        
        dataset_class = import_module_or_data(dataset_config.dataset_class)
        return dataset_class(
            dataset_args=dataset_config,
            project_args=self.config,
            debug_mode=self.debug_mode,
            mode=mode,
            batch_size=self.config.trainer.train_batch_size,
            device=self.device,
            dtype=self.dtype,
            deepmesh=False,
            rank=self.accelerator.process_index,
            world_size=self.accelerator.num_processes,
        )
    
    def build_dataloader(self, mode='train'):
        collate_fn = import_module_or_data(self.dataset_config.collate_fn)
        dataset = self.build_dataset(mode)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=True,
            collate_fn=collate_fn,
            batch_size=1,  
            num_workers=self.config.data.dataloader_num_workers,
            prefetch_factor=None if self.config.data.dataloader_num_workers == 0 else 16,
            drop_last=True,
        )
        return dataset, dataloader
    
    def prepare_dataloader(self):
        self.train_dataset, self.train_dataloader = self.build_dataloader('train')
        self.val_dataset, self.val_dataloader = self.build_dataloader('val')
        return self.train_dataloader
    
    def get_scaling_factor(self, t, method='clamped_radd'):
        if method == 'exponential':
            return torch.exp(1 - t)
        elif method == 'clamped_radd':
            t_safe = torch.clamp(t, 1e-3, 1-1e-3)
            return torch.clamp((t_safe)/(1-t_safe),min=1.0, max=2.0)
        else:
            raise ValueError(f"Unknown method: {method}")
        
    def prepare_for_training_step(self, batch_data) -> Dict[str, Any]:
        pass
    
    def forward_and_compute_loss(self, trainable_modules, point_cond, input_ids, face_count_id,t,connect_map, **kwargs):
        pass

    def get_trainable_parameters(self, trainable_modules):
        pass
    
    def save_pretrained(self, save_path, trainable_modules):
        pass
    
    def init_pipeline(self, trainable_modules=None):
        meshtron_net = self.get_active_model('meshtron_net', trainable_modules, unwrap_model=True).eval()
        point_encoder = self.get_active_model('point_encoder', trainable_modules, unwrap_model=True).eval()
        
        pipeline = MeshtronPipeline(
            tokenizer=self.tokenizer, 
            meshtron_net=meshtron_net, 
            point_encoder=point_encoder,
            device=self.device,
            dtype=self.dtype,
        )
        return pipeline
    
    def predict_sample(self, batch_data, pipeline, generator=None, **kwargs):
        new_batch_data = {}
        for k, v in batch_data.items():

            if torch.is_floating_point(v) and k == 'point_cond':
                new_batch_data[k] = v.to(device=self.device, dtype=self.dtype)
            elif isinstance(v, torch.Tensor):
                new_batch_data[k] = v.to(device=self.device)
            else:
                new_batch_data[k] = v
        batch_data = new_batch_data
        
        result_dict, metric_dict = pipeline(
            point_cond=batch_data['point_cond'],
            face_count_id=batch_data['face_count_id'].long() if self.config.model.meshtron_net.use_face_count_id else None,
            gt_vertices=batch_data['gt_vertices'],
            generator=generator,
            batch_data=batch_data,
            **kwargs
        )
        return result_dict, metric_dict

    def run_validation(self, trainable_modules, writer, global_step):
        if self.config.validation.skip_validation:
            return
        self.logger.info('start validation', main_process_only=False)
        trainable_modules.eval()
        pipeline = self.init_pipeline(trainable_modules)
        
        validation_info = {}
        for idx in range(self.config.validation.fix_train_num):
            validation_info[f"fix_train_sample-{idx}"] = {'source': 'train_dataset', 'seed': idx}
        for idx in range(self.config.validation.fix_val_num):
            validation_info[f"fix_val_sample-{idx}"] = {'source': 'val_dataset', 'seed': idx}
        for idx in range(self.config.validation.random_val_num):
            validation_info[f"random_val_sample-{idx}"] = {'source': 'val_dataset', 'seed': global_step + idx}

        all_validation_tasks = list(validation_info.keys())
        start, end = calc_index_range(length=len(all_validation_tasks), rank=self.accelerator.process_index,
                                      world_size=self.accelerator.num_processes)
        curr_tasks = all_validation_tasks[start: end]

        results_and_metrics = {}
        for task_name in curr_tasks:
            sample_info = validation_info[task_name]
            batch_data, sample_info = getattr(self, sample_info['source']).random_sample(sample_info['seed'],
                                                                                         batch_size=self.config.validation.batch_size)
            generator = None
            if self.config.validation.seed:
                generator = torch.Generator(device=self.accelerator.device).manual_seed(self.config.validation.seed)
            result_dict, metric_dict = self.predict_sample(
                batch_data=batch_data, 
                pipeline=pipeline, 
                generator=generator,
                **self.config.validation.pipeline_params,
            )
            result_dict['sample_info'] = sample_info
            results_and_metrics.update({task_name: (result_dict, metric_dict)})

        torch.save(results_and_metrics, os.path.join(self.output_dir, f"results-{self.accelerator.process_index}.pt"))
        self.accelerator.wait_for_everyone()
        

        if self.accelerator.is_main_process:
            all_results = {}
            for idx in range(self.accelerator.num_processes):
                load_res = torch.load(os.path.join(self.output_dir, f"results-{idx}.pt"), map_location=torch.device('cpu'), weights_only=False)
                for task_name, res in load_res.items():
                    all_results.update({task_name: res})

            metrics_info = {}
            for idx, (task_name, res) in enumerate(all_results.items()):
             
                result_dict, metric_dict = res
                sample_info: List[dict] = result_dict['sample_info']

                for metric_name, value_list in metric_dict.items():
    
                    metrics_info.setdefault(metric_name, {}).setdefault(f"{task_name.split('-')[0]}", []).extend(value_list)
   
                    for i, value in enumerate(value_list):
                        sample_info[i][metric_name] = value

      
                for result_name, result in result_dict.items():
                    for sample_idx in range(self.config.validation.batch_size):
                        save_dir = os.path.join(writer.log_dir, self.create_time, f"step_{global_step}", task_name, f"sample_{sample_idx}")
                        os.makedirs(save_dir, exist_ok=True)
                        
                        if result_name == 'sample_info':
                            curr_sample_info = result[sample_idx]
                            with open(os.path.join(save_dir, 'sample_info.json'), 'w') as f:
                                json.dump(curr_sample_info, f, indent=4, ensure_ascii=False)
                            continue
                        
                        if result['visualize'] == 'mesh':
                            post_process_mesh(result['data'][sample_idx], filename=os.path.join(save_dir, f"{result_name}.ply"))
                        elif result['visualize'] == 'point_cloud':
                            pointscloud = trimesh.points.PointCloud(vertices=result['data'][sample_idx])
                            pointscloud.export(os.path.join(save_dir, f"{result_name}.ply"))
                        elif result['visualize'] == 'curve':
                            curve_data = result['data'][sample_idx].cpu().float().numpy()
                            save_nll_curve(curve_data=curve_data, save_path=os.path.join(save_dir, f"{result_name}.png"))
                            
                        if result_name == 'beam_meshes':
                            save_dir = os.path.join(save_dir, 'beam_meshes')
                            os.makedirs(save_dir, exist_ok=True)
                            for beam_idx in range(result['data'].shape[0]):
                                post_process_mesh(result['data'][beam_idx], filename=os.path.join(save_dir, f"beam_mesh_{beam_idx}.ply"))            

            for metric_name, task_metric in metrics_info.items():
                for task_type, value_list in task_metric.items():
                    metric_value = sum(value_list) / (len(value_list) + 1)
                    writer.add_scalar(f"val_metric/{task_type}_{metric_name}", metric_value, global_step=global_step)
                    self.logger.info(f"val_metric/{task_type}_{metric_name}: {metric_value}, cnt is {len(value_list)}")

        trainable_modules.train()
        self.logger.info('finish validation', main_process_only=False)
