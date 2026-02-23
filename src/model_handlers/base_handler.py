import os
import torch
from datetime import datetime
from omegaconf import ListConfig
from typing import Any, Dict, Tuple

from src.utils.common import import_module_or_data


class BaseHandler:
    def __init__(self, model_config, dataset_config, accelerator, logger, weight_dtype, output_dir, debug_mode=False):
        self.config = model_config
        self.dataset_config = dataset_config
        self.accelerator = accelerator
        self.logger = logger
        self.dtype = weight_dtype
        self.debug_mode = debug_mode
        self.device = accelerator.device
        self.output_dir = output_dir
        self.create_time = datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S')
        self.pretrained_model_path = self.config.model.pretrained_model_path
        
    def build_hf_model(self, model_name, specific_model_conf, **params_kwargs):
        model_cls = import_module_or_data(specific_model_conf.target)
        model_params = dict(specific_model_conf.get('params', {}))
        model_params.update(params_kwargs)
        for k, v in model_params.items():
            if isinstance(v, ListConfig):
                model_params[k] = list(v)
        
        pretrained_model_path = None
        if self.pretrained_model_path is not None \
                and os.path.exists(os.path.join(self.pretrained_model_path, specific_model_conf.subfolder)):
            pretrained_model_path = os.path.join(self.pretrained_model_path, specific_model_conf.subfolder)
        elif specific_model_conf.get('pretrained_path', None) and os.path.exists(specific_model_conf.pretrained_path):
            pretrained_model_path = specific_model_conf.pretrained_path
        
        if pretrained_model_path is not None:
            self.logger.info(f"load {model_name} from {pretrained_model_path}")
            model, loading_info = model_cls.from_pretrained(
                pretrained_model_path,
                low_cpu_mem_usage=None,
                output_loading_info=True,
                device_map=None,
                **model_params
            )
            self.logger.info(f"loading_info of {model_name} is {loading_info}")
        else:
            self.logger.info(f"initialize {model_name} from scratch")
            model = model_cls(**model_params)
        return model

    def set_module_performance(self):
        # Enable TF32 for faster training on Ampere GPUs,
        # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        if self.config.trainer.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

    def set_module_attributes(self, trainable_modules, eval_mode=False) -> Dict[str, Any]:
        class_vars = vars(trainable_modules)
        for k, v in self.config.model.items():
            if k in class_vars:
                if getattr(v, "trainable", False) and not eval_mode:
                    self.logger.info(f"setting {k} to trainable")
                    setattr(trainable_modules, k, getattr(self, k))
                else:
                    self.logger.info(f"setting {k} to untrainable")
                    getattr(self, k).requires_grad_(False)
                    getattr(self, k).to(self.accelerator.device, dtype=self.dtype)
        return trainable_modules

    def get_active_model(self, model_name, trainable_modules, unwrap_model=False):
        if model_name not in self.config.model or trainable_modules is None:
            return getattr(self, model_name)
        
        model_attr = getattr(self.config.model, model_name)
        if getattr(model_attr, "trainable", False):
            if not unwrap_model:
                return trainable_modules.get_submodule(model_name)
            else:
                return self.accelerator.unwrap_model(trainable_modules.get_submodule(model_name))
        else:
            return getattr(self, model_name)
    
    def prepare_dataloader(self):

        raise NotImplementedError

    def prepare_for_training_step(self, batch) -> Dict[str, Any]:

        raise NotImplementedError

    def forward_and_compute_loss(self, trainable_modules, **model_inputs):

        raise NotImplementedError

    def run_validation(self, trainable_modules, writer, global_step):
        pass

    def get_trainable_parameters(self, trainable_modules):

        return trainable_modules.parameters()

    def log_train_images(self, writer, global_step, **kwargs):
        pass

    def save_pretrained(self, save_path, trainable_modules):
        # models = self.get_active_models(trainable_modules, unwrap_model=True)
        for k, v in self.config.model.items():
            if getattr(v, "save_model", False):
                self.logger.info(f"saving {k}")
                model = self.get_active_model(k, trainable_modules, unwrap_model=True)
                subfolder = getattr(v, "subfolder", k)
                model.save_pretrained(os.path.join(save_path, subfolder), safe_serialization=True)
