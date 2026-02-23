import argparse
import contextlib
import os
import re
import shutil
import time
import torch
import zipfile
import logging
import multiprocessing
import importlib
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from datetime import datetime
from jaxtyping import Float, Int
from PIL import Image


CROSS_ENTROPY_IGNORE_IDX = -100


class AttributeDict(dict):
    def __getattr__(self, item):
        return self[item]

    def __setattr__(self, key, value):
        self[key] = value

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)


def init_logger():
    logger = multiprocessing.get_logger()
    if len(logger.handlers) < 1:
        logger.setLevel('INFO')
        formatter = logging.Formatter(fmt="%(process)d %(asctime)s %(filename)s:%(lineno)d [%(levelname)s] - %(message)s")
        console = logging.StreamHandler()
        console.setLevel('INFO')
        console.setFormatter(formatter)
        logger.addHandler(console)
    return logger


logger = init_logger()


def calc_index_range(length, rank, world_size):
    size_per_rank = [length // world_size for _ in range(world_size)]
    for i in range(length % world_size):
        size_per_rank[i] += 1
    size_per_rank_prefix_sum = [0]
    for i in range(world_size):
        size_per_rank_prefix_sum.append(size_per_rank_prefix_sum[-1] + size_per_rank[i])
    
    start = size_per_rank_prefix_sum[rank]
    end = size_per_rank_prefix_sum[rank + 1]
    return start, end
 
def create_instance(class_name, *args, **kwargs):  
    cls = globals().get(class_name)  
    if cls:  
        return cls(*args, **kwargs)  
    else:  
        raise ValueError(f"No class named {class_name}")  
        
def import_module_or_data(import_path):
    try:
        maybe_module, maybe_data_name = import_path.rsplit(".", 1)
        return getattr(importlib.import_module(maybe_module), maybe_data_name)
    except Exception as e:
        print('Cannot import data from the module path, error {}'.format(str(e)))


def clear_early_checkpoints(logger, output_dir, checkpoints_total_limit=2):
    checkpoints = os.listdir(output_dir)
    checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

    # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
    if len(checkpoints) >= checkpoints_total_limit:
        num_to_remove = len(checkpoints) - checkpoints_total_limit + 1
        removing_checkpoints = checkpoints[0:num_to_remove]

        logger.info(
            f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
        )
        logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

        for removing_checkpoint in removing_checkpoints:
            removing_checkpoint = os.path.join(output_dir, removing_checkpoint, "pytorch_model")
            if os.path.exists(removing_checkpoint):
                shutil.rmtree(removing_checkpoint)


def str2bool(v):
    if v is None:
        return v
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def zip_folder(folder_dir, output_path, skip_patterns=('.git', '.gitignore', '__pycache__'), 
               expect_folders=(r'^conf/', r'^src/', r'^tools/', r'^\.'), expect_files=(r'^start.*\.sh',)):
    def check_match(source, pattern):
        match = re.search(pattern, source)
        return match is not None
    
    
    pack_name = os.path.basename(output_path).replace('.zip', '')
    
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_dir):
            if any([pattern in root for pattern in skip_patterns]):
                continue
            
            relative_dirname = os.path.relpath(root, folder_dir)
            is_expect_folders = any([check_match(relative_dirname, pattern) for pattern in expect_folders])
            if not is_expect_folders:
                continue
            
            for file_name in files:

                if folder_dir == root and not any([check_match(file_name, pattern) for pattern in expect_files]):
                    continue
                
                file_path = os.path.join(root, file_name)
                arcname = os.path.join(pack_name, os.path.relpath(file_path, folder_dir))
                zipf.write(file_path, arcname)
                

def pack_code(output_dir):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    zip_name = f"run_code_{timestamp}.zip"
    output_path = os.path.join(output_dir, 'run_codes', zip_name)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    project_dir = os.getcwd()
    zip_folder(folder_dir=project_dir, output_path=output_path)
    

def pytorch_worker_info():
    worker = 0
    num_workers = 1
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        worker = worker_info.id
        num_workers = worker_info.num_workers
    return worker, num_workers


def open_image_safely(image_path, color='white', alpha_threshold=100):
    if isinstance(image_path, str):
        img = Image.open(image_path).convert("RGBA")
    else:
        img = image_path.convert("RGBA")
    background = Image.new("RGBA", img.size, color=color)
    if alpha_threshold is None:
        paste_mask = img
    else:
        paste_mask = Image.fromarray(np.array(img)[:, :, 3] > alpha_threshold)
    background.paste(img, (0, 0), paste_mask)
    # Image. paste(im, box, mask)
    if not background.mode == "RGB":
        background = background.convert("RGB")
    img = background
    return img


def exists(val):
    return val is not None


# top k filtering
def top_k(logits, thres = 0.9):
    k = int((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs


def save_nll_curve(curve_data, save_path, max_display_nll=12):

    curve_data = curve_data[curve_data != -1]
    x = np.arange(len(curve_data))
    
    dpi = 360
    width = 2048
    height = 1536
    fig_width = width / dpi  # in inches
    fig_height = height / dpi  # in inches
    # Create a figure with specified size and DPI
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)

    plt.plot(x, curve_data)
    ticks = list(range(0, len(x), 512 * 9)) + [x[-1]]
    labels = [i // 9 for i in ticks]
    plt.xticks(ticks, labels)
    plt.ylim((-0.5, max_display_nll + 0.5))
    plt.xlabel('Face position')
    plt.ylabel('Negative log-likehood')
    fig.savefig(save_path)


def focal_loss_fn(logits , targets, gamma=2.0, ignore_index=-100):
    log_softmax = torch.clamp(F.log_softmax(logits, dim=-1), min=-100, max=100)
    if gamma != 0:
        ce_loss = F.nll_loss(input=log_softmax, target=targets, reduction='none', ignore_index=ignore_index)
        prob_all = F.softmax(logits, dim=-1)
        expand_index = targets.unsqueeze(-1)
        prob_target = torch.gather(input=prob_all, dim=1, index=expand_index).squeeze(-1)
        focal_loss = ((1 - prob_target) ** gamma) * ce_loss
        
        valid_mask = (targets != ignore_index).float()
        loss = torch.sum(focal_loss * valid_mask) / torch.sum(valid_mask)
        return loss
    else:
        ce_loss = F.nll_loss(input=log_softmax, target=targets, reduction='mean', ignore_index=ignore_index)
        return ce_loss


@contextlib.contextmanager
def maybe_enable_profiling(enable_profiling, trace_dir):
    if enable_profiling:
        rank = torch.distributed.get_rank()

        def trace_handler(prof):
            curr_trace_dir_name = "iteration_" + str(prof.step_num)
            curr_trace_dir = os.path.join(trace_dir, curr_trace_dir_name)
            if not os.path.exists(curr_trace_dir):
                os.makedirs(curr_trace_dir, exist_ok=True)

            logger.info(f"Dumping profiler traces at step {prof.step_num}")
            begin = time.monotonic()
            prof.export_chrome_trace(f"{curr_trace_dir}/rank{rank}_trace.json")
            logger.info(
                f"Finished dumping profiler traces in {time.monotonic() - begin:.2f} seconds"
            )

        logger.info(f"Profiling active. Traces will be saved at {trace_dir}")

        if not os.path.exists(trace_dir):
            os.makedirs(trace_dir, exist_ok=True)

        warmup, active = 100, 10
        wait = 1000
        assert (
            wait >= 0
        ), "profile_freq must be greater than or equal to warmup + active"
        gpu_device_profiled = None
        if torch.cuda.is_available():
            gpu_device_profiled = torch.profiler.ProfilerActivity.CUDA
        elif torch.xpu.is_available():
            gpu_device_profiled = torch.profiler.ProfilerActivity.XPU
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                gpu_device_profiled,
            ],
            schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, skip_first_wait=1),
            on_trace_ready=trace_handler,
            record_shapes=True,
        ) as torch_profiler:
            yield torch_profiler
    else:
        torch_profiler = contextlib.nullcontext()
        yield None


def get_num_params(model: torch.nn.Module, exclude_embedding: bool = False) -> int:
    num_params = sum(p.numel() for p in model.parameters())
    if exclude_embedding:
        num_params -= sum(
            sum(p.numel() for p in m.parameters())
            for m in model.children()
            if isinstance(m, torch.nn.Embedding)
        )
    return num_params
