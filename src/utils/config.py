import argparse
import os
import re
from dataclasses import dataclass, field
from datetime import datetime

from omegaconf import OmegaConf
from typing import Any, Dict, Optional, Union, Tuple
from omegaconf import DictConfig

from src.utils.common import str2bool

# ============ Register OmegaConf Recolvers ============= #
OmegaConf.register_new_resolver(
    "calc_exp_lr_decay_rate", lambda factor, n: factor ** (1.0 / n)
)
OmegaConf.register_new_resolver("add", lambda a, b: a + b)
OmegaConf.register_new_resolver("sub", lambda a, b: a - b)
OmegaConf.register_new_resolver("mul", lambda a, b: a * b)
OmegaConf.register_new_resolver("div", lambda a, b: a / b)
OmegaConf.register_new_resolver("idiv", lambda a, b: a // b)
OmegaConf.register_new_resolver("basename", lambda p: os.path.basename(p))
OmegaConf.register_new_resolver("rmspace", lambda s, sub: s.replace(" ", sub))
OmegaConf.register_new_resolver("tuple2", lambda s: [float(s), float(s)])
OmegaConf.register_new_resolver("gt0", lambda s: s > 0)
OmegaConf.register_new_resolver("cmaxgt0", lambda s: C_max(s) > 0)
OmegaConf.register_new_resolver("not", lambda s: not s)
OmegaConf.register_new_resolver(
    "cmaxgt0orcmaxgt0", lambda a, b: C_max(a) > 0 or C_max(b) > 0
)
# ======================================================= #


def C_max(value: Any) -> float:
    if isinstance(value, int) or isinstance(value, float):
        pass
    else:
        value = config_to_primitive(value)
        if not isinstance(value, list):
            raise TypeError("Scalar specification only supports list, got", type(value))
        if len(value) >= 6:
            max_value = value[2]
            for i in range(4, len(value), 2):
                max_value = max(max_value, value[i])
            value = [value[0], value[1], max_value, value[3]]
        if len(value) == 3:
            value = [0] + value
        assert len(value) == 4
        start_step, start_value, end_value, end_step = value
        value = max(start_value, end_value)
    return value


def load_config(*yamls: str, cli_args: list = [], from_string=False, **kwargs) -> Any:
    if from_string:
        yaml_confs = [OmegaConf.create(s) for s in yamls]
    else:
        yaml_confs = [OmegaConf.load(f) for f in yamls]
    cli_conf = OmegaConf.from_cli(cli_args)
    cfg = OmegaConf.merge(*yaml_confs, cli_conf, kwargs)
    OmegaConf.resolve(cfg)
    assert isinstance(cfg, DictConfig)
    return cfg


def update_config(cfg, merge_config):
    for k, v in merge_config.items():
        OmegaConf.update(cfg, k, v, merge=True)

def config_to_primitive(config, resolve: bool = True) -> Any:
    return OmegaConf.to_container(config, resolve=resolve)


def dump_config(path: str, config) -> None:
    with open(path, "w") as fp:
        OmegaConf.save(config=config, f=fp)


def parse_structured(fields: Any, cfg: Optional[Union[dict, DictConfig]] = None) -> Any:
    scfg = OmegaConf.structured(fields(**cfg))
    return scfg


def get_value(nested_dict, keys):
    for key in keys:
        try:
            if key.isdigit():
                key = int(key)
            nested_dict = nested_dict[key]
        except:
            return None
    return nested_dict


def get_output_dir(input_str, model_config, pattern=r"\{(.*?)\}"):
    def replace(match):
        keys = match.group(1).split('.')
        value = get_value(model_config, keys)
        return str(value) if value is not None else match.group(1)
    result = re.sub(pattern, replace, input_str)
    return result


if __name__ == '__main__':
    config = load_config("conf/model_configs/part_segmentation.yaml")
    import pdb;pdb.set_trace()


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="model-train",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--dispatch_batches",
        type=str2bool,
        default=None,
        help=''' reference from accelerate v0.21.0, xxx/python3.10/site-packages/accelerate/accelerator.py line: 205
            > dispatch_batches (`bool`, *optional*):
            > If set to `True`, the dataloader prepared by the Accelerator is only iterated through on the main process
            > and then the batches are split and broadcast to each process. Will default to `True` for `DataLoader` whose
            > underlying dataset is an `IterableDataset`, `False` otherwise.
            forbid dispatch_batches will make dataloader faster, since the main process doesn't need broadcast the whole 
            global batch data to others''',
    )
    parser.add_argument(
        "--debug_mode",
        action="store_true",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default=None,
        help="model config file",
    )
    parser.add_argument(
        "--eval_config",
        type=str,
        default=None,
        help="eval config file",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args
