import copy
import math
from typing import Dict, Tuple


def vanilla_bucket_weights(bucket_num_info: Dict[Tuple[int, int], int], **kwargs):
    return copy.deepcopy(bucket_num_info)


def token_balanced_bucket_weights(bucket_num_info, project_args, **kwargs):
    bucket_weights = {}
    for bucket_key, sample_num in bucket_num_info.items():
        low, high = bucket_key
        avg_triangles = (low + high) / 2
        segment_num = math.ceil(avg_triangles / project_args.tokenizer.params.n_sliding_triangles)
        face_train_prob = min(project_args.data.segment_per_batch / segment_num, 1)
        bucket_weights[bucket_key] = sample_num * (1 / face_train_prob)
    return bucket_weights


def scaled_bucket_weights(bucket_num_info, scale_conf: Dict[Tuple[int, int], float], **kwargs):
    bucket_weights = copy.deepcopy(bucket_num_info)
    for face_range, scale_factor in scale_conf.items():
        for bucket_key, weight in bucket_weights.items():
            if (face_range[0] <= bucket_key[0] < face_range[1]) or (bucket_key[0] <= face_range[0] < bucket_key[1]):
                bucket_weights[bucket_key] = weight * scale_factor
    return bucket_weights


def comlpex_bucket_weights(bucket_num_info, project_args, scale_conf=None, **kwargs):
    bucket_weights = token_balanced_bucket_weights(bucket_num_info, project_args=project_args)
    
    if scale_conf is None:
        scale_conf = {(0, 1024): 0.1}
    bucket_weights = scaled_bucket_weights(bucket_weights, scale_conf=scale_conf)
    return bucket_weights


def scaled_category_weights(category_num_in_each_bucket, category_scale_conf=None, **kwargs):
    if category_scale_conf is None:
        return category_num_in_each_bucket
    
    category_weights_in_each_bucket = copy.deepcopy(category_num_in_each_bucket)
    for face_bucket, category_weights in category_weights_in_each_bucket.items():
        for category, scale_factor in category_scale_conf.items():
            if category not in category_weights:
                continue
            category_weights[category] *= scale_factor
    
    return category_weights_in_each_bucket
