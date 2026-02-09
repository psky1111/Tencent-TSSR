#!/bin/bash
if [[ -z "${CONFIG}" ]]; then
  EVAL_CONFIG=conf/eval_configs/eval_flow_new_backbone.yaml
  echo ${MODEL_CONFIG}
else
  EVAL_CONFIG="${CONFIG}"
fi

# 配置不同运行模式特有的参数
if [ "$1" == "DEBUG" ]  # DEBUG模式的参数配置
then
    echo -e "\033[32m$0 Running in DEBUG mode \033[0m"
    export EXTRA_ARGS="--debug_mode"
    export GRADIENT_ACCUMULATION_STEPS=1
else
    echo -e "\033[32m$0 Running in Eval mode, run in DEBUG mode by ./start_evaluation.sh DEBUG \033[0m"
    export EXTRA_ARGS="--resume_from_checkpoint=latest"
    export GRADIENT_ACCUMULATION_STEPS=16
fi

# 其他公共参数配置
export OFFLOAD_DEVICE=cpu
export ZERO_STAGE=2
export MIXED_PRECISION=bf16
export OMP_NUM_THREADS=1
echo -e "EXTRA_ARGS=$EXTRA_ARGS"
pip3 install pyassimp
echo "Y" | apt install libassimp-dev
pip3 install --upgrade diffusers
pip3 install rtree
# 获取当前脚本的绝对路径
script_path="$(dirname "$(realpath "$0")")"
# 切换到脚本所在的目录
cd "$script_path"
export PYTHONPATH=.

# 配置多节点训练
if nvidia-smi --query-gpu=gpu_name --format=csv,noheader | grep -q "H20"; then
    export NCCL_IB_GID_INDEX=3
    export NCCL_IB_SL=3
    export NCCL_CHECK_DISABLE=1
    export NCCL_P2P_DISABLE=0
    export NCCL_IB_DISABLE=0
    export NCCL_LL_THRESHOLD=16384
    export NCCL_IB_CUDA_SUPPORT=1
    export NCCL_SOCKET_IFNAME=bond1
    export UCX_NET_DEVICES=bond1
    export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6
    export NCCL_COLLNET_ENABLE=0
    export SHARP_COLL_ENABLE_SAT=0
    export NCCL_NET_GDR_LEVEL=2
    export NCCL_IB_QPS_PER_CONNECTION=4
    export NCCL_IB_TC=160
    export NCCL_PXN_DISABLE=1
else
    export NCCL_SOCKET_IFNAME=eth1
    export NCCL_IB_GID_INDEX=3
    export NCCL_IB_SL=3
    export NCCL_CHECKS_DISABLE=1
    export NCCL_IB_DISABLE=0
    export NCCL_IBEXT_DISABLE=0
    export NCCL_P2P_DISABLE=0
    export NCCL_LL_THRESHOLD=16384
    export NCCL_IB_CUDA_SUPPORT=1
    # export NCCL_DEBUG=INFO
    nccl_ib_hca=$(bash tools/scripts/show_gids | grep $(hostname -I) | grep v2 | awk '{print $1 ":" $2}' )
    echo "nccl_ib_hca is ${nccl_ib_hca}"
    export NCCL_IB_HCA="=$nccl_ib_hca"
fi

# 配置 accelerate
if [[ -z ${NODE_IP_LIST} ]]; then
    hostfile="/etc/hostfile"
else
    echo $NODE_IP_LIST > /dockerdata/env.txt
    sed "s/:/ slots=/g" /dockerdata/env.txt | sed "s/,/\n/g" > "/dockerdata/hostfile"
    hostfile="/dockerdata/hostfile"
fi
process_num=$(awk -F'slots=' '{sum += $2} END {print sum}' $hostfile)
if [ "$1" == "DEBUG" ]
then
    process_num=1
fi
export accelerate_config_yaml="/dockerdata/accelerate_multinode.yaml"
sed "s#CONFIG_HOSTFILE#${hostfile}#" "config/accelerate_multinode.yaml" \
    | sed "s/CONFIG_HOST_INDEX/${INDEX}/" \
    | sed "s/CONFIG_CHIEF_IP/${CHIEF_IP}/" \
    | sed "s/CONFIG_HOST_NUM/${HOST_NUM}/" \
    | sed "s/CONFIG_PROCESS_NUM/${process_num}/" \
    | sed "s/CONFIG_GRADIENT_ACCUMULATION_STEPS/${GRADIENT_ACCUMULATION_STEPS}/" \
    | sed "s/CONFIG_OFFLOAD_DEVICE/${OFFLOAD_DEVICE}/" \
    | sed "s/CONFIG_ZERO_STAGE/${ZERO_STAGE}/" \
    | sed "s/CONFIG_MIXED_PRECISION/${MIXED_PRECISION}/" \
    > ${accelerate_config_yaml}

# 启动评估任务
accelerate launch --config_file=${accelerate_config_yaml} --mixed_precision=$MIXED_PRECISION src/eval.py \
  --eval_config ${EVAL_CONFIG} \
  --gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
  --mixed_precision $MIXED_PRECISION \
  $EXTRA_ARGS
