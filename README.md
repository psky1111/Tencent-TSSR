# Tencent-TSSR
Official implementation of TSSR


# Hugging-face pre-train models 2B 7k face

https://huggingface.co/skyofsky/TSSR/tree/main


# Inference
bash start_eval.sh

Two file need to update when using the inference code:

1. Please download the ckpt from huggingface and then put it to ckpt/checkpoint-####/

2. Modify the ./config/eval_configs/eval.yaml to match the path. We offer an example in this path.

3. Parse the json file for your eval point cloud. We also offer an example in ./data/ including the strcture of the json file that you need to parse.

4. Modify the ./config/eval_configs/eval.yaml 


Note: This code only test in the linux. We will update the code in Windows as soon as possible


# Tips
We use assimp lib. It needs install assimp-dev for your system

linux can install via: apt install libassimp-dev

