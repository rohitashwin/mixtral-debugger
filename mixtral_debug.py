import os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import deepspeed
from torch.onnx import export
import torch.nn.functional as F


# get total tensor-parallel ranks (set by deepspeed launcher)
world_size = int(os.getenv("WORLD_SIZE", "1"))
rank = int(os.getenv("RANK", "0"))

model_name = "mistralai/Mixtral-8x7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
orig_model = AutoModelForCausalLM.from_pretrained(model_name)
print(orig_model)

if rank == 0:
    print("Original Model Topology: ")
    print(orig_model)

# wrap for inference with tensor parallelism only
model = deepspeed.init_inference(
    orig_model,
    tensor_parallel={"tp_size": world_size},
    dtype=torch.float,
)

if rank == 0:
    model = model.module  # unwrap the model to get the original topology
    print(f"Parallelism: WORLD_SIZE={world_size}, RANK={rank}")
    print("Deepspeed Inference Model Topology: ")
    print(model)
