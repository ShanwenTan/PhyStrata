# Getting started

This guide separates source code from large, licensed assets. The repository can be cloned or uploaded without a dataset or checkpoint; training and inference require paths to them.

## 1. Environment

The reference runs used Python 3.10, CUDA-enabled PyTorch 2.6.0, diffusers 0.33.0, and transformers 4.49.0 on Linux. Install a PyTorch build compatible with your GPU driver, then run `python -m pip install -r requirements.txt` and `python -m pip install -e .`. For a close environment match, consult `requirements.lock.txt`; it contains platform-specific packages and should be applied only on a compatible system.

For training, verify that `python -c "import torch; print(torch.cuda.device_count())"` sees the expected GPUs. `ffmpeg` is needed for MP4 generation and evaluation. `wandb login` is optional and required only for online tracking.

## 2. Prepare assets

Obtain these independently under their original licenses:

1. A Diffusers-format CogVideoX-2B or CogVideoX-5B base model directory.
2. The OpenVid64K metadata CSV and referenced video files for training.
3. A PhyStrata LoRA adapter and `predictor.safetensors` for inference or evaluation.
4. The VideoCon physics scorer for VideoPhy evaluation.

Six selected generated examples are included under `assets/examples/`. No base model, dataset shard, adapter, or scorer is bundled. The training launcher checks the supplied model directory, CSV, and video root before starting distributed workers. A dataset row can use `caption` and `video_path` columns, with a relative video filename under the supplied video root.

## 3. Inspect and run the training recipe

Start with `--dry-run`. It writes `resolved_config.yaml` and `dataset_config.json` in the output directory so every path and hyperparameter can be reviewed:

```bash
python scripts/launch_train.py --size 2b \
  --model-path /models/CogVideoX-2b \
  --dataset-csv /datasets/openvid64k.csv \
  --video-root /datasets/openvid/videos \
  --output-dir /runs/phystrata-2b --dry-run
```

Remove `--dry-run` to start 8 local workers. Add `--wandb-project PROJECT` for online logging. The recipe sets the drift-weighting flag and writes checkpoints every 500 steps. It uses a per-device batch of four for 2B and two for 5B.

For 5B, use two 8-GPU nodes with the same shared absolute paths. Run one launcher per node with `--size 5b --nnodes 2 --nproc-per-node 8 --master-addr NODE_0_IP`; set node ranks to 0 and 1 respectively. The per-device batch is two, giving a global batch of 32, and the recipe runs for 2,000 steps. Set `--master-port` to a free port accessible between nodes. A full run needs enough GPU memory for CogVideoX-5B training and a shared output filesystem.

## 4. Inference from a checkpoint

The expected checkpoint layout is:

```text
checkpoint/
├── pytorch_lora_weights.safetensors
└── predictor.safetensors
```

Use `python tests/generate_phystrata.py --help` for all options. A minimal command is:

```bash
CUDA_VISIBLE_DEVICES=0 python tests/generate_phystrata.py \
  --model_path /models/CogVideoX-5b \
  --lora_path /checkpoints/phystrata-5b-step500 \
  --prompt "A stream of water flows into a ceramic bowl." \
  --output_file outputs/preview
```

The predictor loader matches parameter names after an optional single namespace, allowing compatible checkpoints saved by different wrappers. It does not rewrite checkpoint files. The LoRA file must match the selected 2B or 5B base architecture.

## 5. VideoPhy evaluation

VideoPhy generation and scoring are performed by `eval/scripts/run_videophy_eval.sh`. Set `BASE_MODEL`, `CHECKPOINT_PATH`, `VIDEOCON_CHECKPOINT`, and `RESULT_PATH` explicitly. For example:

```bash
CUDA_VISIBLE_DEVICES=0,1 BASE_MODEL=/models/CogVideoX-5b \
CHECKPOINT_PATH=/checkpoints/phystrata-5b-step500 \
VIDEOCON_CHECKPOINT=/models/videocon_physics \
MODEL_TYPE=phystrata GENERATE_TYPE=lora WORKERS_PER_RANK=2 \
RESULT_PATH=outputs/eval/step500 bash eval/scripts/run_videophy_eval.sh
```

Use `EVAL_LIMIT` for a smoke run, then omit it for the full 344-prompt split. Use a new result directory per checkpoint. `RESUME_GENERATION=1` can resume generation with the same checkpoint and settings after interruption. The output summary reports SA and PC.

## 6. Common checks

- If a path check fails, inspect `resolved_config.yaml`, the dataset JSON, and the existence of files named in the CSV.
- If multi-node rendezvous hangs, check `NODE_0_IP`, port reachability, world size, and shared paths on both hosts.
- If generation runs out of GPU memory, lower `WORKERS_PER_RANK` before changing the inference settings. The 5B reference evaluation used two workers per 80 GB GPU.
- If scorer loading fails, verify the scorer directory separately; no scorer weights are included in this repository.
