import yaml
import subprocess
import sys
import os
import argparse
from pathlib import Path

def build_torchrun_cmd(config, nproc_per_node=1):
    base_cmd = [
        "torchrun",
        "--nnodes=1",
        f"--nproc_per_node={nproc_per_node}",
        "--master_port=29501",
        "-m", "SYMBA_SSM.main"
    ]

    # Convert config dict to CLI arguments
    for key, value in config.items():
        cli_key = f"--{key}"
        if isinstance(value, bool):
            if value:  # only include if True
                base_cmd.append(cli_key)
        else:
            base_cmd.extend([cli_key, str(value)])

    return base_cmd

def main():
    parser = argparse.ArgumentParser(description="Run torchrun with config file")
    parser.add_argument("--config_path",  type=str, help="Path to the YAML config file")
    parser.add_argument("--cuda-visible-devices", type=str,
                       help="Comma-separated list of GPU indices (e.g., '0,1,2,3')")
    parser.add_argument("--nproc-per-node", type=int, default=1,
                       help="Number of processes per node (default: 1)")
    parser.add_argument("--lm_head", action="store_true", help="Use MambaLMHead")
    parser.add_argument("--vanilla_transformer", action="store_true", help="Use Vanilla Transformer")

    args = parser.parse_args()

    config_path = Path(args.config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Set CUDA_VISIBLE_DEVICES if provided
    env = os.environ.copy()
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
        print(f"Setting CUDA_VISIBLE_DEVICES={args.cuda_visible_devices}")

    cmd = build_torchrun_cmd(config, args.nproc_per_node)
    if args.lm_head:
        cmd.append('--lm_head')
    elif args.vanilla_transformer:
        cmd.append('--vanilla_transformer')
    print("Running command:\n", " ".join(cmd))

    subprocess.run(cmd, env=env)

if __name__ == "__main__":
    main()