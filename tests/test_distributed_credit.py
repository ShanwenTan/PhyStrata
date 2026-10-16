"""Run with torchrun --nproc_per_node=2 to test global-batch midranks."""

import os

import torch
import torch.distributed as dist

from finetrainers.models.wan.physical_motion_phystrata import _motion_credit_weights


def main():
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    target = torch.zeros(2, 5, 1, 1, 1)
    if rank == 1:
        target[0, :, 0, 0, 0] = torch.arange(5)
        target[1, :, 0, 0, 0] = 2 * torch.arange(5)
    video, _ = _motion_credit_weights(target, tau=2, beta_frame=0)
    expected = torch.tensor([0.8, 0.8] if rank == 0 else [1.1, 1.3])
    torch.testing.assert_close(video, expected, atol=1e-6, rtol=0)
    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print("two-rank global-batch midrank check passed")


if __name__ == "__main__":
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        raise SystemExit("Run this check with exactly two torchrun workers")
    main()
