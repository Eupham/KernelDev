"""
Custom PyTorch Samplers for advanced data loading control.
"""

import torch
from torch.utils.data.distributed import DistributedSampler
from typing import Optional, Iterator
import torch.distributed as dist
import math
from itertools import islice

class ResumableDistributedSampler(DistributedSampler):
    """
    A DistributedSampler that can be resumed from a specific batch.

    This sampler is designed to work with a DataLoader to efficiently skip
    a specified number of batches at the beginning of an epoch. Instead of
    iterating through the DataLoader and discarding batches, this sampler
    skips the initial data *indices* before they are even fetched by the
    DataLoader. This significantly speeds up resumption from a mid-epoch checkpoint.

    Args:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, `world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within `num_replicas`.
            By default, `rank` is retrieved from the current distributed group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.
        start_batch (int, optional): The batch index to start from. The sampler
            will skip all indices corresponding to batches before this one.
            Default: ``0``.
        batch_size (int, optional): The per-replica batch size used by the
            DataLoader. This is required to calculate the number of indices to
            skip. Default: ``1``.
    """
    def __init__(self, dataset, num_replicas: Optional[int] = None, rank: Optional[int] = None,
                 shuffle: bool = True, seed: int = 0, drop_last: bool = False,
                 start_batch: int = 0, batch_size: int = 1):

        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)

        self.start_batch = start_batch
        # Note: The batch_size here must be the PER-REPLICA batch size.
        self.start_index = self.start_batch * batch_size

        # Keep track of the original number of samples for the full epoch
        self.num_samples_full_epoch = self.num_samples

        # Adjust num_samples to reflect the number of remaining samples to be processed
        if self.start_index >= self.num_samples_full_epoch:
            print(f"Warning: start_index ({self.start_index}) is "
                  f"greater than or equal to the number of samples per replica "
                  f"({self.num_samples_full_epoch}). This sampler will yield no data.")
            self.num_samples = 0
        else:
            self.num_samples = self.num_samples_full_epoch - self.start_index

    def __iter__(self) -> Iterator[int]:
        """
        Returns an iterator over the indices of the dataset.

        If start_batch > 0, the iterator will efficiently skip the first
        `start_index` indices.
        """
        # Get the full iterator of indices from the parent class
        full_iterator = super().__iter__()

        if self.start_index > 0:
            # Use itertools.islice to create an iterator that starts from the desired index.
            # This is highly efficient as it doesn't process the skipped items.
            resumed_iterator = islice(full_iterator, self.start_index, self.num_samples_full_epoch)
            print(f"Sampler for Rank {self.rank} resuming from batch {self.start_batch} "
                  f"(skipping {self.start_index} indices).")
            return resumed_iterator
        else:
            # If starting from the beginning, return the full iterator
            return full_iterator

    def __len__(self) -> int:
        """
        Returns the number of samples that will be yielded by this sampler.

        This is the total number of samples for the replica, adjusted for the
        starting batch, so the DataLoader knows when the epoch has ended.
        """
        return self.num_samples
