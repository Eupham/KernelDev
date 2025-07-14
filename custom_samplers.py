# custom_samplers.py
import torch
from torch.utils.data import Sampler, Dataset
from typing import List, Tuple, Iterator
import random
import math

# Attempt to import CombinedMultiTaskDataset, assuming it's discoverable.
# If not, this might need adjustment based on actual project structure.
# For now, we'll rely on type hinting and assume it can be imported by entry.py later.
# from combined_dataset import CombinedMultiTaskDataset
# Using forward reference for CombinedMultiTaskDataset type hint if direct import is problematic here.
CombinedMultiTaskDataset = 'CombinedMultiTaskDataset'


class StrictRatioBatchSampler(Sampler[List[int]]):
    def __init__(self,
                 dataset: Dataset,
                 batch_size: int,
                 ratios: Tuple[float, float, float, float], # (rank, nsp, span, lm)
                 drop_last: bool = True):

        super().__init__(dataset)

        if not all(hasattr(dataset, attr) for attr in ['rank_indices', 'nsp_indices', 'span_indices', 'lm_indices']):
            raise ValueError("Dataset must have 'rank_indices', 'nsp_indices', 'span_indices', and 'lm_indices' attributes.")

        self.dataset = dataset
        self.batch_size = batch_size
        self.ratios = ratios
        self.drop_last = drop_last

        self.rank_ratio, self.nsp_ratio, self.span_ratio, self.lm_ratio = ratios

        # Calculate number of samples per task type in each batch
        self.num_rank_per_batch = math.floor(self.batch_size * self.rank_ratio)
        self.num_nsp_per_batch = math.floor(self.batch_size * self.nsp_ratio)
        self.num_span_per_batch = math.floor(self.batch_size * self.span_ratio)

        # Assign remainder to the LM task to ensure batch_size is met
        current_sum = self.num_rank_per_batch + self.num_nsp_per_batch + self.num_span_per_batch
        self.num_lm_per_batch = self.batch_size - current_sum

        if self.num_lm_per_batch < 0 : # Should not happen if ratios sum to 1 and are positive
            raise ValueError("Calculated number of LM samples per batch is negative. Check ratios and batch size.")

        # Sanity check if total numbers make sense with batch size
        if (self.num_rank_per_batch + self.num_nsp_per_batch + self.num_span_per_batch + self.num_lm_per_batch) != self.batch_size:
            # This can happen due to flooring if not careful.
            # A robust way is to calculate for all but one, and the last gets the remainder.
            # The current logic of assigning remainder to lm_task is sufficient.
            # This check is more of a safeguard against logic errors.
            total_calculated = self.num_rank_per_batch + self.num_nsp_per_batch + self.num_span_per_batch + self.num_lm_per_batch
            if total_calculated != self.batch_size:
                 # This path indicates a logic error in the above calculations.
                 print(f"Warning: Batch composition numbers do not sum to batch_size. Sum={total_calculated}, BS={self.batch_size}")


        # These are pointers to the lists in the dataset, which are already shuffled
        self.rank_indices = self.dataset.rank_indices
        self.nsp_indices = self.dataset.nsp_indices
        self.span_indices = self.dataset.span_indices # New
        self.lm_indices = self.dataset.lm_indices

        if not self.rank_indices or not self.nsp_indices or not self.lm_indices or not self.span_indices:
            print("Warning: One or more task-specific index lists in the dataset are empty.")

    def __iter__(self) -> Iterator[List[int]]:
        # Create fresh copies and shuffle them for this epoch/iteration
        current_rank_indices = self.rank_indices[:]
        random.shuffle(current_rank_indices)
        current_nsp_indices = self.nsp_indices[:]
        random.shuffle(current_nsp_indices)
        current_span_indices = self.span_indices[:] # New
        random.shuffle(current_span_indices)
        current_lm_indices = self.lm_indices[:]
        random.shuffle(current_lm_indices)

        # Pointers for where we are in each list
        rank_ptr, nsp_ptr, span_ptr, lm_ptr = 0, 0, 0, 0

        while True:
            batch_indices = []

            # Check if we have enough samples for a full batch, if drop_last is True
            can_form_batch = True
            if (rank_ptr + self.num_rank_per_batch > len(current_rank_indices)) or \
               (nsp_ptr + self.num_nsp_per_batch > len(current_nsp_indices)) or \
               (span_ptr + self.num_span_per_batch > len(current_span_indices)) or \
               (lm_ptr + self.num_lm_per_batch > len(current_lm_indices)):
                can_form_batch = False

            if not can_form_batch:
                if self.drop_last:
                    break # Stop iteration
                else:
                    # Handle not drop_last: Take remaining, batch will be smaller
                    # This logic can get complex if we must strictly meet ratios even for smaller last batch.
                    # For now, let's assume if not drop_last, we just form a smaller batch with what's left,
                    # prioritizing keeping some of each type if possible, or just appending.
                    # A simpler approach for not drop_last is to just stop if any one list is exhausted
                    # if we must maintain the count for each type.
                    # The current __len__ implies drop_last=True behavior.
                    # For simplicity, this iterator will behave like drop_last=True if any list is exhausted.
                    # A more robust non-drop_last would require careful handling of remainders.
                    break


            # Collect Rank indices
            for _ in range(self.num_rank_per_batch):
                if rank_ptr < len(current_rank_indices):
                    batch_indices.append(current_rank_indices[rank_ptr])
                    rank_ptr += 1
                else:
                    break

            # Collect NSP indices
            for _ in range(self.num_nsp_per_batch):
                if nsp_ptr < len(current_nsp_indices):
                    batch_indices.append(current_nsp_indices[nsp_ptr])
                    nsp_ptr += 1
                else:
                     break

            # Collect Span Selection indices
            for _ in range(self.num_span_per_batch):
                if span_ptr < len(current_span_indices):
                    batch_indices.append(current_span_indices[span_ptr])
                    span_ptr += 1
                else:
                    break

            # Collect Span Selection indices
            for _ in range(self.num_span_per_batch):
                if span_ptr < len(current_span_indices):
                    batch_indices.append(current_span_indices[span_ptr])
                    span_ptr += 1
                else:
                    break

            # Collect LM indices
            for _ in range(self.num_lm_per_batch):
                if lm_ptr < len(current_lm_indices):
                    batch_indices.append(current_lm_indices[lm_ptr])
                    lm_ptr += 1
                else:
                    break

            if not batch_indices:
                break

            # If, due to not self.drop_last and exhaustion, the batch is smaller than intended
            # or empty, this might need more logic. For now, if it's empty, we break.
            # If it's smaller than batch_size and not self.drop_last, it will be yielded.

            if len(batch_indices) == self.batch_size or (not self.drop_last and len(batch_indices) > 0) :
                random.shuffle(batch_indices) # Shuffle the combined list for the batch
                yield batch_indices
            elif self.drop_last and len(batch_indices) < self.batch_size and len(batch_indices) > 0:
                # This case should ideally not be hit if can_form_batch logic is correct for drop_last
                # It means we couldn't form a full batch. So, break.
                break
            elif not batch_indices: # Should be caught by the earlier check
                 break
            else: # Not enough for a full batch and drop_last is true
                 break


    def __len__(self) -> int:
        # Calculate how many full batches can be formed
        # This is limited by the task that runs out of samples first.
        if not self.drop_last:
            # A simple approximation for non-drop_last
            total_samples = len(self.rank_indices) + len(self.nsp_indices) + len(self.span_indices) + len(self.lm_indices)
            return (total_samples + self.batch_size - 1) // self.batch_size

        # If drop_last=True, calculate based on the limiting task
        min_batches = float('inf')
        if self.num_rank_per_batch > 0:
            min_batches = min(min_batches, len(self.rank_indices) // self.num_rank_per_batch)
        if self.num_nsp_per_batch > 0:
            min_batches = min(min_batches, len(self.nsp_indices) // self.num_nsp_per_batch)
        if self.num_span_per_batch > 0:
            min_batches = min(min_batches, len(self.span_indices) // self.num_span_per_batch)
        if self.num_lm_per_batch > 0:
            min_batches = min(min_batches, len(self.lm_indices) // self.num_lm_per_batch)

        return 0 if min_batches == float('inf') else int(min_batches)

if __name__ == '__main__':
    # Example Usage (requires a mock CombinedMultiTaskDataset)
    class MockDataset(Dataset):
        def __init__(self, num_samples_per_type=(100, 50, 50)): # lm, lev, nsp
            self.lm_indices = list(range(num_samples_per_type[0]))
            self.lev_indices = list(range(num_samples_per_type[0], num_samples_per_type[0] + num_samples_per_type[1]))
            self.nsp_indices = list(range(num_samples_per_type[0] + num_samples_per_type[1], sum(num_samples_per_type)))

            # Ensure these are shuffled as the sampler expects
            random.shuffle(self.lm_indices)
            random.shuffle(self.lev_indices)
            random.shuffle(self.nsp_indices)

            self.data = {} # Mock actual data
            idx_counter = 0
            for i in self.lm_indices: self.data[i] = (f"lm_sample_{idx_counter}", 0.0); idx_counter+=1
            idx_counter = 0
            for i in self.lev_indices: self.data[i] = (f"lev_sample_{idx_counter}", 1.0); idx_counter+=1
            idx_counter = 0
            for i in self.nsp_indices: self.data[i] = (f"nsp_sample_{idx_counter}", 2.0); idx_counter+=1

        def __len__(self):
            return len(self.lm_indices) + len(self.lev_indices) + len(self.nsp_indices)

        def __getitem__(self, idx):
            # This won't be called if batch_sampler is used correctly by DataLoader
            # but for completeness:
            return self.data[idx]

    mock_dataset = MockDataset(num_samples_per_type=(1000, 500, 500)) # lm, lev, nsp

    # Ratios: (rank_ratio, nsp_ratio, span_ratio, lm_ratio)
    batch_size = 20 # Use a size divisible by 10 for easy ratio checking
    ratios = (0.2, 0.2, 0.2, 0.4)

    sampler = StrictRatioBatchSampler(mock_dataset, batch_size, ratios, drop_last=True)

    print(f"StrictRatioBatchSampler demo:")
    print(f"  Dataset size: {len(mock_dataset)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Ratios (Rank, NSP, Span, LM): {ratios}")
    print(f"  Num Rank/batch: {sampler.num_rank_per_batch}")
    print(f"  Num NSP/batch: {sampler.num_nsp_per_batch}")
    print(f"  Num Span/batch: {sampler.num_span_per_batch}")
    print(f"  Num LM/batch: {sampler.num_lm_per_batch}")
    print(f"  Expected num batches (__len__): {len(sampler)}")

    batch_count = 0
    for batch_idx_list in sampler:
        batch_count += 1
        if batch_count <= 5: # Print details for first 5 batches
            print(f"  Batch {batch_count}: {len(batch_idx_list)} items. Indices: {batch_idx_list[:5]}...")
            # Verify ratios in this batch by checking source of indices (more complex to do here)

            # For a more thorough check, one would need to access the original dataset's type per index
            # For now, we trust the sampler's internal logic.

    print(f"Total batches yielded: {batch_count}")

    # Test with DataLoader
    # Note: When using batch_sampler, DataLoader's batch_size should be 1 (or None),
    # shuffle=False, sampler=None, drop_last=False.
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        mock_dataset,
        batch_sampler=sampler,
        # These must be default when batch_sampler is used:
        batch_size=1, # This is per-item batch_size, not the batch_sampler's output batch size
        shuffle=False,
        sampler=None,
        drop_last=False, # drop_last is handled by our sampler
        num_workers=0 # Keep simple for testing sampler
    )

    print("\nTesting with DataLoader:")
    dl_batch_count = 0
    for i, batch_data_list in enumerate(dataloader):
        dl_batch_count +=1
        # batch_data_list will be a list of items, where len(batch_data_list) == batch_size from sampler
        # if collate_fn is default. Each item is dataset[idx].
        # If default collate is used, and dataset returns tuples,
        # batch_data_list will be like [(text1, type1), (text2, type2), ...]
        # If a custom collate is used, it will be whatever collate returns.
        # For this test, default_collate will make it a list of (text, type_flag) tuples.

        if i < 2: # Print first 2 batches from DataLoader
            print(f"  DataLoader Batch {i+1}: (length {len(batch_data_list)})")
            # To verify types, we'd need to look at batch_data_list[j][1] for each j
            # This is just testing the indices are yielded correctly.
            # The actual data loading uses CombinedMultiTaskDataset with its own __getitem__
            # which will be called with the indices from our sampler.

            # Example of how to check item types if dataset returned type_flag
            # type_counts = {'lm':0, 'lev':0, 'nsp':0}
            # for item_text, item_type_val in batch_data_list:
            #     if item_type_val == 0.0: type_counts['lm'] +=1
            #     elif item_type_val == 1.0: type_counts['lev'] +=1
            #     elif item_type_val == 2.0: type_counts['nsp'] +=1
            # print(f"    Type counts: {type_counts}")


    print(f"Total batches from DataLoader: {dl_batch_count}")
    if dl_batch_count == len(sampler):
        print("✓ DataLoader batch count matches sampler length.")
    else:
        print(f"✗ Mismatch: DataLoader batches: {dl_batch_count}, Sampler length: {len(sampler)}")
