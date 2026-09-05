"""
From: https://github.com/octo-models/octo/blob/main/examples/06_pytorch_oxe_dataloader.py

This example shows how to use the `src.data` dataloader with PyTorch by wrapping it in a simple PyTorch dataloader. The config below also happens to be our exact pretraining config (except for the batch size and shuffle buffer size, which are reduced for demonstration purposes).
"""

from collections import deque
import random

import numpy as np
import tensorflow as tf
import torch

tf.config.set_visible_devices([], "GPU")


class TorchRLDSDataset(torch.utils.data.IterableDataset):
    """Thin wrapper around RLDS dataset for use with PyTorch dataloaders."""

    def __init__(
        self,
        rlds_dataset,
        train=True,
        balance_by_task=False,
        task_sample_counts=None,
        samples_per_task=None,
        seed=42,
    ):
        self._rlds_dataset = rlds_dataset
        self._is_train = train

        self._balance_by_task = balance_by_task
        self._seed = seed
        self._epoch = 0

        if not self._balance_by_task:
            self._task_sample_counts = None
            self._task_ids = None
            self._samples_per_task = None
            self._duplicates_needed = None
            return

        if not self._is_train:
            raise ValueError(
                "Task-balanced sampling is intended only for training."
            )

        if task_sample_counts is None:
            raise ValueError(
                "task_sample_counts must be provided when balance_by_task=True."
            )

        if samples_per_task is None:
            raise ValueError(
                "samples_per_task must be provided when balance_by_task=True."
            )

        self._task_sample_counts = {
            int(task_id): int(count)
            for task_id, count in task_sample_counts.items()
        }

        self._task_ids = tuple(sorted(self._task_sample_counts.keys()))
        self._samples_per_task = int(samples_per_task)

        if len(self._task_ids) == 0:
            raise ValueError("task_sample_counts cannot be empty.")

        if self._samples_per_task <= 0:
            raise ValueError("samples_per_task must be greater than zero.")

        if any(count <= 0 for count in self._task_sample_counts.values()):
            raise ValueError("All task sample counts must be greater than zero.")

        if any(
            count > self._samples_per_task
            for count in self._task_sample_counts.values()
        ):
            raise ValueError(
                "samples_per_task cannot be smaller than an original task count."
            )

        self._duplicates_needed = {
            task_id: self._samples_per_task - count
            for task_id, count in self._task_sample_counts.items()
        }

        # We want duplicated samples to be distinct within the same epoch.
        if any(
            self._duplicates_needed[task_id]
            > self._task_sample_counts[task_id]
            for task_id in self._task_ids
        ):
            raise ValueError(
                "A task requires more duplicates than original samples. "
                "Distinct oversampling without replacement is not possible."
            )


    def _get_task_id(self, sample):
        if "task_id" not in sample:
            raise KeyError(
                "Sample does not contain 'task_id'. "
                "Make sure task_id is propagated through the RLDS pipeline."
            )

        task_id_array = np.asarray(sample["task_id"])

        if task_id_array.size != 1:
            raise ValueError(
                "Expected scalar task_id after trajectory flattening, "
                f"got shape {task_id_array.shape}."
            )

        task_id = int(task_id_array.item())

        if task_id not in self._task_sample_counts:
            raise ValueError(
                f"Unexpected task_id={task_id}. "
                f"Expected one of {self._task_ids}."
            )

        return task_id 


    def __iter__(self):
        # Original behavior.
        if not self._balance_by_task:
            for sample in self._rlds_dataset.as_numpy_iterator():
                yield sample
            return

        # Multiple PyTorch workers would create independent copies of the
        # balancing stream and break the 12-task ordering guarantee.
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            raise RuntimeError(
                "Task-balanced TorchRLDSDataset requires DataLoader(num_workers=0)."
            )

        # Different RNG sequence at every epoch.
        rng = random.Random(self._seed + self._epoch)

        task_buffers = {
            task_id: deque()
            for task_id in self._task_ids
        }
        # self._task_ids = (0, 1, ..., 11)
        # {
        #     0: deque([]),
        #     1: deque([]),
        #     2: deque([]),
        #     ...
        #     11: deque([]),
        # }

        # Reservoir containing the samples that will be duplicated
        # at the end of this epoch.
        duplicate_reservoirs = {
            task_id: []
            for task_id in self._task_ids
        }
        # sample originali che abbiamo scelto come candidati da usare una seconda volta nell'epoca.
        # duplicate_reservoirs[0] len(duplicate_reservoirs[0]) == 302
        # duplicate_reservoirs[1]
        # ...
        # duplicate_reservoirs[11]

        seen_counts = {
            task_id: 0
            for task_id in self._task_ids
        }
        # Tiene traccia di quanti sample originali abbiamo letto dal RLDS.
        # All'inizio:

        # task 0 → 0
        # task 1 → 0
        # ...

        # A fine passata deve essere:

        # task 0  → 1906
        # task 1  → 1808
        # task 2  → 2124
        # ...
        # task 11 → 2199

        
        
        yielded_counts = {
            task_id: 0
            for task_id in self._task_ids
        }
        # Questo conta invece quanti sample sono stati effettivamente consegnati al trainer.

        def emit_balanced_groups():
            # continua soltanto finché tutti e 12 i task hanno almeno un sample disponibile.
            while all(
                len(task_buffers[task_id]) > 0
                for task_id in self._task_ids
            ):
                task_order = list(self._task_ids) # [0,1,2,3,4,5,6,7,8,9,10,11]
                rng.shuffle(task_order) # [7,2,5,11,0,3,9,1,8,4,10,6]

                for task_id in task_order:
                    yielded_counts[task_id] += 1 # registra che un sample del task i è stato usato.
                    yield task_buffers[task_id].popleft() #prende il sample più vecchio ancora in attesa.

        # ------------------------------------------------------------
        # FIRST PHASE:
        # consume every original sample exactly once.
        # ------------------------------------------------------------

        # Questo è l'attraversamento dei 23.997 sample originali.
        for sample in self._rlds_dataset.as_numpy_iterator():
            task_id = self._get_task_id(sample)

            seen_counts[task_id] += 1
            seen_for_task = seen_counts[task_id]

            task_buffers[task_id].append(sample)

            # Reservoir sampling:
            # select exactly K distinct original samples uniformly,
            # where K is the number of duplicates needed for this task.
            num_duplicates = self._duplicates_needed[task_id]

            if num_duplicates > 0:
                reservoir = duplicate_reservoirs[task_id]

                if len(reservoir) < num_duplicates:
                    reservoir.append(sample)
                else:
                    replacement_index = rng.randrange(seen_for_task)

                    if replacement_index < num_duplicates:
                        reservoir[replacement_index] = sample

            # As soon as all tasks are available, emit one sample per task.
            yield from emit_balanced_groups()

        # ------------------------------------------------------------
        # Verify that the finite RLDS dataset contained exactly the
        # expected number of original samples.
        # A questo punto seen_counts deve essere esattamente:

        # {
        #     0: 1906,
        #     1: 1808,
        #     ...
        #     11: 2199,
        # }
        # ------------------------------------------------------------

        if seen_counts != self._task_sample_counts:
            raise RuntimeError(
                "Unexpected number of samples in the training dataset.\n"
                f"Expected: {self._task_sample_counts}\n"
                f"Observed: {seen_counts}"
            )

        # ------------------------------------------------------------
        # SECOND PHASE:
        # append only the selected duplicates.
        # Restano:

        # task 0  → 1906 - 1808 = 98
        # task 1  → 0
        # task 2  → 316
        # task 3  → 180
        # task 4  → 326
        # task 5  → 36
        # task 6  → 400
        # task 7  → 29
        # task 8  → 128
        # task 9  → 267
        # task 10 → 130
        # task 11 → 391

        # Questi sono originali veri che non sono ancora stati emessi.
        # ------------------------------------------------------------

        for task_id in self._task_ids:
            reservoir = duplicate_reservoirs[task_id]

            expected_duplicates = self._duplicates_needed[task_id]

            if len(reservoir) != expected_duplicates:
                raise RuntimeError(
                    f"Task {task_id}: expected {expected_duplicates} "
                    f"duplicates, got {len(reservoir)}."
                )

            # Avoid any artificial order in the duplicated samples.
            rng.shuffle(reservoir)

            task_buffers[task_id].extend(reservoir) # I duplicati vanno in fondo ai buffer

        # At this point every task must have exactly the same number
        # of samples remaining.
        remaining_lengths = {
            task_id: len(task_buffers[task_id])
            for task_id in self._task_ids
        }
        # 400 è la differenza massima in numero di step
        # tra task più lungo e task più corto.
        # {
        #     0: 400,
        #     1: 400,
        #     ...
        #     11: 400,
        # }

        if len(set(remaining_lengths.values())) != 1:
            raise RuntimeError(
                "Task buffers are not balanced after oversampling: "
                f"{remaining_lengths}"
            )

        # Emit all remaining original + duplicated samples.
        yield from emit_balanced_groups()

        # ------------------------------------------------------------
        # Final consistency checks.
        # ------------------------------------------------------------

        non_empty_buffers = {
            task_id: len(task_buffers[task_id])
            for task_id in self._task_ids
            if len(task_buffers[task_id]) != 0
        }

        if non_empty_buffers:
            raise RuntimeError(
                f"Non-empty task buffers at end of epoch: {non_empty_buffers}"
            )

        expected_yielded_counts = {
            task_id: self._samples_per_task
            for task_id in self._task_ids
        }

        if yielded_counts != expected_yielded_counts:
            raise RuntimeError(
                "Unexpected balanced epoch counts.\n"
                f"Expected: {expected_yielded_counts}\n"
                f"Observed: {yielded_counts}"
            )

        self._epoch += 1
            

    def __len__(self):
        if self._balance_by_task:
            return len(self._task_ids) * self._samples_per_task

        # Preserve original behavior when task balancing is disabled.
        return self._rlds_dataset.true_total_length