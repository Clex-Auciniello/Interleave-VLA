import logging

import tensorflow as tf

from src.data.interleaved_dataset import make_interleaved_dataset
from src.data.dataset_torch import TorchRLDSDataset
from src.data.oxe import make_oxe_dataset_kwargs_and_weights
from src.utils.monitor import log_execution_time

tf.config.set_visible_devices([], "GPU")
log = logging.getLogger(__name__)


class TorchRLDSInterleavedDataset:
    @log_execution_time(log)
    def __init__(self, config, train=True):
        balance_by_task = (
            train and config.get("balance_by_task", False)
        )
        repeat_dataset = config.get("repeat_dataset", True)

        if balance_by_task and repeat_dataset:
            raise ValueError(
                "Task-balanced training requires repeat_dataset=False."
            )
        dataset_kwargs_list, sample_weights = make_oxe_dataset_kwargs_and_weights(
            config.dataset_mix,
            config.data_path,
            load_proprio=config.load_proprio,
            load_camera_views=("primary",),
        )
        for dataset_kwargs in dataset_kwargs_list:
            dataset_kwargs["language_key"] = "interleaved_instruction"
        dataset = make_interleaved_dataset(
            dataset_kwargs_list,
            sample_weights,
            train=train,
            split=config.get("split", None),
            shuffle_buffer_size=config.shuffle_buffer_size,
            batch_size=None,  # batching will be handles in PyTorch Dataloader object
            balance_weights=True,
            repeat_dataset=repeat_dataset,
            traj_transform_kwargs=dict(
                # goal_relabeling_strategy="uniform",   # no neeed for goal relabeling
                window_size=config.window_size,
                action_horizon=config.action_horizon,
                subsample_length=None,
                skip_unlabeled=config.skip_unlabeled,  # skip ones without language annotation
            ),
            frame_transform_kwargs=dict(
                image_augment_kwargs={
                    "primary": dict(
                        random_brightness=[0.05],
                        random_contrast=[0.95, 1.05],
                        random_saturation=[0.95, 1.05],
                        augment_order=[
                            "random_brightness",
                            "random_contrast",
                            "random_saturation",
                        ],
                    ),
                    "wrist": dict(
                        random_brightness=[0.05],
                        random_contrast=[0.95, 1.05],
                        random_saturation=[0.95, 1.05],
                        augment_order=[
                            "random_brightness",
                            "random_contrast",
                            "random_saturation",
                        ],
                    ),
                    "interleaved_instruction": dict(
                        random_brightness=[0.05],
                        random_contrast=[0.95, 1.05],
                        random_saturation=[0.95, 1.05],
                        augment_order=[
                            "random_brightness",
                            "random_contrast",
                            "random_saturation",
                        ],
                    ),
                },
                resize_size=dict(
                    primary=(224, 224),
                    wrist=(224, 224),
                    interleaved_instruction=(224, 224),
                ),
                num_parallel_calls=config.num_parallel_calls,
            ),
            traj_transform_threads=config.traj_transform_threads,
            traj_read_threads=config.traj_read_threads,
        )

        # convert for torch
        self.dataset = TorchRLDSDataset(
            dataset,
            train=train,
            balance_by_task=balance_by_task,
            task_sample_counts=config.get(
                "task_sample_counts",
                None,
            ),
            samples_per_task=config.get(
                "samples_per_task",
                None,
            ),
            seed=config.get(
                "balance_seed",
                42,
            ),
        )
