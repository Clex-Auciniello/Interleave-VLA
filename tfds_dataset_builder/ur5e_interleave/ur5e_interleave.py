from typing import Iterator, Tuple, Any

import os
import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds
import re
from scipy.spatial.transform import Rotation

from .conversion_utils import MultiThreadedDatasetBuilder, resize

tfds.core.utils.gcs_utils._is_gcs_disabled = True # Add this line to prevent `tfds build` from accessing google cloud storage

IMAGE_PLACEHOLDER = "<image>"
sample_image_num = 1
RAW_DATA_PATH_ENV = "UR5E_RAW_DATA_PATH"

VAL_EPISODE_IDS = {
    1, 7,
    55, 57,
    88, 94,
    126, 154,
    165, 197,
    202, 227,
    241, 245,
    293, 294,
    352, 358,
    361, 395,
    412, 434,
    454, 466,
}

TARGET_TO_BBOX = {
    "red box": "redbox",
    "green box": "greenbox",
    "blue box": "bluebox",
    "yellow box": "yellowbox",
}

TARGET_PATTERN = re.compile(
    r"\b(?:red|green|blue|yellow)\s+box\b",
    flags=re.IGNORECASE,
)

ACTION_SCALE_FACTOR = 0.05

def _to_numpy(value):
    """Convert a TensorFlow tensor to NumPy, leaving other values unchanged."""
    return value.numpy() if hasattr(value, "numpy") else value


def _to_string(value) -> str:
    """Convert TensorFlow/bytes strings to a Python string."""
    value = _to_numpy(value)

    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()

    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")

    return str(value)


def _to_float32_array(value) -> np.ndarray:
    """Convert a numeric tensor/array to NumPy float32."""
    return np.asarray(_to_numpy(value), dtype=np.float32)


def _to_uint8_image(value) -> np.ndarray:
    """Convert an image tensor to NumPy uint8."""
    return np.asarray(_to_numpy(value), dtype=np.uint8)


def _to_scalar(value, dtype):
    """Convert a scalar TensorFlow value to a Python/NumPy scalar."""
    value = np.asarray(_to_numpy(value))

    if value.size != 1:
        raise ValueError(f"Expected scalar value, got shape {value.shape}")

    return dtype(value.item())

def _binary_gripper(value) -> np.float32:
    """Convert gripper value to 0=open, 1=closed."""
    return np.float32(float(value) >= 0.5)


def _get_raw_data_path() -> str:
    """Return the source dataset path configured by the execution environment."""
    raw_data_path = os.environ.get(RAW_DATA_PATH_ENV)
    if not raw_data_path:
        raise RuntimeError(
            f"Environment variable {RAW_DATA_PATH_ENV} is not set. "
            "Set it to the directory containing the source UR5e TFDS dataset."
        )
    return os.path.abspath(os.path.expanduser(raw_data_path))


def _repair_pose_action_raw(current_step, next_step):
    """
    Recompute the pose action connecting current_step -> next_step.

    The returned action remains in the RAW dataset representation,
    i.e. pose deltas divided by ACTION_SCALE_FACTOR.
    The gripper action of current_step is left unchanged.
    """
    state_0 = _to_float32_array(
        current_step["observation"]["EEF_state"]
    ).astype(np.float64)

    state_1 = _to_float32_array(
        next_step["observation"]["EEF_state"]
    ).astype(np.float64)

    if state_0.shape != (6,) or state_1.shape != (6,):
        raise ValueError(
            f"Expected EEF states with shape (6,), got "
            f"{state_0.shape} and {state_1.shape}."
        )

    repaired_action = _to_float32_array(
        current_step["action"]
    ).copy()

    # Translation: p1 = p0 + delta_p
    delta_position = state_1[:3] - state_0[:3]

    # Rotation convention:
    # R1 = R_delta @ R0
    rotation_0 = Rotation.from_euler(
        "xyz", state_0[3:6]
    ).as_matrix()

    rotation_1 = Rotation.from_euler(
        "xyz", state_1[3:6]
    ).as_matrix()

    delta_rotation = rotation_1 @ rotation_0.T

    delta_rpy = Rotation.from_matrix(
        delta_rotation
    ).as_euler("xyz")

    # Restore RAW source representation.
    repaired_action[:3] = (
        delta_position / ACTION_SCALE_FACTOR
    ).astype(np.float32)

    repaired_action[3:6] = (
        delta_rpy / ACTION_SCALE_FACTOR
    ).astype(np.float32)

    # repaired_action[6] intentionally remains unchanged.

    return repaired_action

def _extract_target_crop(
    front_image,
    bounding_boxes,
    target_bbox_key,
    episode_id,
):
    """Extract target crop from the initial frame and resize it to 224x224."""

    if target_bbox_key not in bounding_boxes:
        raise ValueError(
            f"Missing bbox '{target_bbox_key}' in initial frame "
            f"of episode {episode_id}."
        )

    bbox = bounding_boxes[target_bbox_key]

    upper_left = _to_float32_array(
        bbox["upper_left_corner"]
    )
    bottom_right = _to_float32_array(
        bbox["bottom_right_corner"]
    )

    xa, ya = upper_left
    xb, yb = bottom_right

    x_min = int(round(min(xa, xb)))
    y_min = int(round(min(ya, yb)))
    x_max = int(round(max(xa, xb)))
    y_max = int(round(max(ya, yb)))

    height, width = front_image.shape[:2]

    # Expand the bounding box by 20% on each side,
    # following the Interleave-VLA preprocessing.
    box_width = x_max - x_min
    box_height = y_max - y_min

    x_expand = box_width * 0.2
    y_expand = box_height * 0.2

    x_min = int(max(0, x_min - x_expand))
    y_min = int(max(0, y_min - y_expand))
    x_max = int(min(width, x_max + x_expand))
    y_max = int(min(height, y_max + y_expand))

    if x_max <= x_min or y_max <= y_min:
        raise ValueError(
            f"Invalid bbox '{target_bbox_key}' in initial frame "
            f"of episode {episode_id}: "
            f"({x_min}, {y_min}, {x_max}, {y_max})"
        )

    crop = front_image[y_min:y_max, x_min:x_max]

    if crop.size == 0:
        raise ValueError(
            f"Empty crop '{target_bbox_key}' "
            f"in episode {episode_id}."
        )

    crop = resize(crop)

    return np.asarray(crop, dtype=np.uint8)

def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    """Yields episodes for list of data paths."""
    # the line below needs to be *inside* generate_examples so that each worker creates it's own model
    # creating one shared model outside this function would cause a deadlock

    def _parse_example(
        trajectory,
        shard_index,
        global_episode_index,
        source_tfrecord_path,
    ):
        # load raw data
        if "steps" not in trajectory:
            print("\nLa traiettoria non contiene la chiave 'steps'.")
            return

        steps = list(trajectory["steps"])

        original_instruction = _to_string(
            trajectory["language_instruction"]
        )

        target_match = TARGET_PATTERN.search(original_instruction)

        if target_match is None:
            raise ValueError(
                f"Cannot identify target object in instruction: "
                f"{original_instruction!r}"
            )

        target_name = re.sub(
            r"\s+",
            " ",
            target_match.group(0).lower(),
        )

        target_bbox_key = TARGET_TO_BBOX[target_name]

        language_instruction = TARGET_PATTERN.sub(
            IMAGE_PLACEHOLDER,
            original_instruction,
            count=1,
        )
        
        if language_instruction.count(IMAGE_PLACEHOLDER) != 1:
            raise ValueError(
                f"Expected exactly one {IMAGE_PLACEHOLDER} in "
                f"instruction, got: {language_instruction!r}"
            )

        if len(steps) < 2:
            raise ValueError(
                f"Episode {global_episode_index} contains fewer than two steps."
            )

        first_step = steps[0]
        second_step = steps[1]

        penultimate_step = steps[-2]
        last_step = steps[-1]

        # Repair the first transition.
        repaired_first_action_raw = _repair_pose_action_raw(
            first_step,
            second_step,
        )

        # Repair the last transition:
        # state[-2] -> state[-1].
        # Only pose components are recomputed;
        # the original gripper command is preserved.
        repaired_penultimate_action_raw = _repair_pose_action_raw(
            penultimate_step,
            last_step,
        )

        initial_front_image = _to_uint8_image(
            first_step["observation"]["camera_front_image"]
        )

        target_crop = _extract_target_crop(
            front_image=initial_front_image,
            bounding_boxes=first_step["observation"]["bounding_boxes"],
            target_bbox_key=target_bbox_key,
            episode_id=global_episode_index,
        )

        # Fixed multimodal instruction for the whole episode.
        image_instruction = [target_crop]
        image_mask = [True]

        episode_key = (
            f"shard_{shard_index:02d}_"
            f"episode_{global_episode_index:06d}"
        )
        

        episode = []

        for i, step in enumerate(steps):
            eef_state = _to_float32_array(
                step["observation"]["EEF_state"]
            )

            gripper_state = _to_float32_array(
                step["observation"]["gripper_state"]
            )

            if i == 0:
                # Fix first transition: state[0] -> state[1].
                action_raw = repaired_first_action_raw.copy()

            elif i == len(steps) - 2:
                # Fix final transition: state[-2] -> state[-1].
                action_raw = repaired_penultimate_action_raw.copy()

            else:
                action_raw = _to_float32_array(
                    step["action"]
                )

            assert eef_state.shape == (6,), (
                f"EEF state shape should be (6,), got {eef_state.shape}"
            )

            assert gripper_state.shape == (2,), (
                f"Gripper state shape should be (2,), got {gripper_state.shape}"
            )

            assert action_raw.shape == (7,), (
                f"Action shape should be (7,), got {action_raw.shape}"
            )

            # Absolute robot state:
            # [x, y, z, roll, pitch, yaw, gripper]
            state = np.concatenate([
                eef_state,
                np.asarray(
                    [_binary_gripper(gripper_state[1])],
                    dtype=np.float32,
                ),
            ]).astype(np.float32)

            # Recover physical delta action.
            action = np.empty(7, dtype=np.float32)
            action[:6] = action_raw[:6] * ACTION_SCALE_FACTOR

            scaled_gripper_action = (
                action_raw[6] * ACTION_SCALE_FACTOR
            )
            action[6] = _binary_gripper(scaled_gripper_action)

            camera_image = resize(
                _to_uint8_image(
                    step["observation"]["camera_front_image"]
                )
            )

            gripper_image = resize(
                _to_uint8_image(
                    step["observation"]["camera_gripper_image"]
                )
            )
            
            
            
            episode.append({
                'observation': {
                    'image_0': camera_image, # consistent with configs.py in openvla
                    'image_1': gripper_image,
                    'state': state,
                },
                'action': action,
                'discount': _to_scalar(
                    step["discount"]["discount"],
                    np.float32,
                ),
                'reward': _to_scalar(
                    step["reward"]["reward"],
                    np.float32,
                ),
                'is_first': _to_scalar(
                    step["is_first"],
                    np.bool_,
                ),
                'is_last': _to_scalar(
                    step["is_last"],
                    np.bool_,
                ),
                'is_terminal': _to_scalar(
                    step["is_terminal"],
                    np.bool_,
                ),
                'interleaved_instruction': {
                    'language_instruction': language_instruction,
                    'original_instruction': original_instruction,
                    'image_instruction': image_instruction,
                    'image_mask': image_mask
                }
            })
            # ======================= DEBUG =================================
            # from PIL import Image
            # print(episode[-1])
            # Image.fromarray(episode[-1]['observation']['image_0']).save("obs.jpg")
            # for i, img in enumerate(episode[-1]['interleaved_instruction']['image_instruction']):
            #     Image.fromarray(img).save(f"{i}.jpg")
            # exit(0)
            
        # create output data sample
        sample = {
            'steps': episode,
            'episode_metadata': {
                'file_path': source_tfrecord_path,
                'task_id': np.int32(shard_index),
            }
        }

        return episode_key, sample


    dataset_dir = _get_raw_data_path()
    builder = tfds.builder_from_directory(str(dataset_dir))

    source_split = "train"
    # `paths` contiene le unità di lavoro assegnate a questo worker:
    # (shard_index, global_start, global_stop)
    for shard_index, start, stop, source_tfrecord_path in paths:
        shard_length = stop - start

        shard_dataset = builder.as_dataset(
            split=f"{source_split}[{start}:{stop}]",
            shuffle_files=False,
        )

        for trajectory_index, trajectory in enumerate(shard_dataset):
            global_episode_index = start + trajectory_index

            print(
                f"Parsing shard {shard_index + 1}, "
                f"trajectory {trajectory_index + 1}/{shard_length}"
            )

            ret = _parse_example(
                trajectory,
                shard_index=shard_index,
                global_episode_index=global_episode_index,
                source_tfrecord_path=source_tfrecord_path,
            )

            if ret is not None:
                yield ret


class Ur5eInterleave(MultiThreadedDatasetBuilder):
    """DatasetBuilder for the interleaved UR5e pick-and-place dataset."""

    # VERSION = tfds.core.Version('1.0.0')
    # RELEASE_NOTES = {
    #   '1.0.0': 'Initial release.',
    # }
    VERSION = tfds.core.Version('0.1.0')
    RELEASE_NOTES = {
      '0.1.0': 'Initial UR5e interleaved dataset builder.',
    }
    N_WORKERS = 4             # number of parallel workers for data conversion
    MAX_PATHS_IN_MEMORY = 4  # number of paths converted & stored in memory before writing to disk
                               # -> the higher the faster / more parallel conversion, adjust based on avilable RAM
                               # note that one path may yield multiple episodes and adjust accordingly
    PARSE_FCN = _generate_examples      # handle to parse function from file paths to RLDS episodes

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image_0': tfds.features.Image(
                            shape=(224, 224, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Main camera RGB observation.',
                        ),
                        'image_1': tfds.features.Image(
                            shape=(224, 224, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Secondary camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc=(
                                'Current UR5e end-effector state in base_link frame: '
                                '[x, y, z, roll, pitch, yaw, gripper]. '
                                'Position is expressed in meters, orientation in radians '
                                '(XYZ Euler/RPY), gripper uses 0=open and 1=closed.'
                            ),
                        ),
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc=(
                            'UR5e end-effector delta action in base_link frame: '
                            '[dx, dy, dz, droll, dpitch, dyaw, gripper]. '
                            'Translation deltas are expressed in meters, rotation deltas '
                            'in radians as XYZ Euler/RPY, gripper uses 0=open and 1=closed.'
                        ),
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'interleaved_instruction': tfds.features.FeaturesDict({
                        'language_instruction': tfds.features.Text(
                            doc='Language Instruction, with placeholders <image>.'
                        ),
                        'original_instruction': tfds.features.Text(
                            doc='Language Instruction, without placeholders <image>.'
                        ),
                        'image_instruction': tfds.features.Sequence(
                            tfds.features.Image(
                                shape=(224, 224, 3),
                                dtype=np.uint8,
                                encoding_format='jpeg',
                                doc='Interleaved instruction images.'
                            ),
                            #length=sample_image_num,
                            doc="Image sequence."
                        ),
                        'image_mask': tfds.features.Sequence(
                            tfds.features.Scalar(
                                dtype=np.bool_,
                                doc='Mask indicating whether the image is real (True) or padded (False)'
                            ),
                            #length=sample_image_num,
                            doc="Image mask sequence."
                        )
                    })
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the source TFRecord shard.'
                    ),
                    'task_id': tfds.features.Scalar(
                        dtype=np.int32,
                        doc=(
                            'Task identifier. Corresponds to the source shard/task '
                            'and ranges from 0 to 11 for the current dataset.'
                        ),
                    ),
                }),
            }))

    def _split_paths(self):
        """Define source TFDS episode ranges to be converted."""
        raw_data_path = _get_raw_data_path()
        source_builder = tfds.builder_from_directory(raw_data_path)

        if "train" not in source_builder.info.splits:
            raise RuntimeError(
                "Source UR5e dataset does not contain the 'train' split."
            )

        split_info = source_builder.info.splits["train"]
        shard_lengths = list(split_info.shard_lengths)

        source_tfrecord_paths = sorted(
            os.path.join(raw_data_path, filename)
            for filename in os.listdir(raw_data_path)
            if "-train.tfrecord-" in filename
        )

        if len(source_tfrecord_paths) != len(shard_lengths):
            raise RuntimeError(
                f"TFRecord/metadata mismatch: found "
                f"{len(source_tfrecord_paths)} TFRecord files but "
                f"{len(shard_lengths)} shard entries."
            )

        if len(shard_lengths) != 12:
            print(f"Expected 12 TFRecord shards, found {len(shard_lengths)}.")
        if any(shard_length != 40 for shard_length in shard_lengths):
            print(f"Expected 40 trajectories per shard, found {shard_lengths}.")

        train_work_units = []
        val_work_units = []

        start = 0

        for shard_index, shard_length in enumerate(shard_lengths):
            stop = start + shard_length
            source_tfrecord_path = source_tfrecord_paths[shard_index]

            shard_val_ids = sorted(
                episode_id
                for episode_id in VAL_EPISODE_IDS
                if start <= episode_id < stop
            )

            cursor = start

            for val_episode_id in shard_val_ids:

                # Intervallo train prima dell'episodio di validation.
                if cursor < val_episode_id:
                    train_work_units.append(
                        (
                            shard_index,
                            cursor,
                            val_episode_id,
                            source_tfrecord_path,
                        )
                    )

                # Il singolo episodio riservato alla validation.
                val_work_units.append(
                    (
                        shard_index,
                        val_episode_id,
                        val_episode_id + 1,
                        source_tfrecord_path,
                    )
                )

                cursor = val_episode_id + 1

            # Eventuale intervallo train dopo l'ultimo episodio di validation.
            if cursor < stop:
                train_work_units.append(
                    (
                        shard_index,
                        cursor,
                        stop,
                        source_tfrecord_path,
                    )
                )

            start = stop

        return {
            "train": train_work_units,
            "val": val_work_units,
        }