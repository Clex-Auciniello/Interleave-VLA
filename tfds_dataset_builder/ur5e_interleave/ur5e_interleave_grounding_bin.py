from typing import Iterator, Tuple, Any

import os
import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds
import re
from scipy.spatial.transform import Rotation
from PIL import Image, ImageDraw

from .conversion_utils import MultiThreadedDatasetBuilder, resize

tfds.core.utils.gcs_utils._is_gcs_disabled = True # Add this line to prevent `tfds build` from accessing google cloud storage

IMAGE_PLACEHOLDER = "<image>"
sample_image_num = 2
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

BIN_HIGHLIGHT_COLOR = (180, 0, 255)
BIN_GROUP_MARGIN_RATIO = 0.10
BIN_BBOX_KEYS = (
    "single_bin_0",
    "single_bin_1",
    "single_bin_2",
    "single_bin_3",
)

BIN_HIGHLIGHT_LINE_WIDTH = 2
BIN_CENTER_RADIUS = 4
BIN_OUTPUT_SIZE = 224
BIN_PADDING_COLOR = (184, 167, 72)

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

BIN_TO_BBOX = {
    "first bin": "single_bin_0",
    "second bin": "single_bin_1",
    "third bin": "single_bin_2",
    "fourth bin": "single_bin_3",
}

BIN_PATTERN = re.compile(
    r"\b(?:first|second|third|fourth)\s+bin\b",
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

def _extract_bin_grounding_image(
    front_image,
    bounding_boxes,
    target_bin_bbox_key,
    episode_id,
):
    """
    Build the visual grounding image for the place target.

    The image is extracted from the initial front-camera frame.

    Procedure:
    1. Read the bounding boxes of all four bins.
    2. Verify that they are ordered from left to right.
    3. Compute the union bounding box containing all bins.
    4. Expand the union box by BIN_GROUP_MARGIN_RATIO.
    5. Crop the rectangular four-bin region.
    6. Resize the crop while preserving its aspect ratio.
    7. Center it inside a 224x224 light-brown canvas.
    8. Draw a purple bounding box and center marker on the target bin.

    The output therefore preserves the geometry of the bins and provides
    an explicit spatial reference for the place destination.
    """

    # ---------------------------------------------------------
    # 1. Read and validate all four bin bounding boxes.
    # ---------------------------------------------------------

    bin_boxes = {}

    for bbox_key in BIN_BBOX_KEYS:
        if bbox_key not in bounding_boxes:
            raise ValueError(
                f"Missing bbox '{bbox_key}' in initial frame "
                f"of episode {episode_id}."
            )

        bbox = bounding_boxes[bbox_key]

        upper_left = _to_float32_array(
            bbox["upper_left_corner"]
        )
        bottom_right = _to_float32_array(
            bbox["bottom_right_corner"]
        )
        center = _to_float32_array(
            bbox["center"]
        )

        if upper_left.shape != (2,):
            raise ValueError(
                f"Invalid upper_left_corner shape for '{bbox_key}' "
                f"in episode {episode_id}: {upper_left.shape}"
            )

        if bottom_right.shape != (2,):
            raise ValueError(
                f"Invalid bottom_right_corner shape for '{bbox_key}' "
                f"in episode {episode_id}: {bottom_right.shape}"
            )

        if center.shape != (2,):
            raise ValueError(
                f"Invalid center shape for '{bbox_key}' "
                f"in episode {episode_id}: {center.shape}"
            )

        xa, ya = upper_left
        xb, yb = bottom_right

        x_min = float(min(xa, xb))
        y_min = float(min(ya, yb))
        x_max = float(max(xa, xb))
        y_max = float(max(ya, yb))

        if x_max <= x_min or y_max <= y_min:
            raise ValueError(
                f"Invalid bbox '{bbox_key}' in episode {episode_id}: "
                f"({x_min}, {y_min}, {x_max}, {y_max})"
            )

        bin_boxes[bbox_key] = {
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
            "center": center,
        }

    if target_bin_bbox_key not in bin_boxes:
        raise ValueError(
            f"Target bin '{target_bin_bbox_key}' is not one of "
            f"{BIN_BBOX_KEYS} in episode {episode_id}."
        )

    # ---------------------------------------------------------
    # 2. Sanity check: bins must be ordered left -> right.
    # ---------------------------------------------------------

    center_x = [
        float(bin_boxes[key]["center"][0])
        for key in BIN_BBOX_KEYS
    ]

    if not all(
        center_x[i] < center_x[i + 1]
        for i in range(len(center_x) - 1)
    ):
        raise ValueError(
            f"Unexpected bin order in episode {episode_id}. "
            f"Expected single_bin_0 -> single_bin_3 from left to right, "
            f"got center x coordinates {center_x}."
        )

    # ---------------------------------------------------------
    # 3. Compute union bounding box of all four bins.
    # ---------------------------------------------------------

    group_x_min = min(
        box["x_min"] for box in bin_boxes.values()
    )
    group_y_min = min(
        box["y_min"] for box in bin_boxes.values()
    )
    group_x_max = max(
        box["x_max"] for box in bin_boxes.values()
    )
    group_y_max = max(
        box["y_max"] for box in bin_boxes.values()
    )

    group_width = group_x_max - group_x_min
    group_height = group_y_max - group_y_min

    image_height, image_width = front_image.shape[:2]

    # ---------------------------------------------------------
    # 4. Add a small margin around the complete bin group.
    # ---------------------------------------------------------

    x_margin = group_width * BIN_GROUP_MARGIN_RATIO
    y_margin = group_height * BIN_GROUP_MARGIN_RATIO

    crop_x_min = int(
        max(0, np.floor(group_x_min - x_margin))
    )
    crop_y_min = int(
        max(0, np.floor(group_y_min - y_margin))
    )
    crop_x_max = int(
        min(image_width, np.ceil(group_x_max + x_margin))
    )
    crop_y_max = int(
        min(image_height, np.ceil(group_y_max + y_margin))
    )

    if crop_x_max <= crop_x_min or crop_y_max <= crop_y_min:
        raise ValueError(
            f"Invalid four-bin crop in episode {episode_id}: "
            f"({crop_x_min}, {crop_y_min}, "
            f"{crop_x_max}, {crop_y_max})"
        )

    # ---------------------------------------------------------
    # 5. Extract the rectangular four-bin crop.
    # ---------------------------------------------------------

    bin_crop = front_image[
        crop_y_min:crop_y_max,
        crop_x_min:crop_x_max
    ]

    if bin_crop.size == 0:
        raise ValueError(
            f"Empty four-bin crop in episode {episode_id}."
        )

    original_crop_height, original_crop_width = bin_crop.shape[:2]

    # ---------------------------------------------------------
    # 6. Resize while preserving aspect ratio.
    # ---------------------------------------------------------

    scale = min(
        BIN_OUTPUT_SIZE / original_crop_width,
        BIN_OUTPUT_SIZE / original_crop_height,
    )

    resized_width = max(
        1,
        int(round(original_crop_width * scale)),
    )
    resized_height = max(
        1,
        int(round(original_crop_height * scale)),
    )

    resized_bin_crop = resize(
        bin_crop,
        target_size=(resized_width, resized_height),
    )

    resized_bin_crop = np.asarray(
        resized_bin_crop,
        dtype=np.uint8,
    )

    remaining_x = BIN_OUTPUT_SIZE - resized_width
    remaining_y = BIN_OUTPUT_SIZE - resized_height

    if remaining_x < 0 or remaining_y < 0:
        raise ValueError(
            f"Invalid resized bin crop size in episode {episode_id}: "
            f"{resized_width}x{resized_height}"
        )

    # ---------------------------------------------------------
    # 7. Center the crop on a fixed light-brown 224x224 canvas.
    # ---------------------------------------------------------

    pad_left = remaining_x // 2
    pad_right = remaining_x - pad_left

    # Vertical padding: put all extra space above the bins,
    # so that the bins occupy the lower horizontal band.
    pad_top = remaining_y
    pad_bottom = 0

    resized_crop = np.full(
        (
            BIN_OUTPUT_SIZE,
            BIN_OUTPUT_SIZE,
            3,
        ),
        BIN_PADDING_COLOR,
        dtype=np.uint8,
    )

    resized_crop[
        pad_top:pad_top + resized_height,
        pad_left:pad_left + resized_width,
    ] = resized_bin_crop

    if resized_crop.shape != (
        BIN_OUTPUT_SIZE,
        BIN_OUTPUT_SIZE,
        3,
    ):
        raise ValueError(
            f"Unexpected padded crop shape in episode {episode_id}: "
            f"{resized_crop.shape}"
        )

    # ---------------------------------------------------------
    # 8. Map target-bin coordinates into the padded image.
    # ---------------------------------------------------------

    target_box = bin_boxes[target_bin_bbox_key]

    target_x_min = int(round(
        pad_left
        + (target_box["x_min"] - crop_x_min) * scale
    ))

    target_y_min = int(round(
        pad_top
        + (target_box["y_min"] - crop_y_min) * scale
    ))

    target_x_max = int(round(
        pad_left
        + (target_box["x_max"] - crop_x_min) * scale
    ))

    target_y_max = int(round(
        pad_top
        + (target_box["y_max"] - crop_y_min) * scale
    ))

    target_center_x = int(round(
        pad_left
        + (
            float(target_box["center"][0])
            - crop_x_min
        ) * scale
    ))

    target_center_y = int(round(
        pad_top
        + (
            float(target_box["center"][1])
            - crop_y_min
        ) * scale
    ))

    # Clip all drawing coordinates to the output image.
    max_coord = BIN_OUTPUT_SIZE - 1

    target_x_min = int(
        np.clip(target_x_min, 0, max_coord)
    )
    target_y_min = int(
        np.clip(target_y_min, 0, max_coord)
    )
    target_x_max = int(
        np.clip(target_x_max, 0, max_coord)
    )
    target_y_max = int(
        np.clip(target_y_max, 0, max_coord)
    )

    target_center_x = int(
        np.clip(target_center_x, 0, max_coord)
    )
    target_center_y = int(
        np.clip(target_center_y, 0, max_coord)
    )

    # ---------------------------------------------------------
    # 9. Draw the target annotation after resizing/padding.
    # ---------------------------------------------------------

    annotated_image = Image.fromarray(
        resized_crop,
        mode="RGB",
    )

    draw = ImageDraw.Draw(annotated_image)

    # Purple target-bin bounding box.
    draw.rectangle(
        [
            (target_x_min, target_y_min),
            (target_x_max, target_y_max),
        ],
        outline=BIN_HIGHLIGHT_COLOR,
        width=BIN_HIGHLIGHT_LINE_WIDTH,
    )

    # Purple target-bin center marker.
    draw.ellipse(
        [
            (
                target_center_x - BIN_CENTER_RADIUS,
                target_center_y - BIN_CENTER_RADIUS,
            ),
            (
                target_center_x + BIN_CENTER_RADIUS,
                target_center_y + BIN_CENTER_RADIUS,
            ),
        ],
        fill=BIN_HIGHLIGHT_COLOR,
    )

    return np.asarray(
        annotated_image,
        dtype=np.uint8,
    )

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

        bin_match = BIN_PATTERN.search(original_instruction)

        if bin_match is None:
            raise ValueError(
                f"Cannot identify target bin in instruction: "
                f"{original_instruction!r}"
            )

        bin_name = re.sub(
            r"\s+",
            " ",
            bin_match.group(0).lower(),
        )

        bin_bbox_key = BIN_TO_BBOX[bin_name]

        language_instruction = BIN_PATTERN.sub(
            IMAGE_PLACEHOLDER,
            language_instruction,
            count=1,
        )
        
        if language_instruction.count(IMAGE_PLACEHOLDER) != 2:
            raise ValueError(
                f"Expected exactly two {IMAGE_PLACEHOLDER} placeholders in "
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

        bin_grounding_image = _extract_bin_grounding_image(
            front_image=initial_front_image,
            bounding_boxes=first_step["observation"]["bounding_boxes"],
            target_bin_bbox_key=bin_bbox_key,
            episode_id=global_episode_index,
        )

        # Fixed multimodal instruction for the whole episode.
        image_instruction = [target_crop, bin_grounding_image]
        image_mask = [True, True]

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


class Ur5eInterleaveGroundingBin(MultiThreadedDatasetBuilder):
    """DatasetBuilder for the interleaved UR5e pick-and-place dataset."""

    # VERSION = tfds.core.Version('1.0.0')
    # RELEASE_NOTES = {
    #   '1.0.0': 'Initial release.',
    # }
    VERSION = tfds.core.Version('0.2.0')
    RELEASE_NOTES = {
      "0.2.0": "Bin-grounding dataset preserving scaled action representation.",
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
                            doc='Language instruction with exactly two <image> placeholders.'
                        ),
                        'original_instruction': tfds.features.Text(
                            doc='Language Instruction, without placeholders <image>.'
                        ),
                        'image_instruction': tfds.features.Sequence(
                            tfds.features.Image(
                                shape=(224, 224, 3),
                                dtype=np.uint8,
                                encoding_format='jpeg',
                                doc=(
                                    'Interleaved instruction images. '
                                    'For the UR5e bin-grounding dataset the sequence contains '
                                    'the target object crop followed by the spatial bin reference.'
                                )
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