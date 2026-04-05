"""
VIMA Interleaved Dataset Generator

Collects oracle demonstrations from vima_bench environments and saves them as
.npy files with interleaved image-text instructions, ready for TFDS conversion.

Based on data generation scripts from the Chameleon-VLA project (se2 branch).

Usage:
    # Generate 10 episodes for scene_understanding task (sample run):
    python generate_vima_data.py \
        save_path="vima_dataset_se2" \
        num_episodes_per_task=10 \
        seed=0 \
        parallel=false \
        task_selection='["scene_understanding"]'

    # Generate full dataset across all tasks:
    python generate_vima_data.py \
        save_path="vima_dataset_se2" \
        num_episodes_per_task=50000 \
        seed=0 \
        parallel=true
"""
from __future__ import annotations

import logging
import multiprocessing
import os
import re
import signal
from math import ceil

import cv2
import hydra
import numpy as np
from einops import rearrange
from PIL import Image
from tqdm import tqdm

from utils.prepare import prepare_prompt_images

import vima_bench
import vima_bench.utils as U
from vima_bench import PARTITION_TO_SPECS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

IMAGE_PLACEHOLDER = "<image>"
MAX_TRIES_PER_SEED = 999


def _generate_data_for_one_task(
    task_name: str,
    task_kwargs: dict | None,
    modalities,
    num_episodes: int,
    success_only: bool,
    save_path: str,
    num_save_digits: int,
    start_id: int = 0,
    seed: int | None = None,
):
    save_path = U.f_join(save_path, task_name)
    os.makedirs(save_path, exist_ok=True)

    tbar = tqdm(total=num_episodes, desc=task_name, leave=True)
    n_generated = 0
    num_tried_this_seed = 0

    env = vima_bench.make(
        task_name=task_name, task_kwargs=task_kwargs, modalities=modalities, seed=seed,
        display_debug_window=False, hide_arm_rgb=False,
    )
    task = env.task
    oracle_fn = task.oracle(env)

    # ========= signal handler =============
    def signal_handler(sig, frame):
        env.close()
        cv2.destroyAllWindows()
        logger.info("Exited safely.")
        exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    # =====================================

    while True:
        try:
            env.seed(seed + start_id + n_generated)
            num_tried_this_seed += 1

            obs_cache = []
            action_cache = []

            obs = env.reset()
            obs_cache.append(obs)
            elapsed_steps = 0
            meta, prompt, prompt_assets = env.meta_info, env.prompt, env.prompt_assets

            oracle_failed = False
            for _ in range(task.oracle_max_steps):
                oracle_action = oracle_fn.act(obs)
                if oracle_action is None:
                    logger.warning("No oracle action, skipping episode.")
                    oracle_failed = True
                    break
                # clip action to action space bounds
                oracle_action = {
                    k: np.clip(v, env.action_space[k].low, env.action_space[k].high)
                    for k, v in oracle_action.items()
                }
                obs, _, done, info = env.step(action=oracle_action, skip_oracle=False)
                obs_cache.append(obs)
                action_cache.append(oracle_action)
                elapsed_steps += 1
                if done:
                    break
            if oracle_failed:
                seed += 1
                num_tried_this_seed = 0
                continue
        except ValueError as e:
            logger.warning(f"ValueError: {e}")
            seed += 1
            num_tried_this_seed = 0
            continue

        assert len(obs_cache) == len(action_cache) + 1 == elapsed_steps + 1
        if success_only and not info["success"]:
            if num_tried_this_seed >= MAX_TRIES_PER_SEED:
                seed += 1
                num_tried_this_seed = 0
            continue

        # === Build interleaved episode and save as .npy ===
        episode_save_path = U.f_join(save_path, f"{start_id + n_generated:0{num_save_digits}d}")

        obs = U.stack_sequence_fields(obs_cache)
        action = U.stack_sequence_fields(action_cache)
        assert U.get_batch_size(obs) == U.get_batch_size(action) + 1, "INTERNAL"

        rgb = obs.pop("rgb")
        views = sorted(rgb.keys())
        frames = {view: rearrange(rgb[view], "t c h w -> t h w c") for view in views}

        # Build interleaved prompt: replace {placeholder} with <image>
        prompt_img_with_pos = []
        for placeholder in prompt_assets.keys():
            prompt_img_with_pos.extend(
                [(m.start(), placeholder) for m in re.finditer(re.escape(placeholder), prompt)]
            )
            prompt = prompt.replace('{' + placeholder + '}', IMAGE_PLACEHOLDER)
        prompt_img_with_pos = sorted(prompt_img_with_pos, key=lambda x: x[0])

        _prompt_assets = prepare_prompt_images(prompt_assets, view='front', image_size=(224, 224))
        prompt_imgs = [
            rearrange(_prompt_assets[pair[1]]['rgb']['front'], "c h w -> h w c")
            for pair in prompt_img_with_pos
        ]

        assert len(prompt_imgs) == prompt.count(IMAGE_PLACEHOLDER), \
            f"Mismatch between language and image instruction lengths. prompt: {prompt}."

        # Build per-step episode data
        episode = []
        for i in range(elapsed_steps):
            step_data = {
                'action': np.concatenate([
                    action['pose0_position'][i], action['pose0_rotation'][i],
                    action['pose1_position'][i], action['pose1_rotation'][i],
                ]),
                'observation': {
                    'image': frames['front'][i],
                },
                'language_instruction': prompt,
                'image_instruction': prompt_imgs,
            }
            episode.append(step_data)

        np.save(episode_save_path, episode)

        n_generated += 1
        num_tried_this_seed = 0
        tbar.update(1)

        if n_generated >= num_episodes:
            break
    tbar.close()


def generate_data_for_one_task(kwargs):
    _generate_data_for_one_task(**kwargs)


@hydra.main(config_path=".", config_name="conf", version_base="1.1")
def main(cfg):
    # pybullet (used by vima_bench) is incompatible with fork-based multiprocessing.
    # Must use 'spawn' for parallel workers to function correctly.
    multiprocessing.set_start_method('spawn', force=True)

    tasks = sorted(list(PARTITION_TO_SPECS["train"].keys()))
    task_selection = cfg.task_selection
    if task_selection is not None:
        if isinstance(task_selection, str):
            task_selection = [task_selection]
        tasks = [task for task in tasks if task in task_selection]

    logger.info(f"Tasks: {tasks}")

    if cfg.parallel:
        max_workers = multiprocessing.cpu_count()
        logger.info(f"Working on {max_workers} cores")
        if max_workers <= len(tasks):
            num_batches = 1
            for i in tqdm(range(num_batches), desc="Parallel"):
                tasks_this_batch = tasks[i * max_workers : (i + 1) * max_workers]
                num_workers = min(len(tasks_this_batch), max_workers)
                with multiprocessing.Pool(num_workers) as pool:
                    pool.map(
                        generate_data_for_one_task,
                        [
                            dict(
                                task_name=t,
                                task_kwargs=PARTITION_TO_SPECS["train"][t],
                                modalities=cfg.modalities,
                                num_episodes=cfg.num_episodes_per_task,
                                success_only=cfg.success_only,
                                save_path=cfg.save_path,
                                num_save_digits=cfg.num_save_digits,
                                start_id=0,
                                seed=cfg.seed,
                            )
                            for t in tasks_this_batch
                        ],
                    )
        else:
            # More workers than tasks: split episodes across workers within each task
            num_workers = max_workers
            episodes_per_worker = ceil(cfg.num_episodes_per_task / num_workers)
            logger.info(
                f"Splitting {cfg.num_episodes_per_task} episodes across "
                f"{num_workers} workers ({episodes_per_worker} eps/worker)"
            )
            job_args = []
            for t in tasks:
                for w in range(num_workers):
                    start_id = w * episodes_per_worker
                    remaining = cfg.num_episodes_per_task - start_id
                    if remaining <= 0:
                        break
                    n_eps = min(episodes_per_worker, remaining)
                    job_args.append(dict(
                        task_name=t,
                        task_kwargs=PARTITION_TO_SPECS["train"][t],
                        modalities=cfg.modalities,
                        num_episodes=n_eps,
                        success_only=cfg.success_only,
                        save_path=cfg.save_path,
                        num_save_digits=cfg.num_save_digits,
                        start_id=start_id,
                        seed=cfg.seed + w * 10000,
                    ))
            logger.info(f"Launching {len(job_args)} worker jobs")
            with multiprocessing.Pool(num_workers) as pool:
                pool.map(generate_data_for_one_task, job_args)
    else:
        for t in tasks:
            _generate_data_for_one_task(
                task_name=t,
                task_kwargs=None,
                modalities=cfg.modalities,
                num_episodes=cfg.num_episodes_per_task,
                success_only=cfg.success_only,
                save_path=cfg.save_path,
                num_save_digits=cfg.num_save_digits,
                seed=cfg.seed,
            )


if __name__ == "__main__":
    main()
