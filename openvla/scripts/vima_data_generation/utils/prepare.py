import cv2
import numpy as np
from einops import rearrange

def prepare_prompt_images(prompt_assets: dict, view: str = 'front', image_size: int = (224, 224)):
    prompt_imgs = prompt_assets.copy()
    for asset_name, asset in prompt_assets.items():
        obj_info = asset["segm"]["obj_info"]
        placeholder_type = asset["placeholder_type"]
        rgb_this_view = asset["rgb"][view]
        segm_this_view = asset["segm"][view]
        if placeholder_type == "object":
            obj_id = obj_info["obj_id"]
            ys, xs = np.nonzero(segm_this_view == obj_id)
            if len(xs) < 2 or len(ys) < 2:
                continue
            xmin, xmax = np.min(xs), np.max(xs)
            ymin, ymax = np.min(ys), np.max(ys)
            cropped_img = rgb_this_view[:, ymin : ymax + 1, xmin : xmax + 1]
            if cropped_img.shape[1] != cropped_img.shape[2]:
                diff = abs(cropped_img.shape[1] - cropped_img.shape[2])
                pad_before, pad_after = int(diff / 2), diff - int(diff / 2)
                if cropped_img.shape[1] > cropped_img.shape[2]:
                    pad_width = ((0, 0), (0, 0), (pad_before, pad_after))
                else:
                    pad_width = ((0, 0), (pad_before, pad_after), (0, 0))
                cropped_img = np.pad(
                    cropped_img,
                    pad_width,
                    mode="constant",
                    constant_values=255,
                )
                assert cropped_img.shape[1] == cropped_img.shape[2], "INTERNAL"
        elif placeholder_type == "scene":
            cropped_img = rgb_this_view
        else:
            raise ValueError(f"Unknown placeholder type: {placeholder_type}")
        cropped_img = rearrange(cropped_img, "c h w -> h w c")
        cropped_img = np.asarray(cropped_img)
        cropped_img = cv2.resize(
            cropped_img,
            image_size,
            interpolation=cv2.INTER_AREA,
        )
        cropped_img = rearrange(cropped_img, "h w c -> c h w")
        prompt_imgs[asset_name]["rgb"][view] = cropped_img
    return prompt_imgs
