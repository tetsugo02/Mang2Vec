# ! encoding:UTF-8
import os
import time

import cv2
import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from src.models.DRL.actor import ResNet
from src.utils import point2svg as ps
from src.utils import vectorize_utils as vu
from src.utils.decode import Decode_np, del_file
from src.utils.point2svg import Point2svg


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patch_width = cfg.width
    patch_divide = cfg.divide
    output_dir = cfg.output_dir
    patch_count = patch_divide * patch_divide
    use_patch_fill = True
    use_pruning_module = False
    T = torch.ones([1, 1, patch_width, patch_width], dtype=torch.float32).to(device)
    coord_tensor = vu.get_coord(width=patch_width, device=device)
    os.makedirs(output_dir, exist_ok=True)
    del_file(output_dir)

    actor = ResNet(5, 18, 9)
    actor.load_state_dict(torch.load(cfg.actor))
    actor = actor.to(device).eval()

    input_img = cv2.imread(cfg.img, cv2.IMREAD_GRAYSCALE)
    original_img = input_img
    (img_height, img_width) = input_img.shape
    origin_shape = (input_img.shape[1], input_img.shape[0])

    if use_patch_fill:
        canvas_img, patch_status_list = ps.patch_fill(
            img=input_img, div_num=patch_divide
        )
        canvas_img = cv2.resize(
            canvas_img, (patch_width * patch_divide, patch_width * patch_divide)
        ).astype("float32")
        canvas_tensor = torch.from_numpy(canvas_img / 255)
        canvas_tensor = canvas_tensor.unsqueeze(0).unsqueeze(0).to(device)
    else:
        canvas_tensor = torch.ones([1, 1, patch_width, patch_width]).to(device)
        _, patch_status_list = ps.patch_fill(img=input_img, div_num=patch_divide)

    patch_img = cv2.resize(
        input_img, (patch_width * patch_divide, patch_width * patch_divide)
    )
    patch_img = vu.binarize(patch_img)
    patch_img = vu.gray_div(patch_img)
    aspect_ratio = img_height / img_width
    resized_patch_img = cv2.resize(
        patch_img,
        (
            int(patch_width * patch_divide),
            int(patch_width * patch_divide * aspect_ratio),
        ),
    )
    cv2.imwrite(filename=output_dir + "target.png", img=resized_patch_img)
    patch_img = vu.large2small(patch_img, patch_count, patch_divide, patch_width)
    patch_img = np.transpose(patch_img, (0, 3, 1, 2))
    patch_img_tensor = torch.tensor(patch_img).to(device).float() / 255.0

    input_img_resized = cv2.resize(input_img, (patch_width, patch_width))
    input_img_resized = input_img_resized.reshape(1, patch_width, patch_width, 1)
    input_img_resized = np.transpose(input_img_resized, (0, 3, 1, 2))
    # input_img_tensor = torch.tensor(input_img_resized).to(device).float() / 255.0

    os.system("mkdir output")
    p2s = Point2svg(
        width=patch_width,
        div_num=patch_divide,
        save_path=output_dir,
        init_num=0,
        img_w=img_width,
        img_h=img_height,
        img=original_img,
        use_patch_fill=use_patch_fill,
        patch_done_list=patch_status_list,
    )

    # action_list = []

    with torch.no_grad():
        if patch_divide != 0:
            canvas_tensor = canvas_tensor[0].detach().cpu().numpy()
            canvas_tensor = np.transpose(canvas_tensor, (1, 2, 0))
            canvas_tensor = cv2.resize(
                canvas_tensor, (patch_width * patch_divide, patch_width * patch_divide)
            )
            canvas_tensor = vu.large2small(
                canvas_tensor, patch_count, patch_divide, patch_width
            )
            canvas_tensor = np.transpose(canvas_tensor, (0, 3, 1, 2))
            canvas_tensor = torch.tensor(canvas_tensor).to(device).float()
            coord_tensor = coord_tensor.expand(patch_count, 2, patch_width, patch_width)
            T = T.expand(patch_count, 1, patch_width, patch_width)
            vu.save_img(
                canvas_tensor,
                cfg.imgid,
                divide_number=patch_divide,
                width=patch_width,
                origin_shape=origin_shape,
                divide=True,
            )
            imgid = cfg.imgid + 1
            start_time = time.time()
            for step in range(cfg.max_step):
                step_tensor = T * step / cfg.max_step
                actions = actor(
                    torch.cat(
                        [canvas_tensor, patch_img_tensor, step_tensor, coord_tensor], 1
                    )
                )
                p2s.reset_gt_patch(gt=patch_img_tensor)
                canvas_tensor, result = vu.decode_list(actions, canvas_tensor)
                print(
                    f"divided canvas step {step}, Loss = {(canvas_tensor - patch_img_tensor).pow(2).mean()}"
                )
                p2s.add_action_div(actions)
                vu.save_img(
                    canvas_tensor,
                    imgid,
                    divide_number=patch_divide,
                    width=patch_width,
                    origin_shape=origin_shape,
                    divide=True,
                )
                imgid += 1

            end_time_actor = time.time()
            time_draw_action = p2s.draw_action_list_for_all_patch(path_or_circle="path")
            time_decode_start = time.time()
            decoder = Decode_np(div_num=patch_divide, use_PM=use_pruning_module)
            time_decode_end = time.time()
            decoder.draw_decode()
            end_time_total = time.time()

            print(f"actor time is : {end_time_actor - start_time}")
            print(
                f"paint time is : {end_time_total - end_time_actor - time_draw_action - (time_decode_end - time_decode_start)}"
            )


if __name__ == "__main__":
    main()
