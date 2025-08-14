# ! encoding:UTF-8
import cv2
import numpy as np
import torch


def get_coord(width: int = 128, device: str = "cuda:0") -> torch.Tensor:
    coord = torch.zeros([1, 2, width, width])
    for i in range(width):
        for j in range(width):
            coord[0, 0, i, j] = i / (width - 1.0)
            coord[0, 1, i, j] = j / (width - 1.0)
    coord = coord.to(device)
    return coord


def normal(x: float, width: int) -> int:
    return int(x * (width - 1) + 0.5)


def gray_div_01_tensor(x: torch.Tensor) -> torch.Tensor:
    return torch.round(x * 10) / 10


def Decoder_cv(f: np.ndarray, width: int = 128, device: str = "cuda:0") -> torch.Tensor:
    stroke = []
    for action in f:
        x0, y0, x1, y1, x2, y2, z0, z2 = action
        x1 = x0 + (x2 - x0) * x1
        y1 = y0 + (y2 - y0) * y1
        x0 = normal(x0, width * 2)
        x1 = normal(x1, width * 2)
        x2 = normal(x2, width * 2)
        y0 = normal(y0, width * 2)
        y1 = normal(y1, width * 2)
        y2 = normal(y2, width * 2)
        z0 = int(1 + z0 * width // 2)
        z2 = int(1 + z2 * width // 2)
        canvas = np.ones([width * 2, width * 2]).astype("float32") * 255
        rate = 1000
        tmp = 1.0 / rate
        w = 0
        for i in range(rate):
            t = i * tmp
            x = int((1 - t) * (1 - t) * x0 + 2 * t * (1 - t) * x1 + t * t * x2)
            y = int((1 - t) * (1 - t) * y0 + 2 * t * (1 - t) * y1 + t * t * y2)
            z = int((1 - t) * z0 + t * z2)
            cv2.circle(canvas, (x, y), radius=z, color=w, thickness=-1)
        result = cv2.resize(canvas, dsize=(width, width))
        result = result.astype("float32") / 255
        result = np.round(result)
        stroke.append(result)
    stroke = np.array(stroke).astype("float32")
    stroke = torch.from_numpy(stroke).to(device)
    return stroke


def decode(
    x: torch.Tensor, canvas: torch.Tensor, device: str = "cuda:0"
) -> torch.Tensor:
    x = x.view(-1, 9)
    f = x[:, :8]
    color = x[:, -1:]
    color = gray_div_01_tensor(color)
    canvas = gray_div_01_tensor(canvas)
    d = torch.round(Decoder_cv(f.detach().cpu().numpy(), device=device))
    stroke = 1 - d
    stroke = stroke.view(-1, 128, 128, 1)
    color_stroke = stroke * color.view(-1, 1, 1, 1)
    stroke = stroke.permute(0, 3, 1, 2)
    color_stroke = color_stroke.permute(0, 3, 1, 2)
    stroke = stroke.view(-1, 1, 1, 128, 128)
    color_stroke = color_stroke.view(-1, 1, 1, 128, 128)
    for i in range(1):
        canvas = canvas * (1 - stroke[:, i])
        canvas = canvas + color_stroke[:, i]
    return canvas


def decode_list(
    x: torch.Tensor, canvas: torch.Tensor, device: str = "cuda:0"
) -> tuple[torch.Tensor, list]:
    canvas = decode(x, canvas, device=device)
    res = [canvas]
    return canvas, res


def small2large(x: np.ndarray, divide: int, width: int = 128) -> np.ndarray:
    x = x.reshape(divide, divide, width, width, -1)
    x = np.transpose(x, (0, 2, 1, 3, 4))
    x = x.reshape(divide * width, divide * width, -1)
    return x


def large2small(
    x: np.ndarray, canvas_cnt: int, divide: int, width: int = 128
) -> np.ndarray:
    x = x.reshape(divide, width, divide, width, 1)
    x = np.transpose(x, (0, 2, 1, 3, 4))
    x = x.reshape(canvas_cnt, width, width, 1)
    return x


def smooth(img: np.ndarray, divide: int, width: int) -> np.ndarray:
    def smooth_pix(img, tx, ty):
        if tx == divide * width - 1 or ty == divide * width - 1 or tx == 0 or ty == 0:
            return img
        img[tx, ty] = (
            img[tx, ty]
            + img[tx + 1, ty]
            + img[tx, ty + 1]
            + img[tx - 1, ty]
            + img[tx, ty - 1]
            + img[tx + 1, ty - 1]
            + img[tx - 1, ty + 1]
            + img[tx - 1, ty - 1]
            + img[tx + 1, ty + 1]
        ) / 9
        return img

    for p in range(divide):
        for q in range(divide):
            x = p * width
            y = q * width
            for k in range(width):
                img = smooth_pix(img, x + k, y + width - 1)
                if q != divide - 1:
                    img = smooth_pix(img, x + k, y + width)
            for k in range(width):
                img = smooth_pix(img, x + width - 1, y + k)
                if p != divide - 1:
                    img = smooth_pix(img, x + width, y + k)
    return img


def save_img(
    res: torch.Tensor,
    imgid: int,
    divide_number: int,
    width: int,
    origin_shape: tuple,
    divide: bool = False,
) -> None:
    output = res.detach().cpu().numpy()
    output = np.transpose(output, (0, 2, 3, 1))
    if divide:
        output = small2large(output, divide_number, width)
        output = smooth(output, width=width, divide=divide_number)
    else:
        output = output[0]
    output = (output * 255).astype("uint8")
    output = cv2.resize(output, origin_shape)
    path = "output/" + str(imgid) + ".png"
    cv2.imwrite(path, output)
    print(path)


def binarize(img: np.ndarray) -> np.ndarray:
    (h, w) = img.shape
    img = img.astype("float32") / 255
    img = np.around(img, 1) * 255
    img = img.astype("uint8")
    img = np.require(img, dtype="f4", requirements=["O", "W"])
    for j in range(w):
        for i in range(h):
            pix = img[i, j]
            if pix >= 200:
                img[i, j] = 255
            if pix <= 50:
                img[i, j] = 0
    return img


def gray_div(img: np.ndarray) -> np.ndarray:
    img = img.astype("float32") / 255
    img = np.around(img, 1) * 255
    img = img.astype("uint8")
    return img


def img2patch(div_num: int, width: int = 128) -> list:
    coord_x = np.ones([div_num, div_num])
    coord_y = np.ones([div_num, div_num])
    for i in range(div_num):
        for j in range(div_num):
            coord_y[i][j] = i * (width - 1)
            coord_x[i][j] = j * (width - 1)
    return [coord_x, coord_y]


def Gray_to_Hex(gray: int) -> str:
    RGB = [gray, gray, gray]
    color = "#"
    for i in RGB:
        num = int(i)
        color += str(hex(num))[-2:].replace("x", "0").upper()
    return color
