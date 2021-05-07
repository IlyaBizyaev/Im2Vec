import argparse
import glob
import os

from PIL import Image

import torch
from torchvision.utils import save_image
from torchvision.transforms.functional import to_tensor


def load_img(filename):
    x = Image.open(filename).convert('RGB')
    return to_tensor(x)[None, :, :, :]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--svg", help="source SVG path")
    args = parser.parse_args()

    svg_folder = os.path.join(args.svg)
    svgs = glob.glob(svg_folder + '/*.png')
    renders = []

    for file in range(len(svgs)):
        name = svg_folder + f'/{file}.png'
        print(name)
        tensor = load_img(name)
        renders.append(tensor)
    render = torch.cat(renders, dim=0)

    save_image(render.cpu().data,
               svg_folder + f"/img.png",
               normalize=False,
               nrow=10)


if __name__ == "__main__":
    main()
