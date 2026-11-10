"""Generate Gaussian-blur and Gaussian-noise image defense datasets.

Example:
    python utils/generate_defense_dataset.py --input-folders dataset/images
"""

import argparse
from pathlib import Path


def process_images(input_folder, method, args):
    import torch
    from torchvision.io import read_image, write_png
    from torchvision.transforms import GaussianBlur

    input_folder = Path(input_folder).expanduser().resolve()
    images = sorted(p for p in input_folder.iterdir()
                    if p.is_file() and p.suffix.lower() in ('.png', '.jpg', '.jpeg'))
    if not images:
        raise ValueError(f'No PNG or JPEG images found in {input_folder}')
    if method == 'gaussian_blur':
        prefix = f'GaussianBlur-k{args.kernel_size}s{args.sigma}'
    else:
        prefix = f'GaussianNoise-m{args.mean}s{args.std}'
    output_root = args.output_root or input_folder.parent
    output_folder = Path(output_root).expanduser().resolve() / f'{prefix}-{input_folder.name}'
    if output_folder == input_folder:
        raise ValueError('Output directory must differ from the input directory')
    # Keep the original extension in the name to distinguish a.jpg from a.png.
    destinations = [output_folder / (p.name if p.suffix == '.png' else p.name + '.png')
                    for p in images]
    if len(set(destinations)) != len(destinations):
        raise ValueError('Input filenames map to duplicate output names')
    if not args.overwrite and any(p.exists() for p in destinations):
        raise FileExistsError(f'Output images already exist in {output_folder}; use --overwrite to replace them')
    output_folder.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(args.seed)
    blur = GaussianBlur(args.kernel_size, args.sigma)
    with torch.inference_mode():
        for source, destination in zip(images, destinations):
            image = read_image(str(source)).float() / 255.0
            if method == 'gaussian_blur':
                processed = blur(image)
            else:
                noise = torch.randn(image.shape, generator=generator) * args.std + args.mean
                processed = (image + noise).clamp(0, 1)
            write_png((processed * 255).byte(), str(destination))
    print(f'Processed {len(images)} images: {output_folder}')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-folders', nargs='+', required=True)
    parser.add_argument('--methods', nargs='+', choices=['gaussian_blur', 'gaussian_noise'],
                        default=['gaussian_blur', 'gaussian_noise'])
    parser.add_argument('--output-root', type=Path, help='Default: each input directory\'s parent')
    parser.add_argument('--kernel-size', type=int, default=3)
    parser.add_argument('--sigma', type=float, default=0.1)
    parser.add_argument('--mean', type=float, default=0.0)
    parser.add_argument('--std', type=float, default=0.005)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.kernel_size <= 0 or args.kernel_size % 2 == 0:
        parser.error('--kernel-size must be a positive odd integer')
    if args.sigma <= 0 or args.std < 0:
        parser.error('--sigma must be positive and --std must be nonnegative')
    return args


if __name__ == '__main__':
    args = parse_args()
    for folder in args.input_folders:
        for method in args.methods:
            process_images(folder, method, args)
