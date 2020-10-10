import numpy as np
from PIL import Image
import os
import argparse


def GetPatchImage(patch_id, container_dir):
    """Returns a 64 x 64 patch with the given patch_id. Catch container images to
       reduce loading from disk.
    """
    # Define constants. Each container image is of size 1024x1024. It packs at
    # most 16 rows and 16 columns of 64x64 patches, arranged from left to right,
    # top to bottom.
    PATCHES_PER_IMAGE = 16 * 16
    PATCHES_PER_ROW = 16
    PATCH_SIZE = 64

    # Calculate the container index, the row and column index for the given
    # patch.
    container_idx, container_offset = divmod(patch_id, PATCHES_PER_IMAGE)
    row_idx, col_idx = divmod(container_offset, PATCHES_PER_ROW)

    # Read the container image if it is not cached.
    if GetPatchImage.cached_container_idx != container_idx:
        GetPatchImage.cached_container_idx = container_idx
        GetPatchImage.cached_container_img = np.asarray(Image.open('{}/patches{%04d}.bmp'.format(container_dir, container_idx)))

    # Extract the patch from the image and return.
    patch_image = GetPatchImage.cached_container_img[ \
        PATCH_SIZE * row_idx:PATCH_SIZE * (row_idx + 1), \
        PATCH_SIZE * col_idx:PATCH_SIZE * (col_idx + 1)]
    return patch_image


# Static variables initialization for GetPatchImage.
GetPatchImage.cached_container_idx = None
GetPatchImage.cached_container_img = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser('match net')
    parser.add_argument("--datadir", type=str,
                        default='yosemite', help="data directory.")
    # parser.add_argument('info_file',
    #                     help='Path to info.txt file in the dataset.')

    # parser.add_argument('info_file',
    #                     help='Path to info.txt file in the dataset.')
    # parser.add_argument('interest_file',
    #                     help='Path to interest.txt file in the dataset.')
    # parser.add_argument('container_dir',
    #                     help='Patch to the directory of .bmp files.')
    # parser.add_argument('output_db', help='Path to output database.')
    args = parser.parse_args()

    info_file = os.path.join(args.datadir, 'info.txt')
    interest_file = os.path.join(args.datadir, 'interest.txt')

    # Read the 3Dpoint IDs from the info file.
    with open(info_file) as f:
        point_id = [int(line.split()[0]) for line in f]

    # Read the interest point from the interest file. The fields in each line
    # are: image_id, x, y, orientation, and scale. We parse all of them as float
    # even though image_id is integer.
    with open(interest_file) as f:
        interest = [[float(x) for x in line.split()] for line in f]

    img_list = []
    for i in range(interest):
        img_list.append(np.expand_dims(GetPatchImage(i, args.datadir), axis=0))

    img_array = np.stack(img_list)
    assert img_array.shape == (len(interest), 1, 64, 64)

    np.savez_compressed('{}.npz'.format(args.datadir),
                        img=img_array,
                        label=np.array(point_id),
                        metadata=np.array(interest))