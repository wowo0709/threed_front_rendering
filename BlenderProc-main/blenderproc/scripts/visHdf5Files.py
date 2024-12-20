""" Visualize the hdf5 file """

import os
import argparse
from pathlib import Path
import json
import re

import h5py
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

default_rgb_keys = ["colors", "normals", "diffuse", "nocs"]
default_flow_keys = ["forward_flow", "backward_flow"]
default_segmap_keys = ["segmap", ".*_segmaps"]
default_segcolormap_keys = ["segcolormap"]
default_depth_keys = ["distance", "depth", "stereo-depth"]
all_default_keys = default_rgb_keys + default_flow_keys + default_segmap_keys + default_segcolormap_keys + \
                   default_depth_keys
default_depth_max = 20.0

# Example command
# blenderproc vis hdf5 /root/data/3D-FRONT/3D-FRONT-processed/bedrooms_without_lamps_full_raw/raw --scene_idx 0 --frame_idx 0 --save /root/data/3D-FRONT/3D-FRONT-processed/bedrooms_without_lamps_full_images/images_512


def flow_to_rgb(flow):
    """
    Visualizes optical flow in hsv space and converts it to rgb space.
    :param flow: (np.array (h, w, c)) optical flow
    :return: (np.array (h, w, c)) rgb data
    """
    # pylint: disable=import-outside-toplevel
    import cv2
    # pylint: enable=import-outside-toplevel

    im1 = flow[:, :, 0]
    im2 = flow[:, :, 1]

    h, w = flow.shape[:2]

    # Use Hue, Saturation, Value colour model
    hsv = np.zeros((h, w, 3), dtype=np.float32)
    hsv[..., 1] = 1

    mag, ang = cv2.cartToPolar(im1, im2)
    hsv[..., 0] = ang * 180 / np.pi
    hsv[..., 2] = cv2.normalize(mag, None, 0, 1, cv2.NORM_MINMAX)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def key_matches(key, patterns, return_index=False):
    """
    Match the key to the patterns
    """
    for p, pattern in enumerate(patterns):
        if re.fullmatch(pattern, key):
            return (True, p) if return_index else True

    return (False, None) if return_index else False


def vis_data(key, data, full_hdf5_data=None, file_label="", rgb_keys=None, flow_keys=None, segmap_keys=None,
             segcolormap_keys=None, depth_keys=None, depth_max=default_depth_max, save_to_file=None, flip=None, render_top=False):
    """
    Visualize the data
    """
    if rgb_keys is None:
        rgb_keys = default_rgb_keys[:]
    if flow_keys is None:
        flow_keys = default_flow_keys[:]
    if segmap_keys is None:
        segmap_keys = default_segmap_keys[:]
    if segcolormap_keys is None:
        segcolormap_keys = default_segcolormap_keys[:]
    if depth_keys is None:
        depth_keys = default_depth_keys[:]

    # If key is valid and does not contain segmentation data, create figure and add title
    if key_matches(key, flow_keys + rgb_keys + depth_keys):
        plt.figure()
        plt.title(f"{key} in {file_label}")

    if key_matches(key, flow_keys):
        try:
            # Visualize optical flow
            if save_to_file is None:
                plt.imshow(flow_to_rgb(data), cmap='jet')
            else:
                plt.imsave(save_to_file, flow_to_rgb(data), cmap='jet')
                plt.close()
        except ImportError as e:
            raise ImportError("Using .hdf5 containers, which contain flow images needs opencv-python to be "
                              "installed!") from e
    elif key_matches(key, segmap_keys):
        # Try to find labels for each channel in the segcolormap
        channel_labels = {}
        _, key_index = key_matches(key, segmap_keys, return_index=True)
        if key_index < len(segcolormap_keys):
            # Check if segcolormap_key for the current segmap key is configured and exists
            segcolormap_key = segcolormap_keys[key_index]
            if full_hdf5_data is not None and segcolormap_key in full_hdf5_data:
                # Extract segcolormap data
                segcolormap = json.loads(np.array(full_hdf5_data[segcolormap_key]).tostring())
                segcolormap = json.loads(np.array(full_hdf5_data[segcolormap_key]).tostring())
                if len(segcolormap) > 0:
                    # Go through all columns, we are looking for channel_* ones
                    for colormap_key, colormap_value in segcolormap[0].items():
                        if colormap_key.startswith("channel_") and colormap_value.isdigit():
                            channel_labels[int(colormap_value)] = colormap_key[len("channel_"):]

        # Make sure we have three dimensions
        if len(data.shape) == 2:
            data = data[:, :, None]
        if render_top:
            data = np.flip(np.flip(data, axis=0), axis=1)
        # Go through all channels
        for i in range(data.shape[2]):
            # Try to determine label
            channel_label = channel_labels.get(i, i)

            # Visualize channel
            if save_to_file is None:
                plt.figure()
                plt.title(f"{key} / {channel_label} in {file_label}")
                plt.imshow(data[:, :, i], cmap='jet')
            else:
                if flip:
                    data = np.flip(data, axis=1)
                # print(data.shape)
                # print(data[-1, -1])
                if data.shape[2] > 1:
                    filename = save_to_file.replace(".png", f"_{channel_label}.png")
                else:
                    filename = save_to_file
                plt.imsave(filename, data[:, :, i], cmap='jet')
                plt.close()

    elif key_matches(key, depth_keys):
        # Make sure the data has only one channel, otherwise matplotlib will treat it as a rgb image
        if len(data.shape) == 3:
            if data.shape[2] != 1:
                print(f"Warning: The data with key '{key}' has more than one channel which would not allow using "
                      f"a jet color map. Therefore only the first channel is visualized.")
            data = data[:, :, 0]

        if save_to_file is None:
            plt.imshow(data, cmap='summer', vmax=depth_max)
            plt.colorbar()
        else:
            if flip:
                data = np.flip(data, axis=1)
            # plt.imsave(save_to_file, data, cmap='summer', vmax=depth_max)
            # plt.imsave(save_to_file, data, cmap='summer')
            mask = data != np.max(data)
            # print("data", type(data), np.min(data), np.max(data[mask]), np.max(data)) # data <class 'numpy.ndarray'> 0.765375 3.389505 10000000000.0
            ### Normalize raw float16 depth value
            # Clip max depth
            data = np.clip(data, 0.0, 20.0)
            # print("data", data.dtype)
            np.save(os.path.splitext(save_to_file)[0], data)
            """
            # print("data", type(data), np.min(data), np.max(data[mask]), np.max(data))
            # Normalize to [0, 1]
            # data = (data - np.min(data)) / (np.max(data) - np.min(data))
            data /= 20.0
            # print("data", type(data), np.min(data), np.max(data[mask]), np.max(data))
            # Map to [0, 65536] in uint16 type
            data = (data * 65535).astype(np.uint16)
            # print("data", type(data), np.min(data), np.max(data[mask]), np.max(data))
            # raise NotImplementedError()
            image = Image.fromarray(data)
            image.save(save_to_file)
            # plt.close()
            """
    elif key_matches(key, rgb_keys):
        if save_to_file is None:
            plt.imshow(data)
        else:
            if flip:
                data = np.flip(data, axis=1)
            if key == 'colors':
                background = (np.ones((*data.shape[:2], 3))*255).astype(np.uint8)
                background = Image.fromarray(background)
                data2 = Image.fromarray(data)
                background.paste(data2, mask=data2.convert('RGBA'))
                background = np.array(background)
                data = background
            else:
                data = (data*255).astype(np.uint8)
            if render_top:
                data = np.flip(np.flip(data, axis=0), axis=1)
            plt.imsave(save_to_file, data)
            plt.close()
    else:
        if save_to_file is None:
            plt.imshow(data)
        else:
            plt.imsave(save_to_file, data)
            plt.close()


def vis_file(path, keys_to_visualize=None, rgb_keys=None, flow_keys=None, segmap_keys=None, segcolormap_keys=None,
             depth_keys=None, depth_max=default_depth_max, save_to_path=None, flip=None, appendix_name='', render_top=False):
    """ Visualize a file """
    if save_to_path is not None and not os.path.exists(save_to_path):
        os.makedirs(save_to_path)

    # Check if file exists
    if os.path.exists(path):
        if os.path.isfile(path):
            with h5py.File(path, 'r') as data:
                print(path + ": ")

                # Select only a subset of keys if args.keys is given
                if keys_to_visualize is not None:
                    keys = [key for key in data.keys() if key_matches(key, keys_to_visualize)]
                else:
                    keys = list(data.keys())
                # Visualize every key
                res = []
                for key in keys:
                    value = np.array(data[key])

                    if sum(ele for ele in value.shape) < 5 or "version" in key:
                        if value.dtype == "|S5":
                            res.append(
                                (key, str(value).replace("[", "").replace("]", "").replace("b'", "").replace("'", "")))
                        else:
                            res.append((key, value))
                    else:
                        res.append((key, value.shape))

                if res:
                    res = [f"'{key}': {key_res}" for key, key_res in res]
                    print("Keys: " + ', '.join(res))

                for key in keys:
                    value = np.array(data[key])
                    if save_to_path is not None:
                        # save_to_file = os.path.join(save_to_path,
                        #                             str(os.path.basename(path)).split('.', maxsplit=1)[0] +
                        #                             f"_{key}.png")
                        # out_path_file = os.path.join(save_to_path, key)
                        # os.makedirs(out_path_file, exist_ok=True)
                        out_path_file = save_to_path
                        file_ind = str(os.path.basename(path)).split('.', maxsplit=1)[0]
                        file_ind = f'{int(file_ind):04d}'
                        keyname = f"_{key}"
                        # save_to_file = os.path.join(out_path_file, f"{file_ind}{appendix_name}_{key}.png")
                        save_to_file = os.path.join(out_path_file, f"{file_ind}{keyname}{appendix_name}.png")
                    else:
                        save_to_file = None

                    print("KEY", key)
                    print("SAVE_TO_FILE", save_to_file)

                    # Check if it is a stereo image
                    if len(value.shape) >= 3 and value.shape[0] == 2:
                        # Visualize both eyes separately
                        for i, img in enumerate(value):
                            if save_to_file:
                                save_to_file = str(Path(save_to_file).with_suffix("")) + (
                                    "_left" if i == 0 else "_right") + Path(save_to_file).suffix
                            vis_data(key, img, data, os.path.basename(path) + (" (left)" if i == 0 else " (right)"),
                                     rgb_keys, flow_keys, segmap_keys, segcolormap_keys, depth_keys, depth_max,
                                     save_to_file, flip, render_top)
                    else:
                        vis_data(key, value, data, os.path.basename(path), rgb_keys, flow_keys, segmap_keys,
                                 segcolormap_keys, depth_keys, depth_max, save_to_file, flip, render_top)
        else:
            print("The path is not a file")
    else:
        print(f"The file does not exist: {path}")


def cli():
    """
    Command line function
    """
    parser = argparse.ArgumentParser("Script to visualize hdf5 files")

    parser.add_argument('hdf5_paths', help='Path to hdf5 file/s')
    parser.add_argument('--keys', nargs='+', help='Keys that should be visualized. If none is given, '
                                                  'all keys are visualized.', default=all_default_keys)
    parser.add_argument('--rgb_keys', nargs='+', help='Keys that should be interpreted as rgb data.',
                        default=default_rgb_keys)
    parser.add_argument('--flow_keys', nargs='+', help='Keys that should be interpreted as optical flow data.',
                        default=default_flow_keys)
    parser.add_argument('--segmap_keys', nargs='+', help='Keys that should be interpreted as segmentation data.',
                        default=default_segmap_keys)
    parser.add_argument('--segcolormap_keys', nargs='+', help='Keys that point to the segmentation color maps '
                                                              'corresponding to the configured segmap_keys.',
                        default=default_segcolormap_keys)
    parser.add_argument('--depth_keys', nargs='+', help='Keys that contain additional non-RGB data which should be '
                                                        'visualized using a jet color map.', default=default_depth_keys)
    parser.add_argument('--depth_max', type=float, default=default_depth_max)
    parser.add_argument('--save', default=None, type=str, help='Saves visualizations to file.')
    parser.add_argument('--scene_idx', type=int)
    parser.add_argument('--frame_idx', type=int)
    parser.add_argument('--flip', type=bool, default=False)
    parser.add_argument('--path_labels', default=None, type=str)
    parser.add_argument('--appendix_name', default='', type=str, required=False)
    parser.add_argument('--render_top', type=bool, default=False)

    args = parser.parse_args()
    if args.path_labels is not None:
        scene_in = sorted(os.listdir(args.path_labels))[args.scene_idx]
    else:
        scene_in = sorted(os.listdir(args.hdf5_paths))[args.scene_idx]
    args.hdf5_paths = [os.path.join(args.hdf5_paths, scene_in, f'{args.frame_idx}.hdf5')]
    args.save = os.path.join(args.save, scene_in)
    file_path = os.path.join(args.save, f"{args.frame_idx}.png")
    if os.path.isfile(file_path):
        exit()

    # Visualize all given files
    for path in args.hdf5_paths:
        vis_file(
            path=path,
            keys_to_visualize=args.keys,
            rgb_keys=args.rgb_keys,
            flow_keys=args.flow_keys,
            segmap_keys=args.segmap_keys,
            segcolormap_keys=args.segcolormap_keys,
            depth_keys=args.depth_keys,
            depth_max=args.depth_max,
            save_to_path=args.save,
            flip=args.flip,
            appendix_name=args.appendix_name,
            render_top=args.render_top
        )
    if args.save is None:
        plt.show()


if __name__ == "__main__":
    cli()
