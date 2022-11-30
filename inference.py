import torch
import numpy as np
import celldetection as cd
import cv2
import neurips as ne
import os
from os.path import join, dirname, basename
import argparse
from psutil import cpu_count
from glob import glob
import tifffile as tif
from imageio import imread
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # there's at least one in unlabeled data

import warnings
warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser('Contour Proposal Networks for Instance Segmentation', add_help=False)
    parser.add_argument('-i', '--input_path', default='./inputs', type=str,
                        help='training data path; subfolders: images, labels')
    parser.add_argument("-o", '--output_path', default='./outputs', type=str, help='output path')
    parser.add_argument('--model', default='./ginoro_CpnResNeXt101UNet_epoch_225.pt',  # ./vayibu_CpnResNeXt101UNet_epoch_160.pt
                        help='Filename of model.')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Device.')
    parser.add_argument('--workers', default=min(4, max(0, int(cpu_count(logical=False) * .8))),
                        type=int, help='Number of workers.')
    parser.add_argument('--iconfig', default=None, type=str, help='Inference config.')
    parser.add_argument('--outtype', default='tif', type=str, help='Inference config.')
    args = parser.parse_args()

    if args.iconfig is None:
        inf_conf = cd.Config(
            crop_size=(768,) * 2,
            strides=(768 - 384,) * 2,
            window_removal_pad=3,
            amp=True,
            nms_crop_size=(8192, 8192),
            nms_strides=(8100, 8100),
            reps=1,  # multiples of 4, or change batch_size
            weight_final_nms=True,
            weighted_tile_nms=True,
            tiled_final_nms=True,
            min_vote_fraction=None,
            score_thresh=.891415,
            nms_thresh=.3141592653589793,
            aug_transforms=['ToRGB'],
            voting_method='mean',
            uncertainty_nms=True,
            batch_size=4,
            certainty_thresh=None,
            img_norm='cstm-mix',
            uncertainty_sigmoid=True,
        )
    else:
        inf_conf = cd.Config.from_json(args.iconfig)

    # Set up environment
    device = args.device
    num_workers = args.workers
    input_path = args.input_path
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)
    filenames = glob(join(input_path, '*.*'))
    model_name = args.model

    print('Settings:\n ', '\n  '.join([
        f'Input path: {input_path}',
        f'Output path: {output_path}',
        f'Num inputs: {len(filenames)}',
        f'Device: {device}',
        f'Num workers: {num_workers}',
        f'Model: {model_name}',
    ]))

    ld = torch.load(model_name, map_location=device)
    model_conf = ld['config']
    model_conf.checkpoint = None  # should not be set for docker use
    model = ne.nn.build_cpn_model(model_conf)
    model.score_thresh = inf_conf.score_thresh
    model.nms_thresh = inf_conf.nms_thresh
    model.certainty_thresh = inf_conf.certainty_thresh
    model.uncertainty_nms = inf_conf.uncertainty_nms
    model.refinement = inf_conf.get('refinement', True)

    print('Load state dict')
    model.load_state_dict(ld['state_dict'])

    def transforms(image):
        img = image
        if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return {'image': img}

    kwargs = inf_conf.kwargs(ne.nn.CpnInference.__init__)
    print('Inference settings:\n ', '\n  '.join([f'({k}): {v}' for k, v in kwargs.items()]))
    ci = ne.nn.CpnInference(model=model.to(device), transforms=transforms, **kwargs)
    ci.loader_conf.batch_size = inf_conf.batch_size
    ci.loader_conf.num_workers = num_workers

    for fi, filename in enumerate(filenames):
        dst = join(output_path, basename(filename).split('.')[0] + '_label.tiff')
        print(f'({fi + 1}/{len(filenames)})\n', filename, '->', dst)
        img = raw_img = imread(filename)
        img = ne.multi_norm(img, inf_conf.img_norm)

        y = ci(img)
        contours = cd.asnumpy(y[0])
        labels = ne.contours2labels(contours, img.shape[:2])
        if args.outtype == 'tif' or args.outtype == 'tiff':
            tif.imwrite(dst, labels)
        else:
            scores, boxes, uncertainty, weights, score_maps = y[1:]
            score_maps = torch.squeeze(score_maps, 0)
            if inf_conf.uncertainty_sigmoid:
                uncertainty = uncertainty.sigmoid()
            cd.to_h5(
                dst.replace('.tiff', '.h5'),
                image=raw_img,
                labels=labels,
                contours=contours,
                scores=cd.asnumpy(scores),
                boxes=cd.asnumpy(boxes),
                uncertainty=cd.asnumpy(uncertainty),
                weights=cd.asnumpy(weights),
                score_maps=cd.asnumpy(score_maps),
            )


if __name__ == "__main__":
    main()
