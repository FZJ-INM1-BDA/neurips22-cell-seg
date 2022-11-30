from glob import glob
import argparse
from shutil import move
from os.path import join, basename
from os import makedirs
import numpy as np


parser = argparse.ArgumentParser('Export model', add_help=False)
parser.add_argument('-i', '--input', type=str, help='Input (directory).')
parser.add_argument('-o', '--output', type=str, help='Output (directory).')
parser.add_argument('-n', '--num', type=int, help='Number of items.')
parser.add_argument('-s', '--seed', default=None, type=int, help='Random seed.')
args = parser.parse_args()


files = sorted(glob(join(args.input, 'images', '*.*')))
if args.seed is not None:
    np.random.seed(args.seed)
sel = np.random.choice(files, args.num, replace=False)
makedirs(join(args.output, 'images'), exist_ok=True)
makedirs(join(args.output, 'labels'), exist_ok=True)
for f in sel:
    f_, = glob(join(args.input, 'labels', '.'.join(basename(f).split('.')[:-1]) + '*'))
    f_dst = join(args.output, 'images', basename(f))
    f_dst_ = join(args.output, 'labels', basename(f_))

    print(f, '-->', f_dst)
    move(f, f_dst)
    print(f_, '-->', f_dst_)
    move(f_, f_dst_)
    print()
