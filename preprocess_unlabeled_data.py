from glob import glob
from imageio import imread, imwrite
from os.path import basename, dirname, join
import neurips as cs
import tifffile as tif
from tqdm import tqdm

in_dir = 'inputs'
out_dir = 'inputs/unlabeled_patches'


def write(filename, img):
    if filename.endswith('tiff') or filename.endswith('tif'):
        tif.imwrite(filename, img)
    else:
        imwrite(filename, img)


files = sorted(glob(join(in_dir, 'neurips_data', 'Train_Unlabeled', '**', '*.*'), recursive=True))
for f in tqdm(files):
    img = imread(f)
    if 'whole_slide' in f:
        tl = cs.nn.inference.TileLoader(img, crop_size=(1024, 1024), strides=(1024 - 128, 1024 - 128), reps=1)
        for crop, meta in tl:
            y, x = meta[:2]
            sp = basename(f).split('.')
            dst = '.'.join(sp[:-1]) + '_' + 'x'.join([str(i) for i in crop.shape]) + f'_{y}_{x}.' + sp[-1]
            dst = join(out_dir, dst)
            write(dst, crop)
    else:
        dst = join(out_dir, basename(f))
        write(dst, img)
