import neurips as ne
import celldetection as cd
from os.path import join

in_dir = 'inputs'
out_dir = 'inputs/external_data'

print('This takes about 2 hours. Be patient ☕️', flush=True)
for idx, (img, lbl) in enumerate(ne.data.gen_external(
        bbbc039=f'{in_dir}/raw_external/bbbc039',
        bbbc038=f'{in_dir}/raw_external/bbbc038',
        omnipose=f'{in_dir}/raw_external/omnipose',
        cellpose=f'{in_dir}/raw_external/cellpose',
        sartorius=f'{in_dir}/raw_external/sartorius',
        livecell=f'{in_dir}/raw_external/livecell',
)):
    cd.to_h5(join(out_dir, '%09d.h5' % idx), image=img, labels=lbl, chunks=None)
