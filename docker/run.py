import time
import os
import shutil
from os.path import join, basename
from glob import glob

TIMES = {
    'cell_00001.tiff': 10,
    'cell_00002.png': 49,
    'cell_00003.tiff': 10,
    'cell_00004.png': 90,
    'cell_00005.png': 90,
    'cell_00006.png': 49,
    'cell_00007.tiff': 10,
    'cell_00008.tiff': 10,
    'cell_00009.png': 90,
    'cell_00010.png': 49,
    'cell_00011.png': 90,
    'cell_00012.png': 49,
    'cell_00013.tiff': 10,
    'cell_00014.tiff': 10,
    'cell_00015.png': 90,
    'cell_00016.tiff': 10,
    'cell_00017.png': 90,
    'cell_00018.tiff': 10,
    'cell_00019.tiff': 10,
    'cell_00020.tiff': 10,
    'cell_00021.tiff': 10,
    'cell_00022.tiff': 10,
    'cell_00023.tiff': 10,
    'cell_00024.tiff': 10,
    'cell_00025.png': 49,
    'cell_00026.png': 49,
    'cell_00027.tiff': 12,
    'cell_00028.tiff': 12,
    'cell_00029.tiff': 12,
    'cell_00030.tiff': 12,
    'cell_00031.tiff': 12,
    'cell_00032.tiff': 12,
    'cell_00033.tiff': 12,
    'cell_00034.tiff': 12,
    'cell_00035.tiff': 12,
    'cell_00036.tiff': 12,
    'cell_00037.tiff': 12,
    'cell_00038.tiff': 12,
    'cell_00039.tiff': 12,
    'cell_00040.png': 90,
    'cell_00041.png': 90,
    'cell_00042.png': 90,
    'cell_00043.png': 10,
    'cell_00044.png': 10,
    'cell_00045.png': 10,
    'cell_00046.png': 10,
    'cell_00047.png': 10,
    'cell_00048.png': 10,
    'cell_00049.png': 10,
    'cell_00050.png': 10,
    'cell_00051.png': 10,
    'cell_00052.png': 10,
    'cell_00053.png': 10,
    'cell_00054.png': 10,
    'cell_00055.png': 10,
    'cell_00056.png': 10,
    'cell_00057.png': 10,
    'cell_00058.png': 10,
    'cell_00059.png': 10,
    'cell_00060.png': 10,
    'cell_00061.png': 10,
    'cell_00062.png': 10,
    'cell_00063.png': 10,
    'cell_00064.png': 10,
    'cell_00065.png': 10,
    'cell_00066.png': 10,
    'cell_00067.png': 10,
    'cell_00068.png': 10,
    'cell_00069.png': 10,
    'cell_00070.png': 90,
    'cell_00071.tif': 42,
    'cell_00072.tif': 42,
    'cell_00073.tif': 42,
    'cell_00074.tif': 42,
    'cell_00075.tif': 14,
    'cell_00076.tif': 10,
    'cell_00077.tif': 10,
    'cell_00078.tif': 10,
    'cell_00079.tif': 10,
    'cell_00080.tif': 10,
    'cell_00081.tif': 10,
    'cell_00082.tif': 10,
    'cell_00083.tif': 10,
    'cell_00084.tif': 10,
    'cell_00085.tif': 10,
    'cell_00086.tif': 10,
    'cell_00087.tif': 10,
    'cell_00088.tif': 10,
    'cell_00089.tif': 10,
    'cell_00090.tif': 10,
    'cell_00091.tif': 10,
    'cell_00092.tif': 10,
    'cell_00093.tif': 10,
    'cell_00094.tif': 10,
    'cell_00095.tif': 10,
    'cell_00096.tif': 10,
    'cell_00097.tif': 10,
    'cell_00098.tif': 10,
    'cell_00099.tif': 42,
    'cell_00100.tif': 10,
    'cell_00101.tif': 883
}

actual_times = {}

teamname = 'cells'
case = '-'

test_img_path = '../inputs/neurips_data/TuningSet'
input_temp = 'inputs'
test_cases = sorted(glob(join(test_img_path, '*.*')))
os.makedirs(input_temp, exist_ok=True)

for case in test_cases:
    shutil.copy(case, input_temp)
    assert len(glob(join(input_temp, '*.*'))) == 1

    # cmd = 'docker container run --gpus="device=0" -m 28g --shm-size=28gb --name {} --rm -v $PWD/CellSeg_Test/:/workspace/inputs/ -v $PWD/cells_outputs/:/workspace/outputs/ {}:latest /bin/bash -c "sh predict.sh" '.format(teamname, teamname)
    cmd = 'docker container run --gpus="device=0" -m 28g --name {} --rm -v $PWD/{}/:/workspace/inputs/ -v $PWD/cells_outputs/:/workspace/outputs/ {}:latest /bin/bash -c "sh predict.sh" '.format(
        teamname, input_temp, teamname)
    print(teamname, ' docker command:', cmd, '\n', 'testing image name:', case)
    start_time = time.time()
    os.system(cmd)
    real_running_time = time.time() - start_time
    print(f"{case} finished! Inference time: {real_running_time} (budget: {TIMES.get(basename(case), '?')})\n")
    actual_times[basename(case)] = real_running_time

    # Cleanup
    os.remove(join(input_temp, basename(case)))

for k in sorted(list(actual_times.keys())):
    allowed = TIMES.get(k, -1)
    actual = actual_times[k]
    print(f'{k} - allowed: {allowed}, actual: {actual}, pass: {actual <= allowed}')
