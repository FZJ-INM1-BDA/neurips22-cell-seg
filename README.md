# neurips22-cell-seg

This is the _ported_ code of the team "cells" of the [NeurIPS 22 Cell Segmentation Challenge](https://neurips22-cellseg.grand-challenge.org/).
Most of the functions are also available in the `celldetection` python package: https://github.com/FZJ-INM1-BDA/celldetection

## Project structure and general notes
The source code is located in `neurips`.
All docker related code is located in `docker`.
Note that the docker code includes a minimal _copy_ of the source code, which excludes several unnecessary imports and thus reduces docker setup overhead.
Profiling was performed using [`tuna`](https://github.com/nschloe/tuna).

## Environments and requirements
- Run `pip install -r requirements.txt` to install requirements.
- If you cannot install `mpi4py` via pip, consider using anaconda and the binaries it provides: `conda install mpi4py`
- Development was performed on the high-performance computing (HPC) system [JURECA](https://apps.fz-juelich.de/jsc/hps/jureca/configuration.html). Please refer to the linked site for specific specs.
- Training was performed on 2 nodes (8×NVIDIA A100)
- Following the challenge guidelines, inference was performed on a single Nvidia 2080 Ti with a patch size of 768×768 and a batch size of 4.

## Datasets
Download related datasets and place the data in the respective directory `inputs/raw_external/<dataset_name>`.
- [BBBC039](https://bbbc.broadinstitute.org/BBBC039)
- [BBBC038](https://bbbc.broadinstitute.org/BBBC038)
- [Omnipose](https://www.cellpose.org/dataset_omnipose)
- [Cellpose](https://www.cellpose.org/dataset)
- [SCIS](https://www.kaggle.com/competitions/sartorius-cell-instance-segmentation)
- [Livecell](https://github.com/sartorius-research/LIVECell) (also mirrored in SCIS)

The challenge dataset can be accessed here: [NeuriPS22](https://neurips22-cellseg.grand-challenge.org/dataset/).
Download and extract it to `inputs/neurips_data`.

## Preprocessing
- Run `python preprocess_external_data.py` to preprocess external data
- Preprocessing results are written to `inputs/external_data`
- To convert unlabeled data to patches run `python preprocess_unlabeled_data.py`
- To create a validation split run ` python preprocess_val_data.py -i inputs/neurips_data/Train_Labeled -o inputs/validation_data -n 50` 

## Training
- Run `mpirun -n <ranks> python train.py -i "./inputs" -o "./outputs" -s "schedule.json" -c 0`
- Settings are defined in `schedule.json`
- A _schedule_ can define configurations for multiple experiments. Specify the configuration index via `-c <index>`
- To continue the training of a saved model use the `checkpoint` parameter in the schedule.

## Inference

### Inference for label images
Find all images in input folder, produce label images and save to output folder.
```
python inference.py -i inputs/neurips_data/unlabeled_patches -o output_folder
```

### Inference for pseudo-labels
```
python inference.py -i inputs/neurips_data/unlabeled_patches -o inputs/pseudo_labels0 --outtype=h5
```

## Models
A trained CPN can be found [here](https://celldetection.org/torch/models/ginoro.pt).
Use it as follows:
```python
pt = torch.load(model_name, map_location=device)
model = neurips.nn.build_cpn_model(pt['config'])
model.load_state_dict(pt['state_dict'])
```

## Docker
Find the full list of docker images [here](https://hub.docker.com/repository/docker/ericup/neurips22-cell-seg/tags?page=1&ordering=last_updated).

1. Docker pull:
    ```
    docker pull ericup/neurips22-cell-seg:v0.0.1
    ```
2. Docker run
    ```
    docker container run --gpus="device=0" --name cells --rm -v $PWD/inputs/:/workspace/inputs/ -v $PWD/cells_outputs/:/workspace/outputs/ ericup/neurips22-cell-seg:v0.0.1 /bin/bash -c "sh predict.sh"
    ```
The latter will process all images from a directory called `inputs` and write results to a directory called `cells_outputs`.

## Build Docker
- Saved models can be exported for docker via `python export_model.py -i model.pt -o ./exported_models`
- Place the exported model in the `docker` directory and adapt `predict.sh`.
- The following snippet creates and loads a docker solution:
```
cd docker

sh scripts/build.sh
sh scripts/export.sh
sh scripts/load.sh
```

## Evaluation
For evaluation, please refer to the officially provided code: https://github.com/JunMa11/NeurIPS-CellSeg#compute-evaluation-metric-f1-score

## Acknowledgement
We thank the contributors of public datasets.
