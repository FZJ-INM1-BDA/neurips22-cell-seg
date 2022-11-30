import celldetection as cd
import torch
import argparse
from os.path import join, dirname, basename, isfile


parser = argparse.ArgumentParser('Export model', add_help=False)
parser.add_argument('-i', '--input', type=str, help='Input (filename).')
parser.add_argument('-o', '--output', type=str, help='Output (directory).')
args = parser.parse_args()
model_name = args.input

model = torch.load(model_name, map_location='cpu')
config_name = join(dirname(model_name), 'config.json')
if not isfile(config_name):
    config_name = join(dirname(model_name), 'config_r0.json')
conf = cd.Config.from_json(join(dirname(model_name), 'config.json'))

bn = basename(model_name)
tag = basename(dirname(model_name))
export_name = f'{tag}_{bn}'

dst = join(args.output, export_name)
if not isfile(dst):
    print(dst)
    torch.save(dict(
        state_dict=model.state_dict(),
        config=conf,
    ), dst)
