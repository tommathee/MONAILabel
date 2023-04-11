import numpy as np
import os
import shutil
import json


def custom_infere(data, configs, paths, xml_path):
    directory = 'tmp'
    os.mkdir(directory)
    json.dump(configs, open(f'{directory}/configs.json', 'w'))
    json.dump(paths, open(f'{directory}/paths.json', 'w'))
    np.save(f'{directory}/data.npy', data)

    result = os.system(
        f'python ./apps/pathology/lib/infers/infere.py --directory "{directory}" --xml_path "./datasets/labels/final/{xml_path}"')

    mask = np.load(f'{directory}/mask.npy')
    shutil.rmtree('tmp')
    return mask
