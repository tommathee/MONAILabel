import glob
import os

# TODO: Add import for your model here
from model.pathology_structure_segmentation_deeplabv3plus.models.model import DeepLabV3Plus


class TensorflowConfig():
    def __init__(self):
        self.models = {
            "deeplabv3plus": {
                "config": [
                    'apps/pathology/model/pathology_structure_segmentation_deeplabv3plus/configs/config.json',
                    'apps/pathology/model/pathology_structure_segmentation_deeplabv3plus/configs/config2.json',
                    'apps/pathology/model/pathology_structure_segmentation_deeplabv3plus/configs/config3.json',
                ],
                "model": [
                    DeepLabV3Plus,
                    DeepLabV3Plus,
                    DeepLabV3Plus,
                ],
            },
            # TODO: Add your model here
        }

    def get_model(self, model_name):
        if model_name in self.models:
            return self.models[model_name]["model"]
        return None

    def get_config(self, model_name):
        if model_name in self.models:
            return self.models[model_name]["config"]
        return None
