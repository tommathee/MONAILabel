from typing import Sequence


class BundleConstants:
    def __init__(
        self,
        configs: Sequence[str] = ["inference.json", "inference.yaml"],
        metadata_json: str = "metadata.json",
        model_pytorch: str = "model.pt",
        model_torchscript: str = "model.ts",
        key_device: str = "device",
        key_bundle_root: str = "bundle_root",
        key_network_def: str = "network_def",
        key_preprocessing: Sequence[str] = ["preprocessing", "pre_transforms"],
        key_postprocessing: Sequence[str] = ["postprocessing", "post_transforms"],
        key_inferer: Sequence[str] = ["inferer"],
        key_detector: Sequence[str] = ["detector"],
        key_detector_ops: Sequence[str] = ["detector_ops"],
        key_displayable_configs: Sequence[str] = ["displayable_configs"],
    ):
        self._configs = configs
        self._metadata_json = metadata_json
        self._model_pytorch = model_pytorch
        self._model_torchscript = model_torchscript
        self._key_device = key_device
        self._key_bundle_root = key_bundle_root
        self._key_network_def = key_network_def
        self._key_preprocessing = key_preprocessing
        self._key_postprocessing = key_postprocessing
        self._key_inferer = key_inferer
        self._key_detector = key_detector
        self._key_detector_ops = key_detector_ops
        self._key_displayable_configs = key_displayable_configs

    def configs(self) -> Sequence[str]:
        return self._configs

    def metadata_json(self) -> str:
        return self._metadata_json

    def model_pytorch(self) -> str:
        return self._model_pytorch

    def model_torchscript(self) -> str:
        return self._model_torchscript

    def key_device(self) -> str:
        return self._key_device

    def key_bundle_root(self) -> str:
        return self._key_bundle_root

    def key_network_def(self) -> str:
        return self._key_network_def

    def key_preprocessing(self) -> Sequence[str]:
        return self._key_preprocessing

    def key_postprocessing(self) -> Sequence[str]:
        return self._key_postprocessing

    def key_inferer(self) -> Sequence[str]:
        return self._key_inferer

    def key_detector(self) -> Sequence[str]:
        return self._key_detector

    def key_detector_ops(self) -> Sequence[str]:
        return self._key_detector_ops

    def key_displayable_configs(self) -> Sequence[str]:
        return self._key_displayable_configs
