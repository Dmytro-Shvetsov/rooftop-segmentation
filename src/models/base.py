from abc import ABC, abstractmethod
from typing import Union

from numpy import ndarray
from torch import Tensor

from src.utils.config_reader import Config

ArrayLike = Union[ndarray, Tensor]


class SegmentationModelInterface(ABC):
    """
    Base segmentation model interface that defines main methods how one can manipulate with models.
    """

    _cfg: Config
    _is_loaded: bool = False

    def __init__(self, config:Config, **kwargs) -> None:
        self._cfg = config

    @property
    def is_loaded(self) -> bool:
        """
        Property whether the model was loaded from a pretrained checkpoint
        """
        return self._is_loaded

    @abstractmethod
    def load(self) -> bool:
        """
        Tries to restore the model from a checkpoint

        Returns:
            bool: boolean whether the loading was successful
        """
        pass

    @abstractmethod
    def preprocess_inputs(self, images:ArrayLike) -> ArrayLike:
        """
        Preprocesses the input batch of images. Preprocessing may include normalization, resizing, etc.

        Args:
            images (ArrayLike): batch of images (NxCxHxW)

        Returns:
            ArrayLike: preprocessed images (NxCxHxW)
        """
        pass

    @abstractmethod
    def forward(self, inputs:ArrayLike) -> ArrayLike:
        """
        Runs the images batch through the model.

        Args:
            inputs (ArrayLike): batch of images (NxCxHxW)

        Returns:
            ArrayLike: predicted logits/probs (NxCxHxW)
        """
        pass

    @abstractmethod
    def __call__(self, inputs:ArrayLike, *args, **kwds) -> ArrayLike:
        """
        Runs the images batch through the model.

        Args:
            inputs (ArrayLike): batch of images (NxCxHxW)

        Returns:
            ArrayLike: predicted logits/probs (NxCxHxW)
        """
        pass

    @abstractmethod
    def parse_outputs(self, outs:ArrayLike) -> ArrayLike:
        """
        Applies operations to convert the model outputs to final segmentation masks.

        Args:
            outs (ArrayLike): raw outputs of the model

        Returns:
            ArrayLike: mask predictions (NxCxHxW)
        """
        pass

    def process(self, images:ArrayLike, raw_outputs=False) -> ArrayLike:
        """
        Preprocess -> forward -> parse flow to inference a batch of images in one method.

        Args:
            images (ArrayLike): batch of images (NxCxHxW)
            raw_outputs (bool, optional): whether to parse model outputs or outputs raw predictions. Defaults to False.

        Returns:
            ArrayLike: model predictions (NxCxHxW)
        """
        images = self.preprocess_inputs(images)
        outs = self(images)
        return outs if raw_outputs else self.parse_outputs(outs)
