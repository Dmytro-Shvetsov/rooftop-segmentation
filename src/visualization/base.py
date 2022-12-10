from abc import ABC, abstractmethod
from os import PathLike
from typing import Union

import numpy as np
from PIL import Image


class BaseVisualizer(ABC):
  
    @abstractmethod
    def draw(self, *args, **kwargs) -> None:
        """
        Visualizes the results on the images.
        """
        pass

    def get_image(self, pil=False) -> Union[np.ndarray, Image.Image]:
        """
        Returns the final image after all of the visualizations.
        Args:
            pil (bool, optional): whether to convert the image to PIL.Image. Defaults to False.
        """
        pass

    @abstractmethod
    def save(self, file_path: PathLike) -> None:
        """
        Saves the visualizations as image to file.
        """
        pass
