"""Preprocessing functions for image object detection tasks.
"""

import numpy as np
from PIL import Image


def preprocess(image: Image, img_max_side_length: int = 640, normalize: bool = True) -> np.array:
    """Preprocess image

    Preprocess images so that they can be used by the ONNX-based model.

    Preprocessing consists of 3 steps:
    - Downscaling the image if it is too big (this is controlled by the img_max_side_length parameter)
    - Normalizing the image colors in accordance with ImageNet's mean and standard deviation. This is a direct
        recommendation from the Azure Machine Learning documentation linked in the "See Also" section.
    - Shaping the image into a numpy array, adding a dimension so that it can be properly processed by the ONNX-based
        model

    See Also:
        https://learn.microsoft.com/en-us/azure/machine-learning/how-to-inference-onnx-automl-image-models?view=azureml-api-2&tabs=instance-segmentation#preprocessing

    Args:
        image: Image to preprocess
        img_max_side_length: Maximum side length for the output image in pixels. Input images which have any of their
            sides longer than this variable will be downscaled. There is no upscaling. While the model may produce more
            accurate outputs if the sizes of the supplied images are similar to the sizes of the images used in
            training, smaller images result in a faster inference time. For this reason, a relatively small
            img_max_side_length, 640, is chosen as a default
        normalize: This parameter controls whether normalization is run. Setting this to False helps in the case where
            the resulting image should be fed back to the user, so they only see it rescaled, but with the original
            colors intact. Normalization should be enabled when the preprocessing is for producing images which will
            be fed to the machine learning model for inference.

    Returns:
        img: Preprocessed image as numpy array
    """
    if max(image.size) > img_max_side_length:
        # Downscale if image too big
        image = image.resize((int(image.size[0] / max(image.size) * img_max_side_length),
                              int(image.size[1] / max(image.size) * img_max_side_length)))

    # HWC -> CHW
    image_np = np.array(image)
    image_np = image_np.transpose(2, 0, 1)  # CxHxW

    if normalize:
        # normalize the image
        mean_vec = np.array([0.485, 0.456, 0.406])
        std_vec = np.array([0.229, 0.224, 0.225])
        image_np_norm = np.zeros(image_np.shape).astype('float32')
        for i in range(image_np.shape[0]):
            image_np_norm[i, :, :] = (image_np[i, :, :] / 255 - mean_vec[i]) / std_vec[i]
        image_np = image_np_norm

    image_np = np.expand_dims(image_np, axis=0)  # 1xCxHxW

    return image_np
