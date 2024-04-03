"""Module for loading, running, and displaying the results of the TrashRecognizer model."""

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import List

import numpy as np
import onnxruntime
import torch
from torchvision.transforms.functional import adjust_brightness
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes

from utils.image_helpers import apply_nms_to_onnx_predictions
from utils.logger_config import configure_logger
from utils.preprocessing import preprocess

logger = configure_logger(__name__)

MODELS_DIR = Path(__file__).parent.parent.parent.joinpath("./models/")


def _load_labels(model_dir: Path) -> dict:
    labels_file = model_dir.joinpath("./labels.json")
    with open(labels_file, "r") as f:
        labels = json.load(f)
    return labels


class TrashRecognizerModel:

    def __init__(self):
        """Initialize the TrashRecognizer model.

        This class loads the TrashRecognizer model and makes it available for inference.
        """
        self.model_dir = MODELS_DIR.joinpath("./trash_recognizer")
        self.session = onnxruntime.InferenceSession(
            self.model_dir.joinpath("./model.onnx")
        )
        self.classes = _load_labels(self.model_dir)

        self.model_version = self.session.get_modelmeta().version
        structured_info = {
            "model_name": self.__class__.__name__,
            "model_version": self.model_version,
        }
        logger.info(
            f"Machine learning model loaded: {self.__class__},version: {self.session.get_modelmeta().version}",
            extra={"custom_dimensions": structured_info},
        )

        self.class_colors = {
            "blue": "rgb(0, 0, 255)",
            "glass": "rgb(128, 128, 128)",
            "other": "rgb(255, 0, 0)",
            "yellow": "rgb(255, 255, 0)",
        }

    def _get_predictions_from_onnx(self, imgs: List[np.ndarray]) -> list:
        """perform predictions with ONNX Runtime

        Args:
            imgs: pre-processed numpy images

        Returns:
            list: boxes, labels , scores
        """
        sess_input = self.session.get_inputs()
        sess_output = self.session.get_outputs()

        output_names = [output.name for output in sess_output]

        preds = []
        for img in imgs:
            pred = self.session.run(
                output_names=output_names,
                input_feed={sess_input[0].name: img},
            )

            # Handle cases where there is just one or no classes
            pred[1] = np.atleast_1d(pred[1])
            preds.append(pred)
        return preds

    def predict(self, data: list[np.array]):
        """Identify trash in images

        Args:
            data: List of numpy arrays representing images.

        Returns:

        """
        start_time = time.time()

        # Preprocess images
        img_data = [preprocess(img) for img in data]

        # Get prediections
        raw_predictions = self._get_predictions_from_onnx(img_data)

        # Clean up predictions
        predictions = list(map(apply_nms_to_onnx_predictions, raw_predictions))

        # map class name to class index, modifying prediction[1] in every item of predictions
        for prediction in predictions:
            prediction[1] = [self.classes[class_idx] for class_idx in prediction[1]]
        duration_s = time.time() - start_time
        self._log_predictions(predictions, duration_s)
        return predictions

    @staticmethod
    def _log_predictions(predictions, duration):
        no_of_images = len(predictions)
        no_of_images_wo_predictions = 0
        class_stats = defaultdict(int)
        for prediction in predictions:
            if len(prediction[0]) == 0:
                no_of_images_wo_predictions += 1
                continue
            for class_ in prediction[1]:
                class_stats[class_] += 1
        class_stats_as_str = ", ".join(
            [f"{class_}: {count}" for class_, count in class_stats.items()]
        )
        structured_info = {
            "duration": duration,
            "no_of_images": no_of_images,
            "no_of_images_wo_predictions": no_of_images_wo_predictions,
        }
        structured_info.update(class_stats)
        logger.info(
            f"Prediction made. Duration: {duration:.2f}s, No. of images: {no_of_images}, "
            f"No. of images without objects: {no_of_images_wo_predictions}, "
            f"Class stats: {class_stats_as_str}",
            extra={"custom_dimensions": structured_info},
        )

    def plot_predictions(
        self,
        img,
        predictions,
        plot_masks: bool = True,
        plot_bounding_boxes: bool = True,
    ):
        preprocessed_img = preprocess(img, normalize=False)
        # Disregard all classes that are not in the class_colors
        valid_indices = [
            i for i, class_ in enumerate(predictions[1]) if class_ in self.class_colors
        ]
        classes = [class_ for class_ in predictions[1] if class_ in self.class_colors]
        masks = predictions[3][valid_indices]
        bounding_boxes = torch.tensor(predictions[0][valid_indices])
        colors = [self.class_colors[class_] for class_ in classes]

        img_predictions = torch.tensor(
            preprocessed_img[0, :], dtype=torch.uint8
        )
        # Darken image slightly so that the masks are more visible
        img_predictions = adjust_brightness(img_predictions, 0.5)
        if plot_masks:
            img_predictions = draw_segmentation_masks(
                image=img_predictions,
                masks=torch.tensor(masks[:, 0, :, :] > 0.5, dtype=torch.bool),
                colors=colors,
                alpha=0.5,
            )
        if plot_bounding_boxes:
            img_predictions = draw_bounding_boxes(
                image=img_predictions,
                boxes=bounding_boxes,
                colors=colors,
            )

        img_predictions = img_predictions.permute(1, 2, 0).numpy()
        return img_predictions

    def print_stats(self):
        sess_input = self.session.get_inputs()
        sess_output = self.session.get_outputs()
        print(f"No. of inputs : {len(sess_input)}, No. of outputs : {len(sess_output)}")

        for idx, input_ in enumerate(range(len(sess_input))):
            input_name = sess_input[input_].name
            input_shape = sess_input[input_].shape
            input_type = sess_input[input_].type
            print(
                f"{idx} Input name : {input_name}, Input shape : {input_shape}, \
            Input type  : {input_type}"
            )

        for idx, output in enumerate(range(len(sess_output))):
            output_name = sess_output[output].name
            output_shape = sess_output[output].shape
            output_type = sess_output[output].type
            print(
                f" {idx} Output name : {output_name}, Output shape : {output_shape}, \
            Output type  : {output_type}"
            )
