import numpy as np
import torch

from torchvision.ops.boxes import nms


def apply_nms_to_onnx_predictions(predictions: list) -> list:
    """Apply Non-Maximum Suppression (NMS) to the predictions obtained from an ONNX model

    This function applies Non-Maximum Suppression (NMS) to the predictions obtained from an ONNX model.
    NMS is a technique used in object detection to select the best bounding box from a set of overlapping boxes.

    Args:
        predictions: A list containing the predictions from an ONNX model.
        The list is expected to have the following structure:
        - predictions[0]: Bounding boxes
        - predictions[2]: Confidence scores for each bounding box

    Returns:
        list: A list containing the filtered predictions after applying NMS.
        The list has the same structure as the input list, but only includes the selected bounding boxes.
    """
    if len(predictions[1]) == 0:
        return predictions

    # Convert the bounding boxes and confidence scores to PyTorch tensors
    boxes, scores = map(torch.tensor, (predictions[0], predictions[2]))

    # Apply NMS to the bounding boxes using the confidence scores with the specified intersection
    # over union (IoU) threshold
    new_boxes_nx = nms(boxes, scores, 0.01)

    # Create a new list of predictions that only includes the selected bounding boxes
    res = [predictions[i][new_boxes_nx] if len(new_boxes_nx) > 1 else predictions[i][new_boxes_nx, np.newaxis] for i, _ in enumerate(predictions)]

    return res
