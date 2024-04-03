"""Module for logging user feedback.
"""
from utils.logger_config import configure_logger

logger = configure_logger(__name__)


def log_feedback(user_response: dict, reason: str, image_name: str):
    """Log user feedback.

    This function logs user feedback received from streamlit-feedback to the application's logger.
    Feedback is intended to be logged in the context of specific images, for this reason the image name
    should be supplied.

    Args:
        user_response: Feedback data received from streamlit-feedback.
        reason: Context from which the feedback was received, e.g. "image recognition". This helps in
            log interpretation.
        image_name: Name of the image for which the feedback was received.
    """
    structured_info = {
        "reason": reason,
        "image_name": image_name,
    }
    structured_info.update(user_response)
    logger.info(
        f"Feedback received: {reason}, file name: {image_name}, response: {user_response}",
        extra={'custom_dimensions': structured_info}
    )
