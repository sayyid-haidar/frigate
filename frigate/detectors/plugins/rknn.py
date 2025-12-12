import logging
import os.path
import urllib.request
from typing import Literal

import numpy as np

try:
    from hide_warnings import hide_warnings
except:  # noqa: E722

    def hide_warnings(func):
        pass


from pydantic import Field

from frigate.detectors.detection_api import DetectionApi
from frigate.detectors.detector_config import BaseDetectorConfig

logger = logging.getLogger(__name__)

DETECTOR_KEY = "rknn"

supported_socs = ["rk3562", "rk3566", "rk3568", "rk3588"]

# Updated model definitions with support for multiple resolutions
# Format: model_name -> (suffix, supported_sizes)
yolov8_models = {
    # 320x320 models (legacy naming for backward compatibility)
    "default-yolov8n": {"suffix": "n", "size": 320},
    "default-yolov8s": {"suffix": "s", "size": 320},
    "default-yolov8m": {"suffix": "m", "size": 320},
    "default-yolov8l": {"suffix": "l", "size": 320},
    "default-yolov8x": {"suffix": "x", "size": 320},
    # 320x320 models (explicit size)
    "default-yolov8n-320": {"suffix": "n", "size": 320},
    "default-yolov8s-320": {"suffix": "s", "size": 320},
    "default-yolov8m-320": {"suffix": "m", "size": 320},
    "default-yolov8l-320": {"suffix": "l", "size": 320},
    "default-yolov8x-320": {"suffix": "x", "size": 320},
    # 640x640 models (better accuracy for distant objects)
    "default-yolov8n-640": {"suffix": "n", "size": 640},
    "default-yolov8s-640": {"suffix": "s", "size": 640},
    "default-yolov8m-640": {"suffix": "m", "size": 640},
    "default-yolov8l-640": {"suffix": "l", "size": 640},
    "default-yolov8x-640": {"suffix": "x", "size": 640},
}

# RKNN model download base URL (updated to latest version)
RKNN_MODEL_VERSION = "v2.0.0"
RKNN_MODEL_BASE_URL = "https://github.com/airockchip/rknn-model-zoo/releases/download"


class RknnDetectorConfig(BaseDetectorConfig):
    type: Literal[DETECTOR_KEY]
    core_mask: int = Field(default=0, ge=0, le=7, title="Core mask for NPU.")


class Rknn(DetectionApi):
    type_key = DETECTOR_KEY

    def __init__(self, config: RknnDetectorConfig):
        # create symlink for Home Assistant add on
        if not os.path.isfile("/proc/device-tree/compatible"):
            if os.path.isfile("/device-tree/compatible"):
                os.symlink("/device-tree/compatible", "/proc/device-tree/compatible")

        # find out SoC
        try:
            with open("/proc/device-tree/compatible") as file:
                soc = file.read().split(",")[-1].strip("\x00")
        except FileNotFoundError:
            logger.error("Make sure to run docker in privileged mode.")
            raise Exception("Make sure to run docker in privileged mode.")

        self.soc = soc
        if soc not in supported_socs:
            logger.error(
                "Your SoC is not supported. Your SoC is: {}. Currently these SoCs are supported: {}.".format(
                    soc, supported_socs
                )
            )
            raise Exception(
                "Your SoC is not supported. Your SoC is: {}. Currently these SoCs are supported: {}.".format(
                    soc, supported_socs
                )
            )

        if not os.path.isfile("/usr/lib/librknnrt.so"):
            if "rk356" in soc:
                os.rename("/usr/lib/librknnrt_rk356x.so", "/usr/lib/librknnrt.so")
            elif "rk3588" in soc:
                os.rename("/usr/lib/librknnrt_rk3588.so", "/usr/lib/librknnrt.so")

        self.model_path = config.model.path or "default-yolov8n"
        self.core_mask = config.core_mask
        self.height = config.model.height
        self.width = config.model.width

        if self.model_path in yolov8_models:
            model_info = yolov8_models[self.model_path]
            model_suffix = model_info["suffix"]
            model_size = model_info["size"]

            # Check if using default 320 model that ships with image
            if self.model_path == "default-yolov8n":
                self.model_path = "/models/rknn/yolov8n-320x320-{soc}.rknn".format(
                    soc=soc
                )
            else:
                self.model_path = "/config/model_cache/rknn/yolov8{suffix}-{size}x{size}-{soc}.rknn".format(
                    suffix=model_suffix, size=model_size, soc=soc
                )

                os.makedirs("/config/model_cache/rknn", exist_ok=True)
                if not os.path.isfile(self.model_path):
                    logger.info(
                        "Downloading yolov8{suffix} {size}x{size} model for {soc}...".format(
                            suffix=model_suffix, size=model_size, soc=soc
                        )
                    )
                    # Try new URL format first, fall back to legacy
                    download_urls = [
                        # New airockchip model zoo format
                        "{base}/{version}/yolov8{suffix}-{size}x{size}-{soc}.rknn".format(
                            base=RKNN_MODEL_BASE_URL,
                            version=RKNN_MODEL_VERSION,
                            suffix=model_suffix,
                            size=model_size,
                            soc=soc,
                        ),
                        # Legacy MarcA711 format for 320x320
                        "https://github.com/MarcA711/rknn-models/releases/download/v1.6.0-{soc}/yolov8{suffix}-{size}x{size}-{soc}.rknn".format(
                            soc=soc, suffix=model_suffix, size=model_size
                        ),
                        # Fallback to v1.5.2
                        "https://github.com/MarcA711/rknn-models/releases/download/v1.5.2-{soc}/yolov8{suffix}-{size}x{size}-{soc}.rknn".format(
                            soc=soc, suffix=model_suffix, size=model_size
                        ),
                    ]

                    download_success = False
                    for url in download_urls:
                        try:
                            logger.info(f"Trying to download from: {url}")
                            urllib.request.urlretrieve(url, self.model_path)
                            download_success = True
                            logger.info(f"Successfully downloaded model from: {url}")
                            break
                        except Exception as e:
                            logger.warning(f"Failed to download from {url}: {e}")
                            continue

                    if not download_success:
                        logger.error(
                            "Failed to download model. Please download manually and place in /config/model_cache/rknn/"
                        )
                        raise Exception("Failed to download RKNN model")

            # Validate model dimensions match config
            if (config.model.width != model_size) or (
                config.model.height != model_size
            ):
                logger.error(
                    f"Model size mismatch! Model '{self.model_path}' requires {model_size}x{model_size}, "
                    f"but config specifies {config.model.width}x{config.model.height}. "
                    f"Please set model width and height to {model_size} in your config.yml."
                )
                raise Exception(
                    f"Make sure to set the model width and height to {model_size} in your config.yml."
                )

            if config.model.input_pixel_format != "bgr":
                logger.error(
                    'Make sure to set the model input_pixel_format to "bgr" in your config.yml.'
                )
                raise Exception(
                    'Make sure to set the model input_pixel_format to "bgr" in your config.yml.'
                )

            if config.model.input_tensor != "nhwc":
                logger.error(
                    'Make sure to set the model input_tensor to "nhwc" in your config.yml.'
                )
                raise Exception(
                    'Make sure to set the model input_tensor to "nhwc" in your config.yml.'
                )

        from rknnlite.api import RKNNLite

        self.rknn = RKNNLite(verbose=False)
        if self.rknn.load_rknn(self.model_path) != 0:
            logger.error("Error initializing rknn model.")
        if self.rknn.init_runtime(core_mask=self.core_mask) != 0:
            logger.error(
                "Error initializing rknn runtime. Do you run docker in privileged mode?"
            )

        logger.info(
            f"RKNN detector initialized: SoC={soc}, Model={self.model_path}, "
            f"Size={self.width}x{self.height}, CoreMask={self.core_mask}"
        )

    def __del__(self):
        self.rknn.release()

    def postprocess(self, results):
        """
        Processes yolov8 output.

        Args:
        results: array with shape: (1, 84, n, 1) where n depends on yolov8 model size (for 320x320 model n=2100)

        Returns:
        detections: array with shape (20, 6) with 20 rows of (class, confidence, y_min, x_min, y_max, x_max)
        """

        results = np.transpose(results[0, :, :, 0])  # array shape (2100, 84)
        scores = np.max(
            results[:, 4:], axis=1
        )  # array shape (2100,); max confidence of each row

        # remove lines with score scores < 0.4
        filtered_arg = np.argwhere(scores > 0.4)
        results = results[filtered_arg[:, 0]]
        scores = scores[filtered_arg[:, 0]]

        num_detections = len(scores)

        if num_detections == 0:
            return np.zeros((20, 6), np.float32)

        if num_detections > 20:
            top_arg = np.argpartition(scores, -20)[-20:]
            results = results[top_arg]
            scores = scores[top_arg]
            num_detections = 20

        classes = np.argmax(results[:, 4:], axis=1)

        boxes = np.transpose(
            np.vstack(
                (
                    (results[:, 1] - 0.5 * results[:, 3]) / self.height,
                    (results[:, 0] - 0.5 * results[:, 2]) / self.width,
                    (results[:, 1] + 0.5 * results[:, 3]) / self.height,
                    (results[:, 0] + 0.5 * results[:, 2]) / self.width,
                )
            )
        )

        detections = np.zeros((20, 6), np.float32)
        detections[:num_detections, 0] = classes
        detections[:num_detections, 1] = scores
        detections[:num_detections, 2:] = boxes

        return detections

    @hide_warnings
    def inference(self, tensor_input):
        return self.rknn.inference(inputs=tensor_input)

    def detect_raw(self, tensor_input):
        output = self.inference(
            [
                tensor_input,
            ]
        )
        return self.postprocess(output[0])
