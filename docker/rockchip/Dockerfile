# syntax=docker/dockerfile:1.6

# https://askubuntu.com/questions/972516/debian-frontend-environment-variable
ARG DEBIAN_FRONTEND=noninteractive

FROM wheels as rk-wheels
COPY docker/main/requirements-wheels.txt /requirements-wheels.txt
COPY docker/rockchip/requirements-wheels-rk.txt /requirements-wheels-rk.txt
RUN sed -i "/https:\/\//d" /requirements-wheels.txt
RUN pip3 wheel --wheel-dir=/rk-wheels -c /requirements-wheels.txt -r /requirements-wheels-rk.txt

FROM deps AS rk-deps
ARG TARGETARCH

RUN --mount=type=bind,from=rk-wheels,source=/rk-wheels,target=/deps/rk-wheels \ 
    pip3 install -U /deps/rk-wheels/*.whl 

WORKDIR /opt/frigate/
COPY --from=rootfs / /

# RKNN Runtime Libraries v2.3.0 (latest stable from airockchip)
# These libraries provide NPU acceleration for RK3588/RK356x series
ADD https://raw.githubusercontent.com/airockchip/rknn-toolkit2/v2.3.0/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so /usr/lib/librknnrt_rk3588.so
ADD https://raw.githubusercontent.com/airockchip/rknn-toolkit2/v2.3.0/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so /usr/lib/librknnrt_rk356x.so

# Default YOLOv8n 320x320 models (ships with image for quick start)
ADD https://github.com/MarcA711/rknn-models/releases/download/v1.5.2-rk3562/yolov8n-320x320-rk3562.rknn /models/rknn/
ADD https://github.com/MarcA711/rknn-models/releases/download/v1.5.2-rk3566/yolov8n-320x320-rk3566.rknn /models/rknn/
ADD https://github.com/MarcA711/rknn-models/releases/download/v1.5.2-rk3568/yolov8n-320x320-rk3568.rknn /models/rknn/
ADD https://github.com/MarcA711/rknn-models/releases/download/v1.5.2-rk3588/yolov8n-320x320-rk3588.rknn /models/rknn/

# Optional: Pre-download 640x640 models for better accuracy (larger file size)
# These will be auto-downloaded if configured but not present
# Uncomment if you want to include in base image:
# ADD https://github.com/MarcA711/rknn-models/releases/download/v1.5.2-rk3588/yolov8n-640x640-rk3588.rknn /models/rknn/
# ADD https://github.com/MarcA711/rknn-models/releases/download/v1.5.2-rk3588/yolov8s-640x640-rk3588.rknn /models/rknn/

# Rockchip FFmpeg with MPP (Media Process Platform) and RGA support
# This enables hardware video encoding/decoding using VPU
RUN rm -rf /usr/lib/btbn-ffmpeg/bin/ffmpeg
RUN rm -rf /usr/lib/btbn-ffmpeg/bin/ffprobe

# FFmpeg 6.1 with full Rockchip hardware acceleration:
# - h264_rkmpp / hevc_rkmpp (hardware decode/encode via VPU)
# - scale_rkrga (hardware scaling via RGA)
ADD --chmod=111 https://github.com/MarcA711/Rockchip-FFmpeg-Builds/releases/download/6.1-2/ffmpeg /usr/lib/btbn-ffmpeg/bin/
ADD --chmod=111 https://github.com/MarcA711/Rockchip-FFmpeg-Builds/releases/download/6.1-2/ffprobe /usr/lib/btbn-ffmpeg/bin/

# Set environment variable to indicate Rockchip build
ENV FRIGATE_ROCKCHIP=1
ENV LIBAVFORMAT_VERSION_MAJOR=60
