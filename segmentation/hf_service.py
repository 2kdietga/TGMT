import base64
import json
import logging
from contextlib import contextmanager
import threading
import time
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

from django.conf import settings
from gradio_client import Client, handle_file
from PIL import Image
import os
import inspect


# Bản demo: tối đa một lượt gọi API mỗi tiến trình.
# Request khác sẽ nhận thông báo bận thay vì dồn vào hàng đợi local.
logger = logging.getLogger(__name__)


@contextmanager
def timed_stage(timings, name, request_id):
    started = time.perf_counter()
    logger.info("prediction request=%s stage=%s started", request_id, name)
    try:
        yield
    finally:
        timings[name] = round((time.perf_counter() - started) * 1000, 1)
        logger.info("prediction request=%s stage=%s duration_ms=%.1f",
                    request_id, name, timings[name])


_prediction_slot = threading.BoundedSemaphore(1)
_cached_client = None
_cached_client_key = None


def discard_client():
    global _cached_client, _cached_client_key
    client = _cached_client
    _cached_client = None
    _cached_client_key = None
    if client is not None:
        client.close()
        for name in ("executor", "helper_executor", "stream_executor"):
            executor = getattr(client, name, None)
            if executor is not None:
                executor.shutdown(wait=False, cancel_futures=True)


def get_client(output_dir, request_id):
    # Called only while holding _prediction_slot: output paths cannot overlap.
    global _cached_client, _cached_client_key
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not configured.")
    key = (os.getpid(), settings.HF_API_URL, token)
    reused = _cached_client is not None and _cached_client_key == key
    if not reused:
        discard_client()
        token_parameter = (
            "token" if "token" in inspect.signature(Client).parameters else "hf_token"
        )
        _cached_client = Client(
            settings.HF_API_URL,
            verbose=False,
            download_files=str(output_dir),
            max_workers=1,
            httpx_kwargs={"timeout": 120.0},
            **{token_parameter: token},
        )
        _cached_client_key = key
    else:
        _cached_client.reset_session()
        # gradio_client 2.7.2 downloads using client.output_dir.
        _cached_client.output_dir = str(output_dir)
        _cached_client.download_files = str(output_dir)
    logger.info("prediction request=%s client_reused=%s", request_id, reused)
    return _cached_client


class ServiceBusyError(Exception):
    pass


def to_data_url(image, *, png_path=None):
    # Reuse the normalized input PNG when the preview has not been resized.
    if png_path is not None:
        png_bytes = Path(png_path).read_bytes()
    else:
        buffer = BytesIO()
        image.save(buffer, format="PNG", compress_level=3)
        png_bytes = buffer.getvalue()

    encoded = base64.b64encode(png_bytes).decode("ascii")

    return f"data:image/png;base64,{encoded}"


def read_output_image(path):
    if not isinstance(path, (str, Path)):
        raise ValueError("API không trả về đường dẫn ảnh như mong đợi.")

    with Image.open(path) as source:
        width, height = source.size

        if width * height > settings.MAX_IMAGE_PIXELS:
            raise ValueError("Ảnh trả về từ API quá lớn.")

        source.load()
        return source.copy()


def predict_image(image, *, request_id="-", timings=None):
    if timings is None:
        timings = {}
    if not _prediction_slot.acquire(blocking=False):
        raise ServiceBusyError(
            "Hệ thống đang xử lý một ảnh khác. Bạn thử lại sau ít giây nhé."
        )

    started = time.perf_counter()

    try:
        # File input và output được đặt trong thư mục riêng cho mỗi request.
        # Sau khi xử lý xong, thư mục này được dọn tự động.
        with TemporaryDirectory(prefix="skinscope_") as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "input.png"
            output_dir = temp_path / "outputs"
            output_dir.mkdir()

            # Chuẩn hóa định dạng, không resize hoặc normalize cho model.
            # Phần tiền xử lý model thuộc API Hugging Face.
            with timed_stage(timings, "prepare_png", request_id):
                image.save(input_path, format="PNG", compress_level=3)

            logger.info("prediction request=%s input_width=%s input_height=%s png_bytes=%s",
                        request_id, image.width, image.height, input_path.stat().st_size)

            with timed_stage(timings, "client_init", request_id):
                client = get_client(output_dir, request_id)

            try:
                with timed_stage(timings, "predict_upload_queue_download", request_id):
                    response = client.predict(
                        image=handle_file(str(input_path)),
                        api_name="/predict",
                    )
            except Exception:
                discard_client()
                raise

            if not isinstance(response, (tuple, list)) or len(response) != 3:
                raise ValueError(
                    "API phải trả về mask_path, overlay_path, info."
                )

            mask_path, overlay_path, info = response

            with timed_stage(timings, "postprocess", request_id):
                mask = read_output_image(mask_path).convert("L")
                overlay = read_output_image(overlay_path).convert("RGB")

                # Tính diện tích từ mask trắng/đen API trả về.
                # Giả định: trắng là vùng tổn thương, đen là nền.
                histogram = mask.histogram()
                foreground_pixels = sum(histogram[128:])
                total_pixels = mask.width * mask.height

                area_percent = 100 * foreground_pixels / total_pixels

                original = image.copy()

                # Chỉ thu nhỏ bản hiển thị, sau khi API đã xử lý ảnh đầy đủ.
                original.thumbnail((1200, 1200))
                overlay.thumbnail((1200, 1200))
                mask.thumbnail(
                    (1200, 1200),
                    resample=Image.Resampling.NEAREST,
                )

                if isinstance(info, str):
                    try:
                        info = json.loads(info)
                    except json.JSONDecodeError:
                        pass

            with timed_stage(timings, "encode_result", request_id):
                result = {
                    "original": to_data_url(
                        original,
                        png_path=input_path if original.size == image.size else None,
                    ),
                    "mask": to_data_url(mask),
                    "overlay": to_data_url(overlay),
                    "area_percent": round(area_percent, 2),
                    "elapsed": round(time.perf_counter() - started, 2),
                    "info_json": json.dumps(
                        info,
                        ensure_ascii=False,
                        indent=2,
                        default=str,
                    ),
                }

            result["elapsed"] = round(time.perf_counter() - started, 2)
            return result

    finally:
        timings["service_total"] = round((time.perf_counter() - started) * 1000, 1)
        logger.info("prediction request=%s timings_ms=%s", request_id, json.dumps(timings))
        _prediction_slot.release()