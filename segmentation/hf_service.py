import base64
import json
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
_prediction_slot = threading.BoundedSemaphore(1)


class ServiceBusyError(Exception):
    pass


def to_data_url(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("ascii")

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


def predict_image(image):
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
            image.save(input_path, format="PNG")

            token = os.environ.get("HF_TOKEN")

            if not token:
                raise RuntimeError("Chưa cấu hình biến môi trường HF_TOKEN.")

            # Tương thích cả phiên bản Gradio Client cũ và mới.
            token_parameter = (
                "token"
                if "token" in inspect.signature(Client).parameters
                else "hf_token"
            )

            client = Client(
                settings.HF_API_URL,
                verbose=False,
                download_files=str(output_dir),
                max_workers=1,
                httpx_kwargs={"timeout": 120.0},
                **{token_parameter: token},
            )

            response = client.predict(
                image=handle_file(str(input_path)),
                api_name="/predict",
            )

            if not isinstance(response, (tuple, list)) or len(response) != 3:
                raise ValueError(
                    "API phải trả về mask_path, overlay_path, info."
                )

            mask_path, overlay_path, info = response

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

            result = {
                "original": to_data_url(original),
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

            return result

    finally:
        _prediction_slot.release()