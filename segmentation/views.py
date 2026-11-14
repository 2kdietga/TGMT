import logging
import time
import uuid
from functools import wraps
import warnings

from django.conf import settings
from django.http import HttpResponse
from django.shortcuts import render
from django.views.decorators.http import require_http_methods
from PIL import Image, ImageOps, UnidentifiedImageError

from .hf_service import predict_image, ServiceBusyError, timed_stage


logger = logging.getLogger(__name__)


def log_post_timing(view):
    @wraps(view)
    def wrapped(request, *args, **kwargs):
        if request.method != "POST":
            return view(request, *args, **kwargs)
        request.prediction_id = uuid.uuid4().hex[:12]
        request.prediction_timings = {}
        started = time.perf_counter()
        try:
            response = view(request, *args, **kwargs)
            timings = request.prediction_timings
            timings["view_total"] = round((time.perf_counter() - started) * 1000, 1)
            response["X-Request-ID"] = request.prediction_id
            response["Server-Timing"] = ", ".join(
                f"{name};dur={duration}" for name, duration in timings.items()
            )
            logger.info("prediction request=%s status=%s response_bytes=%s timings_ms=%s",
                        request.prediction_id, response.status_code, len(response.content), timings)
            return response
        finally:
            logger.info("prediction request=%s view_finished_ms=%.1f",
                        request.prediction_id, (time.perf_counter() - started) * 1000)
    return wrapped


def health(request):
    # Health check không gọi API Hugging Face.
    return HttpResponse("ok", content_type="text/plain")


@require_http_methods(["GET", "POST"])
@log_post_timing
def index(request):
    context = {}

    if request.method == "GET":
        return render(request, "segmentation/index.html", context)

    with timed_stage(request.prediction_timings, "parse_upload", request.prediction_id):
        uploaded = request.FILES.get("image")

    if not uploaded:
        context["error"] = "Bạn chưa chọn hoặc chụp ảnh."
        return render(
            request,
            "segmentation/index.html",
            context,
            status=400,
        )

    if uploaded.size > settings.MAX_IMAGE_UPLOAD_BYTES:
        context["error"] = "Ảnh phải nhỏ hơn hoặc bằng 10 MB."
        return render(
            request,
            "segmentation/index.html",
            context,
            status=400,
        )

    try:
        with timed_stage(request.prediction_timings, "decode_image", request.prediction_id):
            with warnings.catch_warnings():
                warnings.simplefilter(
                    "error",
                    Image.DecompressionBombWarning,
                )

                with Image.open(uploaded) as source:
                    if source.format not in {"JPEG", "PNG", "WEBP"}:
                        raise ValueError("Định dạng ảnh không hỗ trợ.")

                    width, height = source.size

                    if width * height > settings.MAX_IMAGE_PIXELS:
                        raise ValueError("Ảnh vượt quá 20 megapixel.")

                    source.load()

                    image = ImageOps.exif_transpose(
                        source
                    ).convert("RGB")

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombWarning,
        Image.DecompressionBombError,
    ):
        context["error"] = (
            "Ảnh không hợp lệ. Hãy dùng JPG, PNG hoặc WebP, "
            "tối đa 10 MB và 20 megapixel."
        )

        return render(
            request,
            "segmentation/index.html",
            context,
            status=400,
        )

    try:
        context["result"] = predict_image(
            image, request_id=request.prediction_id, timings=request.prediction_timings
        )

    except ServiceBusyError as exc:
        context["error"] = str(exc)
        response = render(
            request,
            "segmentation/index.html",
            context,
            status=503,
        )
        response["Retry-After"] = "10"
        return response

    except Exception:
        logger.exception("Gọi API Hugging Face thất bại")

        context["error"] = (
            "Chưa nhận được kết quả từ Hugging Face. "
            "Space có thể đang khởi động, đang bận hoặc gặp lỗi. "
            "Bạn thử lại sau nhé."
        )

        return render(
            request,
            "segmentation/index.html",
            context,
            status=502,
        )

    with timed_stage(request.prediction_timings, "render_html", request.prediction_id):
        return render(request, "segmentation/index.html", context)