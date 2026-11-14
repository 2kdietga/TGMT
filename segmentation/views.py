import logging
import warnings

from django.conf import settings
from django.http import HttpResponse
from django.shortcuts import render
from django.views.decorators.http import require_http_methods
from PIL import Image, ImageOps, UnidentifiedImageError

from .hf_service import predict_image, ServiceBusyError


logger = logging.getLogger(__name__)


def health(request):
    # Health check không gọi API Hugging Face.
    return HttpResponse("ok", content_type="text/plain")


@require_http_methods(["GET", "POST"])
def index(request):
    context = {}

    if request.method == "GET":
        return render(request, "segmentation/index.html", context)

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
        context["result"] = predict_image(image)

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

    return render(request, "segmentation/index.html", context)