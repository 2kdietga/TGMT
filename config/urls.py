from django.urls import path

from segmentation.views import health, index


urlpatterns = [
    path("", index, name="home"),
    path("health/", health, name="health"),
]