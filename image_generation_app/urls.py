# image_generation_app/urls.py
from django.urls import path
from .views import generate_image, home, detect_objects_view, speech_to_text, image_upload_view

app_name = "image_generation_app"
urlpatterns = [
    path('generate_image/', generate_image, name='generate_image'),
    path('home/', home, name='home'),
    path('detect_objects/', detect_objects_view, name='detect_objects'),
    path('speech_to_text/', speech_to_text, name='speech_to_text'),
    path('upload/', image_upload_view, name='upload'),
]