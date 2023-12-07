# image_generation_app/forms.py
from django import forms

class ImageGenerationForm(forms.Form):
    prompt = forms.CharField(label='Enter your prompt', max_length=100)

class ImageUploadForm(forms.Form):
    image = forms.ImageField()

class UploadImageForm(forms.Form):
    image = forms.ImageField()

class SpeechToTextForm(forms.Form):
    pass  # No need for any fields in this case
