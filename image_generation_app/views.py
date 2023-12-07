# image_generation_app/views.py
from django.shortcuts import render
from django.conf import settings
from django.http import FileResponse, JsonResponse

from image_generation_project.settings import BASE_DIR
from .forms import ImageGenerationForm, ImageUploadForm, UploadImageForm
from diffusers import StableDiffusionPipeline
from transformers import DetrImageProcessor, DetrForObjectDetection, AutoTokenizer, VisionEncoderDecoderModel, ViTImageProcessor
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import torch
import os
from django.views.decorators.csrf import csrf_exempt
from PIL import Image
import speech_recognition as sr
from django.core.files.storage import FileSystemStorage
from transformers import pipeline
from datetime import datetime
import numpy as np

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
object_detection_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}


def home(request):
    return render(request, 'image_generation_app/home.html')

def generate_image(request):
    if request.method == 'POST':
        form = ImageGenerationForm(request.POST)
        if form.is_valid():
            prompt = form.cleaned_data['prompt']

            # Load the model
            model_id = "runwayml/stable-diffusion-v1-5"
            pipe = StableDiffusionPipeline.from_pretrained(model_id)
            pipe = pipe.to("cpu")

            # Generate the image
            image = pipe(prompt).images[0]

            # Save the image to the local system
            save_path = os.path.join(settings.MEDIA_ROOT, 'generated_images', 'generated_image.png')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            image.save(save_path)

            # Serve the image as a FileResponse
            response = FileResponse(open(save_path, 'rb'), content_type='image/png')
            return response
    else:
        form = ImageGenerationForm()

    return render(request, 'image_generation_app/generate_image.html', {'form': form})

def predict_caption(image_paths):
    images = []
    for image_path in image_paths:
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")

        images.append(i_image)

    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds[0] if preds else None

def image_upload_view(request):
    if request.method == 'POST':
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            image_path = handle_uploaded_file(request.FILES['image'])
            caption = predict_caption([image_path])
            return render(request, 'image_generation_app/image_caption.html', {'caption': caption, 'image_path': image_path})
    else:
        form = ImageUploadForm()
    return render(request, 'image_generation_app/upload.html', {'form': form})

def handle_uploaded_file(f):
    image_path = '/home/sakhaglobal/GPU/image_generation_project/image_generation_app/static/' + f.name
    with open(image_path, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)
    return image_path

def detect_objects(image_path):
    # Load and preprocess the image
    full_image_path = os.path.join(settings.MEDIA_ROOT, image_path)
    image = Image.open(full_image_path)
    inputs = processor(images=image, return_tensors="pt")

    # Perform object detection
    outputs = object_detection_model(**inputs)

    # Post-process the outputs
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]

    # Extract labels and boxes
    labels = results['labels'].detach().numpy()
    boxes = results['boxes'].detach().numpy()

    # Get the detected objects and their labels
    detected_objects = []
    for label, box in zip(labels, boxes):
        detected_objects.append({
            'label': object_detection_model.config.id2label[label],
            'box': {
                'x_min': box[0],
                'y_min': box[1],
                'x_max': box[2],
                'y_max': box[3],
            }
        })

    return detected_objects


def detect_objects_view(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_image = request.FILES['image']
            fs = FileSystemStorage()
            image_path = fs.save(uploaded_image.name, uploaded_image)

            # Perform object detection
            detection_result = detect_objects(image_path)

            return render(request, 'image_generation_app/detection_result.html', {'image_path': image_path, 'detection_result': detection_result})
    else:
        form = ImageUploadForm()
    return render(request, 'image_generation_app/upload_image.html', {'form': form})



def speech_to_text(request):
    if request.method == 'POST':
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
            
        try:
            text = recognizer.recognize_google(audio)
            return render(request, 'image_generation_app/result.html', {'text': text})
        except sr.UnknownValueError:
            return render(request, 'image_generation_app/result.html', {'text': 'Could not understand audio'})
        except sr.RequestError:
            return render(request, 'image_generation_app/result.html', {'text': 'Speech service is unavailable'})
    return render(request, 'image_generation_app/index.html')

# def speech_to_text(request):
#     if request.method == 'POST':
#         try:
#             # Extract the audio data from the request
#             audio_data = request.FILES.get('audio_data')

#             if audio_data:
#                 # Generate a unique filename using a timestamp
#                 timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#                 audio_filename = f'audio_{timestamp}.wav'

#                 # Save the audio data to the file
#                 with open(os.path.join('image_generation_project/image_generation_app/audio', audio_filename), 'wb') as audio_file:
#                     audio_file.write(audio_data.read())

#                 # Load pre-trained ASR model
#                 asr_pipeline = pipeline(task='automatic-speech-recognition', model='facebook/wav2vec2-base-960h', device="cpu")

#                 # Perform speech-to-text transcription
#                 transcriptions = asr_pipeline(os.path.join('image_generation_project/image_generation_app/audio', audio_filename))

#                 text_result = transcriptions[0]['sentence']

#                 # Return the transcribed text as JSON
#                 return JsonResponse({'text_result': text_result, 'error_message': None})

#             else:
#                 return JsonResponse({'error_message': 'No audio data found in the request.'}, status=400)

#         except Exception as e:
#             error_message = str(e)
#             print(f"Error during transcription: {error_message}")
#             return JsonResponse({'error_message': error_message}, status=400)

#     # Return a default response for GET requests
#     return render(request, 'image_generation_app/speech_to_text_live.html')