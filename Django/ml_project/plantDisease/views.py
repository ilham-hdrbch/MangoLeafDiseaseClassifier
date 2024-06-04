# plantDisease/views.py

from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.contrib import messages
import cv2
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.keras.models import load_model
import tensorflow as tf


model = load_model('models/Mango_classifier.keras')
class_names = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back', 'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould']

def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence

def is_tree_image(im):
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_ratio = cv2.countNonZero(mask) / (im.shape[0] * im.shape[1])
    return green_ratio > 0.2 

def index(request):
    if request.method == 'POST' and request.FILES.get('image'):
        img = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(img.name, img)
        uploaded_file_url = fs.url(filename)

        img_path = fs.path(filename)
        image = cv2.imread(img_path)

        if image is None:
            messages.error(request, "Failed to read the uploaded image.")
            return render(request, 'form.html')
 

        if is_tree_image(image):
            cascade_path = os.path.join(os.path.dirname(__file__), 'cascades', 'cascade1.xml')
            leaf_cascade = cv2.CascadeClassifier(cascade_path)
            leaves = leaf_cascade.detectMultiScale(image, scaleFactor=1.05, minNeighbors=3)
            if len(leaves) == 0:
                messages.warning(request, "No leaf detected!")
            else:
                largest_leaf = None
                max_area = 0
                for (x, y, w, h) in leaves:
                    area = w * h
                    if area > max_area:
                        max_area = area
                        largest_leaf = (x, y, w, h)

                if largest_leaf:
                    x, y, w, h = largest_leaf
                    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    largest_leaf_img = image[y:y + h, x:x + w]
                    resized_leaf_img = cv2.resize(largest_leaf_img, (224, 224))
                    normalized_leaf_img = resized_leaf_img.astype('float32') / 255.0
                    
                    predicted_class, confidence = predict(model, resized_leaf_img )
                    normalized_leaf_img = (normalized_leaf_img * 255).astype(np.uint8)
                    normalized_leaf_img_path = os.path.join(fs.location, f'normalized_{img.name}')
                    cv2.imwrite(normalized_leaf_img_path, normalized_leaf_img)
                    uploaded_file_url = fs.url(f'normalized_{img.name}')

                    return render(request, 'result.html', {'uploaded_file_url': uploaded_file_url, 'predicted_class': predicted_class, 'confidence': confidence})
                else:
                    messages.warning(request, "No leaf detected!")

        else:
            resized_img = cv2.resize(image, (224, 224))
            normalized_img = resized_img.astype('float32') / 255.0
            
            predicted_class, confidence = predict(model, resized_img)
            normalized_img = (normalized_img * 255).astype(np.uint8)
            normalized_img_path = os.path.join(fs.location, f'normalized_{img.name}')
            cv2.imwrite(normalized_img_path, normalized_img)
            uploaded_file_url = fs.url(f'normalized_{img.name}')

            return render(request, 'result.html', {'uploaded_file_url': uploaded_file_url, 'predicted_class': predicted_class, 'confidence': confidence})

    return render(request, 'form.html')
