# MangoLeafDiseaseClassifier
###Overview
The Plant Disease Classification (Mango) project is a web-based application designed to diagnose mango plant diseases using advanced machine learning techniques. Users can upload images of mango leaves or trees, and the application uses a Convolutional Neural Network (CNN) model to analyze the images and determine the health status of the plants. This tool aims to assist farmers and plant specialists in identifying diseases promptly and taking appropriate measures to manage and treat them, thereby enhancing agricultural productivity and reducing crop losses.

###Development Environment
The project utilizes a variety of tools and technologies, including:

Django: A Python web framework for developing the user interface and server-side logic.
CNN Model: A Convolutional Neural Network for image classification, trained to recognize various mango diseases and healthy plants.
OpenCV: A computer vision library for image processing and manipulation.
TensorFlow and Keras: Libraries for building and training the CNN model.
Jupyter Notebook: Used for developing and experimenting with the machine learning model.
Visual Studio Code: An integrated development environment (IDE) for coding.
Python: The main programming language for the web application and image classification model.
###Key Features
Image Upload: Users can upload images of mango leaves or trees to the web application.
Image Verification: The application checks if the uploaded image is a leaf image:
If the image contains a leaf, it is directly sent to the trained model.
If the image contains leaves but is not solely a leaf, the application extracts the leaf portion using Haar Cascade and sends the processed image to the model.
Disease Classification: The CNN model analyzes the image and provides a diagnosis of the plant's health, identifying any specific diseases.

###Mango Leaf Disease Dataset:
https://www.kaggle.com/datasets/aryashah2k/mango-leaf-disease-dataset

This repository contains all the necessary code and documentation to set up and run the web application, including instructions for training the CNN model, processing images, and  the Django-based web interface.
