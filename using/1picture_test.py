from sys import prefix
import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from matplotlib import pyplot as plt
import numpy as np
from file_crop import crop

model = load_model("../model.weights.best.keras")

def predict(image_path):
    image = load_img(image_path, color_mode='grayscale', target_size=(48, 48))

    image_array = img_to_array(image)
    image_array = np.repeat(image_array, 3, axis=-1)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array)
    prediction_list = list(predictions)
    prediction_list = list(prediction_list[0])
    prediction_list = [round(val.item() * 100, 2) for val in prediction_list]

    emotion_dict = {'Злость': prediction_list[0],
                    'Отвращение': prediction_list[1],
                    'Страх': prediction_list[2],
                    'Радость': prediction_list[3],
                    'Грусть': prediction_list[4],
                    'Удивление': prediction_list[5],
                    'Нейтральное': prediction_list[6]}

    predicted_emotion = np.argmax(predictions)
    emotion_labels = ['Злость', 'Отвращение', 'Страх', 'Радость', 'Грусть', 'Удивление', 'Нейтральное']

    print(f'Результаты анализа файла: {image_path.split('/')[-1]}' + '\n')

    for emote in emotion_dict:
        print(f'{emote}: {emotion_dict[emote]}%')

    print(f'Вероятная эмоция: {emotion_labels[predicted_emotion]}' + "\n")

path = '../test_images/dima.JPG'
crop(path, 10)

photos = os.listdir('../croped_files')
photos = ['../croped_files/' + photo for photo in photos]

for photo in photos:
    predict(photo)
    os.remove(photo)
