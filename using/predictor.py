from sys import prefix
import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from matplotlib import pyplot as plt
import numpy as np
from file_crop import Crop

class Predictor:
    def __init__(self):
        self.model = load_model("../model.weights.best.keras")
        self.crop = Crop()
        self.results = {'0%': [], '10%': [], '15%': [], '25%': [], '30%': []}
        self.path = None
        self.predict_info = {}

    def predict(self, file_path, percent):
        image = load_img(file_path, color_mode='grayscale', target_size=(48, 48))

        image_array = img_to_array(image)
        image_array = np.repeat(image_array, 3, axis=-1)
        image_array = image_array / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        predictions = self.model.predict(image_array)
        prediction_list = list(predictions)
        prediction_list = list(prediction_list[0])
        prediction_list = [round(val.item() * 100, 2) for val in prediction_list]

        # max_number = max([float(item.split('%')[0]) for item in prediction_list])
        max_number = max(prediction_list)
        self.results[f'{percent}%'].append(max_number)


        emotion_dict = {'Злость': prediction_list[0],
                        'Отвращение': prediction_list[1],
                        'Страх': prediction_list[2],
                        'Радость': prediction_list[3],
                        'Грусть': prediction_list[4],
                        'Удивление': prediction_list[5],
                        'Нейтральное': prediction_list[6]}

        predicted_emotion = np.argmax(predictions)
        emotion_labels = ['Злость', 'Отвращение', 'Страх', 'Радость', 'Грусть', 'Удивление', 'Нейтральное']

        print(f'Результаты анализа файла: {file_path.split('/')[-1]}' + '\n')

        if file_path not in self.predict_info:
            for emote in emotion_dict:
                print(f'{emote}: {emotion_dict[emote]}%')

            print(f'Вероятная эмоция: {emotion_labels[predicted_emotion]}' + "\n")
        else:
            results = ''

            for emote in emotion_dict:
                results += f'{emote}: {emotion_dict[emote]}% \n'

            results += f'Вероятная эмоция: {emotion_labels[predicted_emotion]}'

            self.predict_info[file_path] = results

    def save_crop(self, percent, ondelete=True):
        self.crop.crop(self.path, percent)
        photos = os.listdir('../croped_files')
        photos = ['../croped_files/' + photo for photo in photos]

        for photo in photos:
            if ondelete:
                self.predict(photo, percent)
                os.remove(photo)
            else:
                self.predict_info[photo] = 'results'
                self.predict(photo, percent)

    def best_crop(self):
        best = 0
        bestcrop = None
        for key in self.results:
            count = 0
            summ = 0
            for item in self.results[key]:
                count += 1
                summ += item
            result = summ / count
            if result > best:
                best = result
                bestcrop = key
        print(f'Лучший кроп: {bestcrop}, имеет {best}% точности')
        return bestcrop

    def start(self, path):
        self.path = path

        photos = os.listdir('../croped_files')
        photos = ['../croped_files/' + photo for photo in photos]
        for photo in photos:
            os.remove(photo)

        self.save_crop(0)
        self.save_crop(10)
        self.save_crop(15)
        self.save_crop(25)
        self.save_crop(30)

        best_crop = self.best_crop()
        self.save_crop(int(best_crop[:-1]), False)

        return self.predict_info



if __name__ == '__main__':
    predictor = Predictor()
    path = '../test_images/enot.png'
    predictor.start(path)
