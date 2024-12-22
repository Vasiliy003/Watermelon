import cv2

def crop(path, percent):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('../haarcascade_frontalface_alt2.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    counter = 1

    for (x, y, w, h) in faces:

        x_new = max(0, x - int(percent / 100  * w))  # Увеличиваем координаты влево
        y_new = max(0, y - int(percent / 100 * h))  # Увеличиваем координаты вверх
        w_new = min(img.shape[1] - x_new, w + int(percent / 50 * w))  # Увеличиваем ширину
        h_new = min(img.shape[0] - y_new, h + int(percent / 50 * h))  # Увеличиваем высоту

        faces = img[y_new:y_new + h_new, x_new:x_new + w_new]

        cv2.imshow("face", faces)
        cv2.imwrite(f'../croped_files/face_crop{counter}_{percent}%.jpg', faces)
        counter += 1

if __name__ == '__main__':
    path = '../test_images/dima.JPG'
    crop(path, 0)