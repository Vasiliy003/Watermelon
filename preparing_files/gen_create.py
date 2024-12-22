import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Укажите путь к вашим директориям с данными
train_dir = 'Z:/Github/Watermelon/train'  # Путь к папке с тренировочными данными
test_dir = 'Z:/Github/Watermelon/test'    # Путь к папке с тестовыми данными

# Создание генераторов данных для тренировки и теста
train_datagen = ImageDataGenerator(
    rescale=1.0/255,               # Нормализация изображений в диапазон [0, 1]
    rotation_range=30,             # Повороты изображений
    width_shift_range=0.2,         # Сдвиг по ширине
    height_shift_range=0.2,        # Сдвиг по высоте
    shear_range=0.2,               # Применение сдвига
    zoom_range=0.2,                # Масштабирование изображений
    horizontal_flip=True,          # Горизонтальное отражение
    fill_mode='nearest'            # Заполнение пустых пикселей
)

# Генератор для теста (без аугментации, только нормализация)
test_datagen = ImageDataGenerator(rescale=1.0/255)

# Загрузка изображений из директорий и создание генераторов
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),        # Изменение размера изображений до 48x48
    batch_size=64,               # Размер батча
    class_mode='categorical',     # Классификация с несколькими метками
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48, 48),        # Изменение размера изображений до 48x48
    batch_size=64,
    class_mode='categorical',
)

# Проверим, как работает загрузка
print("Классы: ", train_generator.class_indices)  # Выведем индексы классов
print(f"Количество тренировочных изображений: {train_generator.samples}")
print(f"Количество тестовых изображений: {test_generator.samples}")

