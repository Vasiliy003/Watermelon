from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from preparing_files.gen_create import train_generator, test_generator
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.config import list_physical_devices
from tensorflow.keras.regularizers import l2
import numpy as np

# Определяем модель
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 3),
        kernel_regularizer=l2(0.001)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu',
           kernel_regularizer=l2(0.001)),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu',
           kernel_regularizer=l2(0.001)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.5),
    Dense(7, activation='softmax')  # 7 классов эмоций
])

# model = Sequential([
#     # Первый сверточный блок
#     Conv2D(16, (3, 3), activation='relu', input_shape=(48, 48, 3)),
#     MaxPooling2D((2, 2)),
#
#     # Второй сверточный блок
#     Conv2D(32, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),
#
#     # Полносвязные слои
#     Flatten(),
#     Dense(64, activation='relu'),
#     Dropout(0.5),  # Снижение переобучения
#     Dense(7, activation='softmax')  # 7 классов эмоций
# ])

# Компиляция модели
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

loaded_model = load_model("model100.keras")

class_weights = compute_class_weight('balanced', classes=np.unique(train_generator.classes), y=train_generator.classes)
class_weights = dict(enumerate(class_weights))

# Обучение модели
history = loaded_model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=100,  # Выберите количество эпох
    class_weight=class_weights,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=test_generator.samples // test_generator.batch_size,
)

loaded_model.save("model200.keras")

import matplotlib.pyplot as plt

# Построить графики
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

test_loss, test_accuracy = loaded_model.evaluate(test_generator)
print(f"Точность на тестовых данных: {test_accuracy*100:.2f}%")
