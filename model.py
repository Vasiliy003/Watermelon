from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, BatchNormalization
from preparing_files.gen_create import train_generator, test_generator
from keras.callbacks import ModelCheckpoint,EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.config import list_physical_devices
from tensorflow.keras.regularizers import l2
import numpy as np
from keras import regularizers

# weight_decay = 1e-4
# num_classes = 7
#
# model = Sequential([
#     Conv2D(64, (4,4), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=(48,48,3)),
#     Activation('elu'),
#     BatchNormalization(),
#     Conv2D(64, (4,4), padding='same', kernel_regularizer=regularizers.l2(weight_decay)),
#     Activation('elu'),
#     BatchNormalization(),
#     MaxPooling2D(pool_size=(2,2)),
#     Dropout(0.2),
#
#     Conv2D(128, (4,4), padding='same', kernel_regularizer=regularizers.l2(weight_decay)),
#     Activation('elu'),
#     BatchNormalization(),
#     MaxPooling2D(pool_size=(2,2)),
#     Dropout(0.3),
#
#     Conv2D(128, (4, 4), padding='same', kernel_regularizer=regularizers.l2(weight_decay)),
#     Activation('elu'),
#     BatchNormalization(),
#     Conv2D(128, (4, 4), padding='same', kernel_regularizer=regularizers.l2(weight_decay)),
#     Activation('elu'),
#     BatchNormalization(),
#     MaxPooling2D(pool_size=(2,2)),
#     Dropout(0.4),
#     Flatten(),
#     Dense(128, activation="linear"),
#     Activation('elu'),
#     Dense(num_classes, activation='softmax')
# ])

loaded_model = load_model("new.keras")

# class_weights = compute_class_weight('balanced', classes=np.unique(train_generator.classes), y=train_generator.classes)
# class_weights = dict(enumerate(class_weights))

checkpointer = [EarlyStopping(monitor = 'val_accuracy', verbose = 1, restore_best_weights=True,mode="max",patience = 10),
                ModelCheckpoint(
                    filepath='new.keras',
                    monitor="val_accuracy",
                    verbose=1,
                    save_best_only=True,
                    mode="max")]

# Обучение модели
history = loaded_model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=10,  # Выберите количество эпох
    callbacks=[checkpointer],
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=test_generator.samples // test_generator.batch_size,
)

# loaded_model.save("best_model10.keras")

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
