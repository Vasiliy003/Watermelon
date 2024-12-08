from tensorflow.keras.models import Sequential, load_model
from preparing_files.gen_create import train_generator, test_generator

loaded_model = load_model("model200.keras")

test_loss, test_accuracy = loaded_model.evaluate(test_generator)
print(f"Точность на тестовых данных: {test_accuracy*100:.2f}%")