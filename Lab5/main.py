import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1)) / 255.0
test_images = test_images.reshape((10000, 28, 28, 1)) / 255.0

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 цифр
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Навчання моделі...")
model.fit(train_images, train_labels, epochs=3, validation_data=(test_images, test_labels))

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"\nТочність на тестових зображеннях: {test_acc:.4f}")

predictions = model.predict(test_images[:5])

for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(test_images[i].reshape(28, 28), cmap='gray')
    plt.title(f"Прогноз: {np.argmax(predictions[i])}")
    plt.axis('off')
plt.suptitle("Навчальні/тестові зображення")
plt.show()

noise_factor = 0.3
noisy_images = test_images[:5] + noise_factor * np.random.normal(size=test_images[:5].shape)
noisy_images = np.clip(noisy_images, 0., 1.)

predictions_noisy = model.predict(noisy_images)

for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(noisy_images[i].reshape(28, 28), cmap='gray')
    plt.title(f"Прогноз: {np.argmax(predictions_noisy[i])}")
    plt.axis('off')
plt.suptitle("Зображення з помилками / шумом")
plt.show()