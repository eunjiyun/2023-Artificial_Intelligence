import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# CIFAR-10 데이터 로드
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# 픽셀 값을 0~1 사이로 정규화
train_images, test_images = train_images / 255.0, test_images / 255.0

# CNN 모델 구축
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# 모델 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 모델 훈련
model.fit(train_images, train_labels, epochs=10)

# 테스트 이미지 중에서 일부를 사용하여 예측
predictions = model.predict(test_images[:5])

# 예측 결과 출력
for i in range(5):
    predicted_label = tf.argmax(predictions[i])
    true_label = test_labels[i][0]
    print(f"Predicted: {predicted_label.numpy()}, True: {true_label}")

# 검증 결과 시각화
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for i in range(5):
    axes[i].imshow(test_images[i])
    axes[i].set_title(f"Predicted: {tf.argmax(predictions[i])}")
    axes[i].axis('off')

plt.show()