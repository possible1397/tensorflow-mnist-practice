import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import numpy as np

# ✅ 載入 MNIST 資料集（自動分訓練與測試）
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# ✅ 預處理：正規化 + 加 channel 維度
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = np.expand_dims(x_train, -1)  # → (batch, 28, 28, 1)
x_test = np.expand_dims(x_test, -1)

# ✅ 模型建立：CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# ✅ 編譯模型
model.compile(optimizer='sgd',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ✅ 訓練模型
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

# ✅ 評估模型
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"測試集準確率：{test_acc * 100:.2f}%")

# ✅ 儲存模型（包含架構與參數）
model.save("model_cnn_tf.h5")
print("✅ 模型已儲存為 model_cnn_tf.h5")
