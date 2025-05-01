import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

# ✅ 載入已儲存的模型（需先訓練並存成 model_cnn_tf.h5）
model = load_model("model_cnn_tf.h5")

# ✅ 載入 MNIST 測試資料
(_, _), (x_test, y_test) = mnist.load_data()

# ✅ 預處理：正規化 + 加通道維度
x_test = x_test.astype("float32") / 255.0
x_test = np.expand_dims(x_test, -1)  # shape: (batch, 28, 28, 1)

# ✅ 從測試集中選一張圖片（你可換 index 看不同預測）
index = 0
image = x_test[index]
label = y_test[index]

# ✅ 模型預測
image_input = np.expand_dims(image, axis=0)  # shape: (1, 28, 28, 1)
prediction = model.predict(image_input)
predicted_label = np.argmax(prediction)

# ✅ 顯示影像與預測結果
plt.imshow(image.squeeze(), cmap="gray")
plt.title(f"預測：{predicted_label}（正確：{label}）")
plt.axis("off")
plt.show()
