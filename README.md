 🧠 TensorFlow MNIST 手寫數字分類

本專案使用 TensorFlow  建立卷積神經網路（CNN），用來訓練與辨識 MNIST 手寫數字影像，並儲存模型後載入做推論。

---

## 🧩 模型架構

- Conv2D (32 filters, 3x3) + ReLU + MaxPooling
- Conv2D (64 filters, 3x3) + ReLU + MaxPooling
- Flatten 展平成向量
- Dense(128) + ReLU
- Dense(10) + Softmax（輸出 0~9 機率）

---

## 📈 訓練資訊

- 訓練資料：MNIST 60,000 張手寫數字圖像
- 驗證準確率：約 97.7%
- 測試集準確率：約 97.7%
- 儲存模型為：`model_cnn_tf.h5`

---

## 🧪 推論流程

1. 執行 `mnist_cnn_tf_inference.py`
2. 自動載入測試集中的某一張圖片
3. 顯示模型預測結果與正確答案

你可以修改 `index = 0` 來測不同的圖片。

---

## 📂 檔案說明

| 檔案 | 用途 |
|------|------|
| `mnist_cnn_tf.py` | 建立並訓練 CNN 模型，儲存為 `.h5` |
| `mnist_cnn_tf_inference.py` | 載入模型並預測單張 MNIST 測試圖片 |
| `model_cnn_tf.h5` | 訓練好的模型參數，可供推論用 |
| `README.md` | 專案說明文件 |
