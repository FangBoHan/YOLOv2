# Keras/Tensorflow Implementation of YOLOv2

1. 這是 [YOLOv2](https://arxiv.org/abs/1612.08242) 的 keras/tensorflow 實現。
2. YOLOv2 pretrined weights (weights.h5) 是從[這裡](https://drive.google.com/drive/folders/1WjjuImQB0WbweNsbMcaOWSdqVFCKayS3)下載的。
3. 檔案說明：
   * model.py：建立 YOLOv2 模型。
   * dataprocessing.py：將原始檔案 (.xml) 轉成模型的輸入型式、展示模型的預測結果。
   * data：測試資料。
4. 輸出結果範例：
   * <img src="https://i.imgur.com/nCmkDD6.png" width="300" height="300">
   * <img src="https://i.imgur.com/Voh6jSW.png" width="300" height="300">
