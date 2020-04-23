# Keras/Tensorflow Implementation of YOLOv2

1. 這是 [YOLOv2](https://arxiv.org/abs/1612.08242) 的 keras/tensorflow 實現。
2. [我的 YOLOv2 筆記](https://docs.google.com/document/d/1AVYrpEHitYIbPsDnhhdpOoQoKsceGuzfKxVUy15F7MQ/edit)。
3. YOLOv2 pretrined weights (weights.h5) 是從[這裡](https://drive.google.com/drive/folders/1WjjuImQB0WbweNsbMcaOWSdqVFCKayS3)下載的。
4. 檔案說明：
   * model.py：建立 YOLOv2 模型。
   * dataprocessing.py：將原始檔案 (.xml) 轉成模型的輸入型式。
   * evaluate.py：計算模型的 mAP、展示模型的預測結果。
   * data：測試資料。
5. 若要計算模型在 COCO 2014 上的效能，請將 COCO 2014 資料集中的 images 與 annotations 分別放在 data/COCO_2014/images 與 data/COCO_2014/annotations 中。並利用 main.py 中的程式測試之。
6. 輸出結果範例：
   * <img src="https://i.imgur.com/nCmkDD6.png" width="300" height="300">
   * <img src="https://i.imgur.com/Voh6jSW.png" width="300" height="300">
