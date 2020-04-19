from model import get_yolov2
from dataprocessing import display_yolo
import glob


# MSCOCO 的參數
LABELS = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', \
          'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', \
          'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', \
          'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', \
          'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', \
          'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', \
          'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', \
          'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', \
          'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', \
          'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', \
          'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', \
          'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')
IMAGE_H, IMAGE_W = 416, 416
GRID_H,  GRID_W  = 416 // 32, 416 // 32 # GRID size = IMAGE size / 32
BOX              = 5
CLASS            = len(LABELS)
SCORE_THRESHOLD  = 0.5
IOU_THRESHOLD    = 0.45
ANCHORS          = [0.57273 ,  0.677385, 1.87446 ,  2.06253, 3.33843 ,  5.47434, 7.88282 ,  3.52778, 9.77052 ,  9.16828]


# 載入模型
model = get_yolov2(image_w=IMAGE_W, image_h=IMAGE_H, box_num=BOX, class_count=CLASS)
model.load_weights("weights.h5")

# 展示預測結果
x_files = glob.glob('data/image_416/*.jpg')

for img_path in x_files:
    display_yolo(img_path, model, SCORE_THRESHOLD, IOU_THRESHOLD, ANCHORS)
