from model import get_yolov2
from evaluate import display_yolo, get_yolo_result, get_COCO_mAP
from dataprocessing import parse_annotation, parse_function, x1y1x2y2_xywh
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
SCORE_THRESHOLD  = 0.6
IOU_THRESHOLD    = 0.45
ANCHORS          = [0.57273 ,  0.677385, 1.87446 ,  2.06253, 3.33843 ,  5.47434, 7.88282 ,  3.52778, 9.77052 ,  9.16828]
IMAGE_COUNT      = 20000


# 載入模型
model = get_yolov2(image_w=IMAGE_W, image_h=IMAGE_H, box_num=BOX, class_count=CLASS)
model.load_weights("weights.h5")

# 展示預測結果
x_files = glob.glob('data/image_416/*.jpg')

for img_path in x_files:
    display_yolo(img_path, model, SCORE_THRESHOLD, IOU_THRESHOLD, ANCHORS)
    
# 計算模型在 COCO 2014 的表現
ann_dir = "data/COCO_2014/annotations"
img_dir = "data/COCO_2014/images"
img_names, ture_boxes = parse_annotation(ann_dir, img_dir, LABELS)  # 這裡的 true_box 是 [x1, y1, x2, y2] 的格式
imgs, ture_boxes = parse_function(img_names, ture_boxes)
for i in range(len(ture_boxes)):    # 將 true_box 轉成 [x, y, w, h]
    ture_boxes[i] = x1y1x2y2_xywh(ture_boxes[i])

boxes = []
scores = []
classes = []
for i in range(IMAGE_COUNT):
    image_path = ""
    b, s, c = get_yolo_result(image_path, model, SCORE_THRESHOLD, IOU_THRESHOLD, ANCHORS)
    boxes.extend(b)
    scores.extend(s)
    classes.extend(c)
    
mAP = get_COCO_mAP(boxes, scores, classes, ture_boxes)
print("mAP =", mAP)
