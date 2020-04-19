import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def get_yolo_result(image_path, model, score_threshold, iou_threshold, anchors):
    """
    Return bounding boxes and probabilities predicted by "model" (yolov2 model).
    
    Parameters
    ----------
    - image_path : image path.
    - model : YOLO model.
    - score_threshold : threshold used for filtering predicted bounding boxes.
    - iou_threshold : threshold used for non max suppression.
    
    Returns
    -------
    - boxes : bounding boxes predicted, shape = (box_count, 4)
    - scores : scores of bounding boxes, shape = (box_count)
    - classes : classes of bounding boxes, shape = (box_count)
    """
    # 下載 image 並對其做正規劃
    image = cv2.imread(image_path)
    input_image = image[:,:,::-1]
    input_image = image / 255.
    
    # 計算 grid cell 的 width 與 height
    image_w = input_image.shape[1]
    image_h = input_image.shape[0]
    grid_w = image_w // 32
    grid_h = image_h // 32
    
    # 將 image 轉成模型的輸入形式 (416, 416, 3) -> (1, 416, 416, 3)
    input_image = np.expand_dims(input_image, 0)
    
    # y_pred = "model" 的原始預測果
    y_pred = model.predict_on_batch(input_image)

    # 原始預測結果：t_x, t_y, t_w, t_h, t_o, ori_prob
    # 實際預測結果：b_x, b_y, b_w, b_h, confidence, class_prob
    # b_x = c_x + sigmoid(t_x)      (c_x, c_y) : 該 grid cell 的左上角座標
    # b_y = c_y + sigmoid(t_y)
    # b_w = p_w * e^(t_w)           (p_w, p_h) : anchor box 的寬和高
    # b_h = p_h * e^(t_h)
    # confidence = sigmoid(t_o)
    # class_prob = softmax(ori_prob)
    coord_x = tf.cast(tf.reshape(tf.tile(tf.range(grid_w), [grid_h]), (1, grid_h, grid_w, 1, 1)), tf.float32)
    coord_y = tf.transpose(coord_x, (0,2,1,3,4))
    coords = tf.tile(tf.concat([coord_x,coord_y], -1), [10, 1, 1, 5, 1])
    dims = K.cast_to_floatx(K.int_shape(y_pred)[1:3])
    dims = K.reshape(dims,(1,1,1,1,2))
    # anchors tensor
    anchors = np.array(anchors)
    anchors = anchors.reshape(len(anchors) // 2, 2)
    # pred_xy and pred_wh shape (m, grid_w, grid_h, Anchors, 2)
    pred_xy = K.sigmoid(y_pred[:,:,:,:,0:2])
    pred_xy = (pred_xy + coords)
    pred_xy = pred_xy / dims
    pred_wh = K.exp(y_pred[:,:,:,:,2:4])
    pred_wh = (pred_wh * anchors)
    pred_wh = pred_wh / dims
    # pred_confidence
    box_conf = K.sigmoid(y_pred[:,:,:,:,4:5])  
    # pred_class
    box_class_prob = K.softmax(y_pred[:,:,:,:,5:])

    # Reshape
    pred_xy = pred_xy[0,...]
    pred_wh = pred_wh[0,...]
    box_conf = box_conf[0,...]
    box_class_prob = box_class_prob[0,...]

    # Convert box coords from x,y,w,h to x1,y1,x2,y2
    box_xy1 = pred_xy - 0.5 * pred_wh
    box_xy2 = pred_xy + 0.5 * pred_wh
    boxes = K.concatenate((box_xy1, box_xy2), axis=-1)

    # Filter boxes
    box_scores = box_conf * box_class_prob
    box_classes = K.argmax(box_scores, axis=-1)     # best score index, shape = (grid_h, grid_w, 5)
    box_class_scores = K.max(box_scores, axis=-1)   # best score, shape = (grid_h, grid_w, 5)
    
    prediction_mask = box_class_scores >= score_threshold
    
    boxes = tf.boolean_mask(boxes, prediction_mask)
    scores = tf.boolean_mask(box_class_scores, prediction_mask)
    classes = tf.boolean_mask(box_classes, prediction_mask)

    # Scale box to image shape
    boxes = boxes * image_h

    # Non Max Supression
    selected_idx = tf.image.non_max_suppression(boxes, scores, 50, iou_threshold=iou_threshold)
    boxes = K.gather(boxes, selected_idx).numpy()
    scores = K.gather(scores, selected_idx).numpy()
    classes = K.gather(classes, selected_idx).numpy()
    
    return boxes, scores, classes


def display_yolo(image_path, model, score_threshold, iou_threshold, anchors):
    '''
    Display predictions from YOLO model.

    Parameters
    ----------
    - image_path : image path.
    - model : YOLO model.
    - score_threshold : threshold used for filtering predicted bounding boxes.
    - iou_threshold : threshold used for non max suppression.
    '''
    image = plt.imread(image_path)
    boxes, scores, classes = get_yolo_result(image_path, model, score_threshold, iou_threshold, anchors)
        
    # Draw image
    plt.figure(figsize=(2,2))
    f, (ax1) = plt.subplots(1,1, figsize=(10, 10))
    ax1.imshow(image)
    count_detected = boxes.shape[0]
    ax1.set_title('Detected objects count : {}'.format(count_detected))
    color = []
    for i in range(5):
        for j in range(5):
            for k in range(5):
                color.append((k*0.2, j*0.2, i*0.2))
    for i in range(count_detected):
        box = boxes[i,...]
        x = box[0]
        y = box[1]
        w = box[2] - box[0]
        h = box[3] - box[1]
        classe = classes[i]
        clr = color[classe]
        rect = patches.Rectangle((x, y), w, h, linewidth = 3, edgecolor=clr,facecolor='none')
        ax1.add_patch(rect)


def get_iou(box1, box2):
    """
    Calculate iou of "box1" and "box2".
    
    Parameters
    ----------
    - box1, box2 : shape = (4), [x, y, w, h]
    """
    x1_max = box1[0] + 0.5*box1[2]
    x1_min = box1[0] - 0.5*box1[2]
    x2_max = box2[0] + 0.5*box2[2]
    x2_min = box2[0] - 0.5*box2[2]
    y1_max = box1[1] + 0.5*box1[3]
    y1_min = box1[1] - 0.5*box1[3]
    y2_max = box2[1] + 0.5*box2[3]
    y2_min = box2[1] - 0.5*box2[3]
    
    inter_w = np.minimum(x1_max, x2_max) - np.maximum(x1_min, x2_min)
    inter_h = np.minimum(y1_max, y2_max) - np.maximum(y1_min, y2_min)
    intersection = max(0, inter_w) * max(0, inter_h)
    union = (box1[2] * box1[3]) + (box2[2] * box2[3]) - intersection
    
    return float(intersection / union)


def get_single_task_AP(predictions, true_boxes, iou_threshold):
    """
    Calaulate AP format "after" PASCAL VOC 2010 
        for "single category" object detection task.
    
    Parameters
    ----------
    - predictions : bounding boxes predicted by our model, 
        shape = (box_count, 5)
        5 : [x, y, w, h, box_class_prob]
    - true_boxes : ground truth boxes, shape = (ground_truth_count, 4)
        4 : [x, y, w, h]
        
    Returns
    -------
    - AP : average precision
    """
    pred_count = len(predictions)
    true_count = len(true_boxes)
    max_iou = np.zeros((pred_count))
    # 找出與 predictions[i] iou 最大的 true_boxes
    for i in range(pred_count):
        for j in range(true_count):
            iou = get_iou(predictions[i, :4], true_boxes[j])
            if iou > max_iou:
                max_iou[i] = iou
    # max_iou 大於 iou_threshold 的才算做 possitive
    for i in range(pred_count):
        if max_iou[i] >= iou_threshold:
            max_iou[i] = 1
        else:
            max_iou[i] = 0
    # 依據 "box_class_prob" 做排序，求出 rank
    rank = np.zeros((pred_count, 2))
    rank[:, 0] = predictions[:, 4]
    rank[:, 1] = max_iou
    rank = sorted(rank, key=lambda t : t[0], reverse=True)
    # 計算各 rank 的 precision 與 recall
    precision = np.zeros((pred_count))
    recall = np.zeros((pred_count))
    gte_iou_count = 0   # number of bbox "greater than or equal to iou threshold"
    for i in range(pred_count):
        if rank[i, 1] == 1:
            gte_iou_count += 1
        precision[i] = gte_iou_count / (i + 1)
        recall[i] = gte_iou_count / true_count
    # 取 recall ≧ 0, 0.14, 0.29, 0.43, 0.57, 0.71, 1，共 7 的範圍的 precision 的最大值做平均。
    select = np.zeros((7))
    r = [0, 0.14, 0.29, 0.43, 0.57, 0.71, 1]
    for i in range(7):
        for j in range(pred_count):
            if recall[j] >= r[i]:
                select[i] = max(precision[j:])
                break
    # 計算 AP
    AP = select.mean()
    
    return AP


def get_COCO_mAP(boxes, scores, classes, true_boxes):
    """
    Calculate mean Average Precision (mAP) in MSCOCO format.
    
    Parameters
    ----------
    - predictions : bounding boxes and class prbabilities predicted by our model, 
        shape = (box_count, 4 + class_count) = (bbox_count, [x, y, w, h, each class' probability])
    - true_boxes : ground truth boxes, shape = (ground_truth_count, 5)
    """
    class_count = 80
    AP = np.zeros((class_count))
    local_AP = np.zeros((10))    # iou_threshold = (0.5, 0.55, ..., 0.95)
    for i in range(class_count): # i = 現在在算第 i 個類別的 AP
        # bbox : 預測結果為 i 的 bbox
        selected_boxes = np.array([b for k, b in enumerate(boxes) if classes[k] == i])
        selected_scores = np.array([s for k, s in enumerate(scores) if classes[k] == i])
        predictions = np.zeros((selected_boxes.shape[0], selected_boxes.shape[1] + 1))
        predictions[:, :4] = selected_boxes
        predictions[:, 4] = selected_scores
        # 要計算 iou_threshold = (0.5, 0.55, ..., 0.95) 時各類別的 AP
        for j in range(10):
            iou_threshold = 0.5 + (j * 0.05)
            local_AP[j] = get_single_task_AP(predictions, true_boxes, iou_threshold)
        AP[i] = local_AP.mean()
    
    mAP = AP.mean()
    
    return mAP

def get_VOC_mAP(boxes, scores, classes, true_boxes):
    """
    Calculate mean Average Precision (mAP) in PASCAL VOC format.
    
    Parameters
    ----------
    - predictions : bounding boxes and class prbabilities predicted by our model, 
        shape = (box_count, 4 + class_count) = (bbox_count, [x, y, w, h, each class' probability])
    - true_boxes : ground truth boxes, shape = (ground_truth_count, 5)
    """
    class_count = 20
    AP = np.zeros((class_count))
    for i in range(class_count):    # i = 現在在算第 i 個類別的 AP
        # bbox : 預測結果為 i 的 bbox
        selected_boxes = np.array([b for k, b in enumerate(boxes) if classes[k] == i])
        selected_scores = np.array([s for k, s in enumerate(scores) if classes[k] == i])
        predictions = np.zeros((selected_boxes.shape[0], selected_boxes.shape[1] + 1))
        predictions[:, :4] = selected_boxes
        predictions[:, 4] = selected_scores
        
        AP = get_single_task_AP(predictions, true_boxes, 0.5)
    
    mAP = AP.mean()
    
    return mAP















