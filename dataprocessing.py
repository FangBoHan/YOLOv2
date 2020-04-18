import xml.etree.ElementTree as ET
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras.backend as K
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def parse_annotation(ann_dir, img_dir, labels):
    """
    Capture bboxes and image names in "ann_dir" and "img_dir" respectively.
    
    Parameters
    ----------
    - ann_dir : annotations files directory, annotation files : .xml
    - img_dir : images files directory
    - labels : labels list
    
    Returns
    -------
    - imgs_name : numpy array of images files path (shape : images count, 1)
    - true_boxes : numpy array of annotations for each image (shape : image count, max annotation count, 5)
        annotation format : xmin, ymin, xmax, ymax, class
        xmin, ymin, xmax, ymax : image unit (pixel)
        class = label index
    """
    
    max_annot = 0       # 一張圖片最多的 object 數量
    imgs_name = []
    annots = []         # object 的 bounding box
    
    for ann in sorted(os.listdir(ann_dir)):
        annot_count = 0
        boxes = []
        tree = ET.parse(ann_dir + ann)    # import xml.etree.ElementTree as ET
        for elem in tree.iter(): 
            if "filename" in elem.tag:
                imgs_name.append(img_dir + elem.text)
            if ("object" in elem.tag) or ("part" in elem.tag):
                box = np.zeros((5))       # 5 : (xmin, ymin, xmax, ymax, class)
                for attr in list(elem):
                    if "name" in attr.tag:
                        box[4] = labels.index(attr.text) + 1 # 0 : label for no bounding box
                    if "bndbox" in attr.tag:
                        annot_count += 1
                        for dim in list(attr):
                            if "xmin" in dim.tag:
                                box[0] = int(round(float(dim.text)))    # round(number, ndigits=None)
                            if "ymin" in dim.tag:                       # 對 number 做四捨五入到小數點後
                                box[1] = int(round(float(dim.text)))    # 第 ndigits 位
                            if "xmax" in dim.tag:                       # ndigits=None 表示到小數點後
                                box[2] = int(round(float(dim.text)))    # 第 0 位 = 個位數
                            if "ymax" in dim.tag:
                                box[3] = int(round(float(dim.text)))
                boxes.append(np.asarray(box))   # np.asarray : 將輸入轉換成 array 的形式
        annots.append(np.asarray(boxes))        # boxes : 一張圖中的 bbox, annots : 所有圖片的 bbox
        
        if annot_count > max_annot:
            max_annot = annot_count
           
    # Rectify annotations boxes : len -> max_annot
    imgs_name = np.array(imgs_name)  
    true_boxes = np.zeros((imgs_name.shape[0], max_annot, 5))
    for idx, boxes in enumerate(annots):
        true_boxes[idx, :boxes.shape[0], :5] = boxes
        
    return imgs_name, true_boxes


def parse_function(imgs_name, true_boxes):
    """
    Decode images in "imgs_name".
    """
    x_img_string = tf.io.read_file(imgs_name)
    x_img = tf.image.decode_png(x_img_string, channels=3)   # dtype=tf.uint8
    x_img = tf.image.convert_image_dtype(x_img, tf.float32) # pixel value /255, dtype=tf.float32, channels : RGB
    return x_img, true_boxes


def get_dataset(img_dir, ann_dir, labels, batch_size):
    """
    Create a dataset.
    Note : The dataset isn't YOLO prediction format.
    
    Parameters
    ----------
    - ann_dir : annotations files directory
    - img_dir : images files directory
    - labels : labels list
    - batch_size : int
    
    Returns
    -------
    - dataset :
        dataset : (batch_count, 0, batch_size, img_w, img_h, 3)
                : (batch_count, 1, batch_size, max_annot, 5)
        batch : tupple(images, annotations)
        batch[0] : images : tensor (shape : batch_size, img_w, img_h, 3)
        batch[1] : annotations : tensor (shape : batch_size, max annot, 5)
                   annotation format : xmin, ymin, xmax, ymax, class
        
    Note : image pixel values = pixels value / 255. channels : RGB
    """
    imgs_name, bbox = parse_annotation(ann_dir, img_dir, labels)
    
    # from_tensor_slices(tensors) :
    # Creates a dataset whose elements are slices of the given tensors.
    # The given tensors are sliced along their first dimension.
    dataset = tf.data.Dataset.from_tensor_slices((imgs_name, bbox))
    
    # (buffer_size, seed=None, reshuffle_each_iteration=None) :
    # Randomly shuffles the elements of this dataset.
    dataset = dataset.shuffle(len(imgs_name))
    
    # repeat(count=None) :
    # Repeats this dataset so each original value is seen "count" times.
    # >>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3]) 
    # >>> dataset = dataset.repeat(3) 
    # >>> list(dataset.as_numpy_iterator()) 
    # [1, 2, 3, 1, 2, 3, 1, 2, 3]
    dataset = dataset.repeat()
    
    # map(map_func, num_parallel_calls=None) :
    # Applies "map_func" to each element of this dataset, 
    # and returns a new dataset containing the transformed elements, 
    # in the same order as they appeared in the input.
    dataset = dataset.map(parse_function, num_parallel_calls=6)  # 利用 parse_function 將 img 轉成 tf 看的懂的形式？
    
    # batch(batch_size, drop_remainder=False) :
    # Combines consecutive (連續的) elements of this dataset into batches.
    # [0, 1, 2, 3, 4, 5, 6, 7] -> dataset = dataset.batch(3) -> [0, 1, 2] [3, 4, 5] [6, 7]
    # 也就是將資料集每 batch_size 個分成一組
    dataset = dataset.batch(batch_size)
    
    # prefetch(buffer_size) :
    # Creates a Dataset that prefetches elements from this dataset.
    # Most dataset input pipelines should end with a call to prefetch. 
    # This allows later elements to be prepared while the current element is being processed. 
    # This often improves latency and throughput, 
    # at the cost of using additional memory to store prefetched elements.
    dataset = dataset.prefetch(10)
    
    return dataset


def process_true_boxes(true_boxes, anchors, image_w, image_h):
    """
    將 "true_boxes" 與 "anchors" 轉成 YOLO 的形式
    (x1, y1, x2, y2) -> (x, y, w, h)
    單位：image pixel -> grid cell
    
    Parameters
    ----------
    - true_boxes : tensor, shape (max_annot, 5), format : x1, y1, x2, y2, class, unit : image pixel
    - anchors : list, [anchor_1_width, anchor_1_height, anchor_2_width, anchor_2_height, ...]
        unit : grid cell
    - image_w, image_h : unit : pixel
    
    Returns
    -------
    - detector_mask : array, shape (grid_w, grid_h, anchors_count, 1)
        用於表示哪個 grid cell 是使用哪個 anchor 偵測，
        若 [x, y, a] = 1 表示第 (x, y) 個 grid 的使用第 a 個 anchor
        若 [x, y, 0:anchors_count] = 0 表示該 grid 上沒有物件
    - matching_true_boxes : array, shape (grid_w, grid_h, anchors_count, 5)
        Contains adjusted coords of bounding box in YOLO format
    - true_boxes_grid : array, same shape than true_boxes (max_annot, 5),
        format : x, y, w, h, class ; unit : grid cell
    
    Note:
    -----
    Bounding box in YOLO Format : x, y, w, h, class
    x, y : center of bounding box, unit : grid cell
    w, h : width and height of bounding box, unit : grid cell
    class : label index
    """
    anchor_count = len(anchors) // 2    # 見 parameters 中 anchors 的描述
    anchors = np.array(anchors)
    anchors = anchors.reshape(anchor_count, 2)
    
    
    scale = 32    # 由於 YOLO 有 5 個 pooling layer，所以要除以 32
    grid_w, grid_h = image_w // scale, image_h // scale
    detector_mask = np.zeros((grid_w, grid_h, anchor_count, 1))
    matching_true_boxes = np.zeros((grid_w, grid_h, anchor_count, 5))
    
    true_boxes = true_boxes.numpy()
    true_boxes_grid = np.zeros(true_boxes.shape)
    
    for i, box in enumerate(true_boxes):
        # (x1, y1, x2, y2) -> (x, y, w, h)
        # image pixel -> grid cell
        x = ((box[0] + box[2]) / 2) / scale
        y = ((box[1] + box[3]) / 2) / scale
        w = (box[2] - box[0]) / scale
        h = (box[3] - box[1]) / scale
        true_boxes_grid[i, ...] = np.array([x, y, w, h, box[4]])
        
        if w * h > 0:   # if box exists
            # find the best anchor to detect object
            best_iou = 0
            best_anchor = 0
            for i in range(anchor_count):
                # calculate iou (anchor and box are shifted to 0,0)
                intersection = np.minimum(w, anchors[i, 0]) * np.minimum(h, anchors[i, 1])
                union = (w * h) + (anchors[i, 0] * anchors[i, 1]) - intersection
                iou = intersection / union
                if iou > best_iou:
                    best_iou = iou
                    best_anchor = i
            if best_iou > 0:
                x_coord = np.floor(x).astype("int")
                y_coord = np.floor(y).astype("int")
                detector_mask[y_coord, x_coord, best_anchor] = 1
                yolo_box = [x, y, w, h, box[4]]
                matching_true_boxes[y_coord, x_coord, best_anchor] = yolo_box
    return matching_true_boxes, detector_mask, true_boxes_grid


def ground_truth_generator(dataset, class_count, anchors):
    '''
    Ground truth batch generator from a yolo dataset, ready to compare with YOLO prediction in loss function.

    Parameters
    ----------
    - YOLO dataset. Generate batch:
        batch : tupple(images, annotations)
        batch[0] : images : tensor (shape : batch_size, IMAGE_W, IMAGE_H, 3)
        batch[1] : annotations : tensor (shape : batch_size, max annot, 5)
        
    Returns
    -------
    - imgs : images to predict. tensor (shape : batch_size, IMAGE_H, IMAGE_W, 3)
    - detector_mask : tensor, shape (batch, size, GRID_W, GRID_H, anchors_count, 1)
        1 if bounding box detected by grid cell, else 0
    - matching_true_boxes : tensor, shape (batch_size, GRID_W, GRID_H, anchors_count, 5)
        Contains adjusted coords of bounding box in YOLO format
    - class_one_hot : tensor, shape (batch_size, GRID_W, GRID_H, anchors_count, class_count)
        One hot representation of bounding box label
    - true_boxes_grid : annotations : tensor (shape : batch_size, max annot, 5)
        true_boxes format : x, y, w, h, c, coords unit : grid cell
    '''
    for batch in dataset:
        # imgs
        imgs = batch[0]
        image_w = imgs.shape[1]
        image_h = imgs.shape[2]
        
        # true boxes
        true_boxes = batch[1]
        
        # matching_true_boxes and detector_mask
        batch_matching_true_boxes = []
        batch_detector_mask = []
        batch_true_boxes_grid = []
        
        for i in range(true_boxes.shape[0]):     
            one_matching_true_boxes, one_detector_mask, true_boxes_grid = process_true_boxes(true_boxes[i],
                                                                                           anchors,
                                                                                           image_w,
                                                                                           image_h)
            batch_matching_true_boxes.append(one_matching_true_boxes)
            batch_detector_mask.append(one_detector_mask)
            batch_true_boxes_grid.append(true_boxes_grid)
                
        detector_mask = tf.convert_to_tensor(np.array(batch_detector_mask), dtype="float32")
        matching_true_boxes = tf.convert_to_tensor(np.array(batch_matching_true_boxes), dtype="float32")
        true_boxes_grid = tf.convert_to_tensor(np.array(batch_true_boxes_grid), dtype="float32")
        
        # class one_hot
        matching_classes = K.cast(matching_true_boxes[..., 4], "int32")    # keras.backend.cast(x, dtype)
                                                                           # 功能：將張量轉換到不同的dtype並返回。
                                                                           # 你可以轉換一個Keras 變量，但它仍然返回一個Keras 張量
        class_one_hot = K.one_hot(matching_classes, class_count + 1)[:,:,:,:,1:]
        class_one_hot = tf.cast(class_one_hot, dtype="float32")
        
        batch = (imgs, detector_mask, matching_true_boxes, class_one_hot, true_boxes_grid)
        yield batch


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
    # load image
    image = cv2.imread(image_path)

    input_image = image[:,:,::-1]
    input_image = image / 255.
    image_w = input_image.shape[1]
    image_h = input_image.shape[0]
    grid_w = image_w // 32
    grid_h = image_h // 32
    
    input_image = np.expand_dims(input_image, 0)

    # prediction
    y_pred = model.predict_on_batch(input_image)

    # post prediction process
    # grid coords tensor
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
    box_classes = K.argmax(box_scores, axis=-1) # best score index
    box_class_scores = K.max(box_scores, axis=-1) # best score
    prediction_mask = box_class_scores >= score_threshold
    boxes = tf.boolean_mask(boxes, prediction_mask)
    scores = tf.boolean_mask(box_class_scores, prediction_mask)
    classes = tf.boolean_mask(box_classes, prediction_mask)

    # Scale box to image shape
    boxes = boxes * image_h

    # Non Max Supression
    selected_idx = tf.image.non_max_suppression(boxes, scores, 50, iou_threshold=iou_threshold)
    boxes = K.gather(boxes, selected_idx)
    scores = K.gather(scores, selected_idx)
    classes = K.gather(classes, selected_idx)
    
    # Draw image
    plt.figure(figsize=(2,2))
    f, (ax1) = plt.subplots(1,1, figsize=(10, 10))
    ax1.imshow(image[:,:,::-1])
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
        classe = classes[i].numpy()
        clr = color[classe]
        rect = patches.Rectangle((x.numpy(), y.numpy()), w.numpy(), h.numpy(), linewidth = 3, edgecolor=clr,facecolor='none')
        ax1.add_patch(rect)
    











