import os
import cv2
import numpy as np
import tensorflow as tf
from core.utils import read_class_names
from core.config import cfg

# function for cropping each detection and saving as new image
def crop_objects(img, data, path, allowed_classes, camera_cluster_id):
    boxes, scores, classes, num_objects = data
    class_names = read_class_names(cfg.YOLO.CLASSES)
    #create dictionary to hold count of objects for image name
    count = 0
    counts = dict()
    for i in range(num_objects):
        # get count of class for part of image name
        class_index = int(classes[i])
        class_name = class_names[class_index]
        if class_name in allowed_classes:
            counts[class_name] = counts.get(class_name, 0) + 1
            # get box coords
            xmin, ymin, xmax, ymax = boxes[i]
            cropped_img = img[int(ymin)-5:int(ymax)+5, int(xmin)-5:int(xmax)+5]
            img_r = cv2.resize(cropped_img, (128, 256), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(path) + str(counts[class_name]) + '_c' + str(camera_cluster_id) + '_image_' + '.jpg', img_r)
        else:
            continue
