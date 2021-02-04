import os
# comment out below line to enable tensorflow outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from core.functions import *
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
#flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    #'path to weights file')
#flags.DEFINE_integer('size', 416, 'resize images to')
#flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
# flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
#flags.DEFINE_string('video', './data/video/cam3_crop.mp4', 'path to input video')
#flags.DEFINE_string('output', None, 'path to output video')
#flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
#flags.DEFINE_float('iou', 0.45, 'iou threshold')
#flags.DEFINE_float('score', 0.75, 'score threshold')
#flags.DEFINE_boolean('count', False, 'count objects within video')
#flags.DEFINE_boolean('dont_show', False, 'dont show video output')
#flags.DEFINE_boolean('info', False, 'print info on detections')
#flags.DEFINE_boolean('crop', False, 'crop detections from images')
#flags.DEFINE_boolean('plate', False, 'perform license plate recognition')


def main():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    #STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = 416
    video_path = ''
    # get video name by using split method
    # video_name = video_path.split('/')[-1]
    # video_name = video_name.split('.')[0]
    video_name = 'detected'
    # if FLAGS.framework == 'tflite':
    #     interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
    #     interpreter.allocate_tensors()
    #     input_details = interpreter.get_input_details()
    #     output_details = interpreter.get_output_details()
    #     print(input_details)
    #     print(output_details)
    # else:
    saved_model_loaded = tf.saved_model.load('./checkpoints/yolov4-416', tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    frame_num = 0
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_num += 1
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break

        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        batch_data = tf.constant(image_data)
        pred_bbox = infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=0.45,
                score_threshold=0.75
                )

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)

        pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)
        allowed_classes = ['person']

        # crop each detection and save it as new image
        # capture images every so many frames (ex. crop photos every 150 frames)
        crop_rate = 3
        crop_path = os.path.join(os.getcwd(), video_name)
        try:
            os.mkdir(crop_path)
        except FileExistsError:
            pass
        if frame_num % crop_rate == 0:
            final_path = os.path.join(crop_path, str(frame_num))
            try:
                os.mkdir(final_path)
            except FileExistsError:
                pass
            #crop_objects(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), pred_bbox, final_path, allowed_classes)
            #os.rmdir(final_path)
        else:
            pass

        image = utils.draw_bbox(frame, pred_bbox, allowed_classes=allowed_classes)

        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        result = np.asarray(image)
        cv2.namedWindow("Tracking on cam", cv2.WINDOW_AUTOSIZE)
        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        cv2.imshow("Tracking on cam", result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()

#main()
# if __name__ == '__main__':
#     app.run(main)



