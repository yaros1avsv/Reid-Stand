from flask import Flask, render_template, redirect, url_for

from detect_video import *
from folders_move import *
from reid import make_reid

app = Flask(__name__)

with open('camera_data.json', 'r') as f:
    camera_data = json.load(f)


@app.route('/')
def index():
    return render_template('map.html', markers=camera_data)


@app.route('/camera/<int:cluster>/<int:id>/')
def camera_detail(cluster, id):
    img_catalog = make_camera_img_catalog(cluster, id)
    address = get_address_from_id(str(cluster) + str(id))
    return render_template('camera_detail.html', cluster=cluster, id=id, address=address, image_catalog=img_catalog)


@app.route('/camera/<int:cluster>/<int:id>/detect')
def detect(cluster, id):
    img_catalog = make_camera_img_catalog(cluster, id)
    img_catalog = add_static_to_catalog(img_catalog)
    camera_cluster_id = str(cluster) + str(id)

    videoname = 'cam_' + str(cluster) + '_' + str(id) + '.mp4'
    video_path = os.path.join(os.getcwd(), 'static/img/cameras/cluster_' + str(cluster) + '/camera_' + str(id), videoname)

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    input_size = 416

    saved_model_loaded = tf.saved_model.load('./checkpoints/yolov4-416', tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

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
        crop_rate = 30
        crop_path = os.path.join(os.getcwd(), 'static/img/cameras/cluster_' + str(cluster) + '/camera_' + str(id) + '/detected')
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
            crop_objects(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), pred_bbox, final_path, allowed_classes, camera_cluster_id)
            os.rmdir(final_path)
        else:
            pass

        image = utils.draw_bbox(frame, pred_bbox, allowed_classes=allowed_classes)

        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        result = np.asarray(image)
        cv2.namedWindow('Tracking on camera_' + str(cluster) + '_' + str(id), cv2.WINDOW_AUTOSIZE)
        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        cv2.imshow('Tracking on camera_' + str(cluster) + '_' + str(id), result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()

    bboxes = make_camera_detected_img_catalog(cluster, id)
    return render_template('detected_images.html', image_catalog=bboxes, cluster=cluster, id=id)


@app.route('/camera/<int:cluster>/<int:id>/detect/toquery')
def detected_to_query(cluster, id):
    move_bboxes_to(cluster, id, 'query')
    return redirect(url_for('query'))


@app.route('/camera/<int:cluster>/<int:id>/detect/togalery')
def detected_to_galery(cluster, id):
    move_bboxes_to(cluster, id, 'galery')
    return redirect(url_for('galery'))


@app.route('/query')
def query():
    query_catalog = prepare_query()
    used_cameras = find_unique_camera_id('static/img/query')
    return render_template('images.html',
                           image_catalog=query_catalog,
                           used_cameras=used_cameras,
                           address=get_address_from_id,
                           query=True)


@app.route('/query/clean')
def clean_query():
    clean_directory('static/img/query')
    return redirect(url_for('query'))


@app.route('/galery')
def galery():
    galery_catalog = prepare_galery()
    used_cameras = find_unique_camera_id('static/img/galery')
    return render_template('images.html',
                           image_catalog=galery_catalog,
                           used_cameras=used_cameras,
                           address=get_address_from_id,
                           galery=True)


@app.route('/galery/clean')
def clean_galery():
    clean_directory('static/img/galery')
    return redirect(url_for('galery'))


@app.route('/reid')
def reid():
    clean_directory('static/img/result')
    camera_count = len(find_unique_camera_id('static/img/galery'))
    make_reid(camera_count)
    result_catalog = get_result_catalog()
    return render_template('result_imgs.html',
                           image_catalog=result_catalog)


if __name__ == '__main__':
    app.run(debug=True)
