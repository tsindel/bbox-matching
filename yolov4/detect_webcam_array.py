import sys
import numpy as np
import tensorflow as tf
import core.utils as utils
import cv2
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

tf.compat.v1.app.flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
tf.compat.v1.app.flags.DEFINE_string('weights', './checkpoints/yolov4-416', 'path to weights file')
tf.compat.v1.app.flags.DEFINE_integer('size', 416, 'resize images to')
tf.compat.v1.app.flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
tf.compat.v1.app.flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
tf.compat.v1.app.flags.DEFINE_string('video', '0', 'path to input video or set to 0 for webcam')
tf.compat.v1.app.flags.DEFINE_string('output', './detections/webcam_live.avi', 'path to output video')
tf.compat.v1.app.flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
tf.compat.v1.app.flags.DEFINE_float('iou', 0.45, 'iou threshold')
tf.compat.v1.app.flags.DEFINE_float('score', 0.25, 'score threshold')
tf.compat.v1.app.flags.DEFINE_boolean('dont_show', False, 'dont show video output')

if not hasattr(sys, 'argv'):
    sys.argv = ['./detect_webcam_array.py']

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
FLAGS = tf.compat.v1.app.flags.FLAGS
STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
input_size = FLAGS.size
video_path = FLAGS.video

if FLAGS.framework == 'tflite':
    interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(input_details)
    print(output_details)
else:
    saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
    inference = saved_model_loaded.signatures['serving_default']

# begin video capture
try:
    vid = cv2.VideoCapture(int(video_path))
except:
    vid = cv2.VideoCapture(video_path)

if FLAGS.output:
    # by default VideoCapture returns float instead of int
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
    out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))


def infer(wire, frame):
    frame = np.array(frame, dtype='uint8')

    image_data = cv2.resize(frame, (input_size, input_size)) / 255.
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    if FLAGS.framework == 'tflite':
        interpreter.set_tensor(input_details[0]['index'], image_data)
        interpreter.invoke()
        pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
        if FLAGS.model == 'yolov3' and FLAGS.tiny:
            boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                            input_shape=tf.constant([input_size, input_size]))
        else:
            boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                            input_shape=tf.constant([input_size, input_size]))

    else:
        batch_data = tf.constant(image_data)
        pred_bbox = inference(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=FLAGS.iou,
        score_threshold=FLAGS.score
    )

    session.close()
    cv2.destroyAllWindows()

    valid_detections = np.array(valid_detections).reshape(-1)  # element 0
    classes = np.array(classes).reshape(-1)  # element 1-50
    scores = np.array(scores).reshape(-1)  # element 51-100
    boxes = np.array(boxes).reshape(-1)  # element 101-300 (each 4 consecutive elements describe one bbox)

    result = np.concatenate((valid_detections, classes, scores, boxes))

    return result


if __name__ == '__main__':
    while True:
        return_value, frame = vid.read()

        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            print('Video has ended or failed, try a different video format!')
        
        results = infer([0], frame)
        print(results)
