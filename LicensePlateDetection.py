from importlib.resources import files
from imutils.video import VideoStream
from PIL import Image
from flask import Flask, render_template, redirect, request, Response, url_for
import os
import tensorflow.compat.v1 as tf
import numpy as np
import cv2
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing import image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
import time
import easyocr
#import keras_ocr
import csv
import uuid
import datetime


app = Flask(__name__)


# Keras OCR / not currently in use ignore the below code
# keras-ocr will automatically download pretrained
# weights for the detector and recognizer.
#pipeline = keras_ocr.pipeline.Pipeline()
# Keras OCR / not currently in use ignore the above code

labels = [{'name':'licence', 'id':1}]

configs = config_util.get_configs_from_pipeline_file('E:/LicensePlateDetection/flask-ml/Tensorfow/workspace/models/my_ssd_mobnet/pipeline.config')
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join('E:/LicensePlateDetection/flask-ml/Tensorfow\workspace/models/my_ssd_mobnet', 'ckpt-11')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

category_index = label_map_util.create_category_index_from_labelmap('E:/LicensePlateDetection/flask-ml/Tensorfow/workspace/models/annotations/label_map.pbtxt')

IMAGE_PATH = 'E:/LicensePlateDetection/flask-ml/Images/test/Cars416.png'

IMAGE_NAME = 'Cars416.png'

print(IMAGE_PATH)


img = cv2.imread(IMAGE_PATH)
image_np = np.array(img)

input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
detections = detect_fn(input_tensor)

num_detections = int(detections.pop('num_detections'))
detections = {key: value[0, :num_detections].numpy()
              for key, value in detections.items()}
detections['num_detections'] = num_detections

# detection_classes should be ints.
detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

label_id_offset = 1
image_np_with_detections = image_np.copy()

viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes']+label_id_offset,
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=5,
            min_score_thresh=.8,
            agnostic_mode=False)

plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
#plt.show()
plt.savefig('E:/LicensePlateDetection/flask-ml/Images/saved/' + IMAGE_NAME + '_detected_plate_full_image.png')

detection_threshold = 0.7

image = image_np_with_detections
scores = list(filter(lambda x: x> detection_threshold, detections['detection_scores']))
boxes = detections['detection_boxes'][:len(scores)]
classes = detections['detection_classes'][:len(scores)]

width = image.shape[1]
height = image.shape[0]

region_threshold = 0.4

def filter_text(region, ocr_result, region_threshold):
    rectangle_size = region.shape[0]*region.shape[1]
    
    plate = []
    
    for result in ocr_result:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))
        
        if length*height / rectangle_size > region_threshold:
            plate.append(result[1])
        
        
    return plate

# filter_text(region, ocr_result, region_threshold)

def ocr_it(image, detections, detection_threshold, region_threshold):
    # Scores, boxes and classes above threshold
    scores = list(filter(lambda x: x> detection_threshold, detections['detection_scores']))
    boxes = detections['detection_boxes'][:len(scores)]
    classes = detections['detection_classes'][:len(scores)]
    
    width = image.shape[1]
    height = image.shape[0]   
   
    # Apply ROI filtering and OCR
    for idx, box in enumerate(boxes):
        roi = box*[height, width, height, width]
        region = image[int(roi[0]):int(roi[2]),int(roi[1]):int(roi[3])]
        reader = easyocr.Reader(['en'])
        ocr_result = reader.readtext(region)
        # Keras OCR / not currently in use ignore the below code
        #read image from the an image path (a jpg/png file or an image url)
        #img = keras_ocr.tools.read(region)
        # Prediction_groups is a list of (word, box) tuples
        #prediction_groups = pipeline.recognize([img])

        #plate_text = ""
        #new_boxes = []
        #for text, box in prediction_groups[0]:
        #    #print(len(prediction_groups))
        #    plate_text += text
        #    plate_text += " "
        #    print(box)
        #    new_boxes.append(box)            
        #    #print(text)
        #print(plate_text.upper())
        #print(new_boxes)
        #plate_text = plate_text.upper()

        # Keras OCR / not currently in use ignore the above code

        text = filter_text(region, ocr_result, region_threshold)

        plt.imshow(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
        plt.savefig('E:/LicensePlateDetection/flask-ml/Images/saved/' + IMAGE_NAME + '_detected_late_cropped_image.png')
    
        return text, region

text, region = ocr_it(image_np_with_detections, detections, detection_threshold, region_threshold)

print(text)

def save_results(text, region, csv_filename, folder_path):
    img_name = '{}.jpg'.format(uuid.uuid1())
    
    cv2.imwrite(os.path.join(folder_path, img_name), region)

    recordTime = datetime.datetime.now()
    
    with open(csv_filename, mode='a', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow([img_name, text, recordTime])

#save_results(text, region, 'E:/LicensePlateDetection/flask-ml/Detection_images/detection_results.csv', 'E:/LicensePlateDetection/flask-ml/Detection_images/')

cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

def webcamdect():
    while cap.isOpened(): 
        ret, frame = cap.read()
        image_np = np.array(frame)
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor)
        
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
                    image_np_with_detections,
                    detections['detection_boxes'],
                    detections['detection_classes']+label_id_offset,
                    detections['detection_scores'],
                    category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=5,
                    min_score_thresh=.8,
                    agnostic_mode=False)   

        try:
            text, region = ocr_it(image_np_with_detections, detections, detection_threshold, region_threshold)            
            save_results(text, region, 'E:/LicensePlateDetection/flask-ml/Detection_images/detection_results.csv', 'E:/LicensePlateDetection/flask-ml/Detection_images/')
        except:
            pass    

        cv2.imshow('object detection',  cv2.resize(image_np_with_detections, (800, 600)))
        frame = cv2.imencode('.jpg', image_np_with_detections)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.01)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

@app.route("/")
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    '''
    Video streaming route.
    '''
    return Response(
        webcamdect(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/success', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    print(uploaded_file.filename)   
    
    img = cv2.imread(uploaded_file.filename)
    image_np = np.array(img)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=5,
                min_score_thresh=.8,
                agnostic_mode=False)

    plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
    #plt.show()
    plt.savefig('E:/LicensePlateDetection/flask-ml/Images/saved/' + uploaded_file.filename + '_detected_plate_full_image.png')

    detection_threshold = 0.7

    image = image_np_with_detections
    scores = list(filter(lambda x: x> detection_threshold, detections['detection_scores']))
    boxes = detections['detection_boxes'][:len(scores)]
    classes = detections['detection_classes'][:len(scores)]

    width = image.shape[1]
    height = image.shape[0]

    region_threshold = 0.4

    text, region = ocr_it(image_np_with_detections, detections, detection_threshold, region_threshold)
    if uploaded_file.filename != '':
        uploaded_file.save(os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename))
    return render_template('index.html')


if __name__ == '__main__':
    app.run()


