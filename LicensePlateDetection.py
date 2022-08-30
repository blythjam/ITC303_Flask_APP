from importlib.resources import files
from imutils.video import VideoStream
from PIL import Image
from flask import Flask, render_template, redirect, request, Response, url_for
from flask import Flask,render_template, request
from flask_mysqldb import MySQL
from flask_socketio import SocketIO
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
import datetime
import easyocr
#import keras_ocr
import csv
import uuid
import datetime
UPLOAD_FOLDER = './Images/uploads/'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif, jfif'}

app = Flask(__name__)
app.config['SECRET_KEY']="secret!"
socket = SocketIO(app)

# MYSQL Connection settings please put your local password here
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'raFFie'
app.config['MYSQL_DB'] = 'license_plate_db'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

mysql = MySQL(app)

@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/paystation', methods = ['POST', 'GET'])
def paystation():
    table_load = "false"
    if request.method == "POST":
        table_load = "true"
        form_value = request.form['Rego']
        cursor =  mysql.connection.cursor()
        cursor.execute("SELECT * FROM license_plates WHERE rego_number=%s;", [form_value] )
        data = cursor.fetchone()
        print(form_value)
        current_time = datetime.datetime.now() 
        if not data["time_exited"]:
            d = (current_time - data["time_entered"]).total_seconds() / 3600
            total = d * 5
            total = round(total, 2)
            amount_owed = total
            print(d) 
        else:
            d = (data["time_exited"] - data["time_entered"]).total_seconds() / 3600
            total = d * 5
            total = round(total, 2)
            amount_owed = total
            print(d)      
        return render_template('paystation.html', form_value=form_value, data=data, table_load=table_load, amount_owed=amount_owed)
    else:  
        table_load = "false"  
        data = []  
        return render_template('paystation.html', table_load=table_load, data=data)

@app.route('/paid/<params>')
def paid(params):
    print(params)
    current_time = datetime.datetime.now()
    cursor = mysql.connection.cursor()
    cursor.execute(''' UPDATE license_plates SET has_paid=1, time_exited=%s WHERE rego_number=%s ''', [current_time, params])
    mysql.connection.commit()
    return render_template('paid.html')
 
@app.route('/login', methods = ['POST', 'GET'])
def login():
    if request.method == 'GET':
        return "Login via the login Form"
     
    if request.method == 'POST':
        rego = request.form['rego']      
        timeEntered = datetime.datetime.now()            

        cursor = mysql.connection.cursor()
        cursor.execute(''' INSERT INTO license_plates (rego_number, time_entered) VALUES(%s, %s)''',[rego, timeEntered])
        mysql.connection.commit()
        cursor.close()
        return f"Done!!"

@app.route('/all_cars_in_park', methods=['GET'])
def all_cars_in_park():
    current_time = datetime.datetime.now() 
    cursor =  mysql.connection.cursor()
    cursor.execute("SELECT rego_number AS rego, time_entered as timeEntered, has_paid AS has_paid from license_plates;")
    data = cursor.fetchall()    
    print(data)  
    i = 0;  
    amount_owed = []
    for d in data:
        d = (current_time - data[i]["timeEntered"]).total_seconds() / 3600
        total = d * 5
        total = round(total, 2)
        amount_owed.append(total)
        print(d)
        i += 1

    return render_template('allcarsinpark.html', data=data, current_time=current_time, amount_owed=amount_owed)
    
 


app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Keras OCR / not currently in use ignore the below code
# keras-ocr will automatically download pretrained
# weights for the detector and recognizer.
#pipeline = keras_ocr.pipeline.Pipeline()
# Keras OCR / not currently in use ignore the above code

labels = [{'name':'licence', 'id':1}]
print('here is the cwd ' + os.getcwd())
configs = config_util.get_configs_from_pipeline_file('./Tensorfow/workspace/models/my_ssd_mobnet/pipeline.config')
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join('./Tensorfow\workspace/models/my_ssd_mobnet', 'ckpt-11')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

category_index = label_map_util.create_category_index_from_labelmap('./Tensorfow/workspace/models/annotations/label_map.pbtxt')

IMAGE_PATH = './Images/test/Loop Images/7.jpg'

IMAGE_NAME1 = '7.jpg'

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
plt.savefig('./Images/saved/' + IMAGE_NAME1 + '_detected_plate_full_image.png')

detection_threshold = 0.6

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

def ocr_it(image, detections, detection_threshold, region_threshold, image_name):
    # Scores, boxes and classes above threshold
    scores = list(filter(lambda x: x> detection_threshold, detections['detection_scores']))
    boxes = detections['detection_boxes'][:len(scores)]
    classes = detections['detection_classes'][:len(scores)]
    
    width = image.shape[1]
    height = image.shape[0]   
    text = []
    region = []
    # Apply ROI filtering and OCR
    for idx, box in enumerate(boxes):
        roi = box*[height, width, height, width]
        region = image[int(roi[0]):int(roi[2]),int(roi[1]):int(roi[3])]
        reader = easyocr.Reader(['en'])
        ocr_result = reader.readtext(region)
        print("ocr result: ", ocr_result)
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
        #plt.savefig('./static/' + image_name + '_detected_plate_cropped_image.png')
    
        return text, region


text, region = ocr_it(image_np_with_detections, detections, detection_threshold, region_threshold, IMAGE_NAME1)
print("here ", text)


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

Stop_Go = False

Entry_stop = "StopWait_placeholder.png"
Entry_go = "go_placeholder.png"

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
            text, region = ocr_it(image_np_with_detections, detections, detection_threshold, region_threshold, IMAGE_NAME1)            
            save_results(text, region, './Detection_images/detection_results.csv', './Detection_images/')
            print(text)
            if (text[0] =='879 GQL'):
                socket.send(Entry_go)              
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

@socket.on('message')
def handlemsg(msg):
    pass
       
@app.route('/success', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    print(uploaded_file)
    img_cropped_url = uploaded_file.filename + '_detected_plate_cropped_image.png'
    img_full_url = uploaded_file.filename + '_detected_plate_full_image.png'
    if uploaded_file.filename != '':
        uploaded_file.save(os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename))   
    img_url = UPLOAD_FOLDER + uploaded_file.filename
    print(img_url)
    img = cv2.imread(img_url)    	
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_np = np.array(img)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    print("pass")
    detections = detect_fn(input_tensor)
    print("fail")
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
    plt.savefig('./static/' + uploaded_file.filename + '_detected_plate_full_image.png')

    detection_threshold = 0.7

    image = image_np_with_detections
    scores = list(filter(lambda x: x> detection_threshold, detections['detection_scores']))
    boxes = detections['detection_boxes'][:len(scores)]
    classes = detections['detection_classes'][:len(scores)]

    width = image.shape[1]
    height = image.shape[0]

    region_threshold = 0.4

    text, region = ocr_it(image_np_with_detections, detections, detection_threshold, region_threshold, uploaded_file.filename)
    if uploaded_file.filename != '':
        uploaded_file.save(os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename))
    return render_template('Image_plate_detection.html', img_cropped_url=img_cropped_url, img_full_url=img_full_url, text=text)





if __name__ == '__main__':
    socket.run(app)


