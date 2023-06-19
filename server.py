import os
import cv2
import time
import redis
import logging
import collections
import numpy as np
from gevent import pywsgi
from datetime import datetime
from openvino.runtime import Core
from flask import Flask, render_template, Response

from utils import VideoPlayer, preprocess, batch_preprocess, process_results, draw_boxes
from deepsort_utils.tracker import Tracker
from deepsort_utils.nn_matching import NearestNeighborDistanceMetric
from deepsort_utils.detection import Detection, compute_color_for_labels, xywh_to_xyxy, xywh_to_tlwh, tlwh_to_xyxy

logging.basicConfig(level=logging.INFO)

source = Flask(__name__)

# RTSP_IP = os.environ["RTSP_IP"] if "RTSP_IP" in os.environ else '192.168.82.117'
# fame_rtsp_url = f"rtsp://admin:pz86824141@{RTSP_IP}:554/cam/realmonitor?channel=1&subtype=0"

RTSP_IP = os.environ["RTSP_IP"] if "RTSP_IP" in os.environ else 'xx'
fame_rtsp_url = f"rtsp://admin:xx@{RTSP_IP}:554/Streaming/Channels/201"

IPADDRESS = os.environ["IPADDRESS"] if "IPADDRESS" in os.environ else '0.0.0.0'
PORT = os.environ["PORT"] if "PORT" in os.environ else 5005
PORT = int(PORT)

const_redis_host = os.environ["REDIS_HOST"] if "REDIS_HOST" in os.environ else "xx"  
const_redis_username = os.environ["REDIS_USERNAME"] if "REDIS_USERNAME" in os.environ else "admin"
const_redis_password = os.environ["REDIS_PASSWORD"] if "REDIS_PASSWORD" in os.environ else "xx"   
pool = redis.ConnectionPool(host=const_redis_host,password=const_redis_password, port=6379, decode_responses=True)  
client = redis.Redis(connection_pool=pool)

people_flow = set()

ie_core = Core()
logging.info(ie_core.available_devices)
class Model:
    """
    This class represents a OpenVINO model object.

    """
    def __init__(self, model_path, batchsize=1, device="AUTO"):
        """
        Initialize the model object
        
        Parameters
        ----------
        model_path: path of inference model
        batchsize: batch size of input data
        device: device used to run inference
        """
        self.model = ie_core.read_model(model=model_path)
        self.input_layer = self.model.input(0)
        self.input_shape = self.input_layer.shape
        self.height = self.input_shape[2]
        self.width = self.input_shape[3]

        for layer in self.model.inputs:
            input_shape = layer.partial_shape
            input_shape[0] = batchsize
            self.model.reshape({layer: input_shape})
        self.compiled_model = ie_core.compile_model(model=self.model, device_name=device)
        self.output_layer = self.compiled_model.output(0)

    def predict(self, input):
        """
        Run inference
        
        Parameters
        ----------
        input: array of input data
        """
        result = self.compiled_model(input)[self.output_layer]
        return result

detector = Model('weights/person-detection-0202.xml')
extractor = Model('weights/person-reidentification-retail-0287.xml', -1)

NN_BUDGET = 100
MAX_COSINE_DISTANCE = 0.6  # threshold of matching object
metric = NearestNeighborDistanceMetric(
    "cosine", MAX_COSINE_DISTANCE, NN_BUDGET
)
tracker = Tracker(
    metric,
    max_iou_distance=0.7,
    max_age=70,
    n_init=3
)

def isEveryHour():
    nows = int(time.time())
    # timestamp = 1591239600
    dt = datetime.fromtimestamp(nows)
    # print(dt.hour, dt.minute,dt.second)
    if dt.minute==0 and dt.second==0 and dt.microsecond==0:
        return True
    return False

aux = False 
def isSaveSet():
    global aux
    # print(aux)
    flag = isEveryHour()
    if flag == True and aux == False:
        date_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        people_count = len(people_flow)
        client.hset(RTSP_IP, date_time, people_count)
        people_flow.clear()
        aux = True
        logging.info(f'The people flow of {date_time} is {people_count}')
    else:
        if flag == False:
            aux = False

# Main processing function to run person tracking.
def run_person_tracking(rtsp_url=0, flip=False, skip_first_frames=0):
    """
    Main function to run the person tracking:
    1. Create a video player to play with target fps (utils.VideoPlayer).
    2. Prepare a set of frames for person tracking.
    3. Run AI inference for person tracking.
    4. Visualize the results.

    Parameters:
    ----------
        source: The webcam number to feed the video stream with primary webcam set to "0", or the video path.  
        flip: To be used by VideoPlayer function for flipping capture image.
        use_popup: False for showing encoded frames over this notebook, True for creating a popup window.
        skip_first_frames: Number of frames to skip at the beginning of the video. 
    """
    logging.info(f'{IPADDRESS}:{PORT}')
    logging.info(rtsp_url)

    player = None
    try:
        # Create a video player to play with target fps.
        player = VideoPlayer(
            source=rtsp_url, flip=flip, fps=30, skip_first_frames=skip_first_frames
        )
        # Start capturing.
        player.start()

        processing_times = collections.deque()
        while True:
            # Grab the frame.
            frame = player.next()
            if frame is None:
                logging.info("Failed to read the video stream.")
                break
            # If the frame is larger than full HD, reduce size to improve the performance.

            # Resize the image and change dims to fit neural network input.
            h, w = frame.shape[:2]
            input_image = preprocess(frame, detector.height, detector.width)

            # Measure processing time.
            start_time = time.time()
            # Get the results.
            output = detector.predict(input_image)
            stop_time = time.time()
            processing_times.append(stop_time - start_time)
            if len(processing_times) > 200:
                processing_times.popleft()

            _, f_width = frame.shape[:2]
            # Mean processing time [ms].
            processing_time = np.mean(processing_times) * 1000
            fps = 1000 / processing_time

            # Get poses from detection results.
            bbox_xywh, score, label = process_results(h, w, results=output)
            
            img_crops = []
            for box in bbox_xywh:
                x1, y1, x2, y2 = xywh_to_xyxy(box, h, w)
                img = frame[y1:y2, x1:x2]
                img_crops.append(img)

            # Get reidentification feature of each person.
            if img_crops:
                # preprocess
                img_batch = batch_preprocess(img_crops, extractor.height, extractor.width)
                features = extractor.predict(img_batch)
            else:
                features = np.array([])

            # Wrap the detection and reidentification results together
            bbox_tlwh = xywh_to_tlwh(bbox_xywh)
            detections = [
                Detection(bbox_tlwh[i], features[i])
                for i in range(features.shape[0])
            ]

            # predict the position of tracking target 
            tracker.predict()

            # update tracker
            tracker.update(detections)

            # update bbox identities
            outputs = []
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                box = track.to_tlwh()
                x1, y1, x2, y2 = tlwh_to_xyxy(box, h, w)
                track_id = track.track_id
                outputs.append(np.array([x1, y1, x2, y2, track_id], dtype=np.int))
            if len(outputs) > 0:
                outputs = np.stack(outputs, axis=0)
                for people in outputs:
                    peopleID = people[-1]
                    people_flow.add(peopleID)

            isSaveSet()

            # print(people_flow)

            # draw box for visualization
            if len(outputs) > 0:
                bbox_tlwh = []
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
                frame = draw_boxes(frame, bbox_xyxy, identities)


            output_image_frame = cv2.putText(
                img=frame,
                text=f"Inference time: {processing_time:.1f}ms ({fps:.1f} FPS)",
                org=(20, 40),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=f_width / 1000,
                color=(0, 0, 255),
                thickness=1,
                lineType=cv2.LINE_AA,
            )
            
            ret, buffer = cv2.imencode('.jpg', output_image_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


    except Exception as e:
        logging.error(e)
 


@source.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(run_person_tracking(rtsp_url=fame_rtsp_url), mimetype='multipart/x-mixed-replace; boundary=frame')

@source.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

if __name__ == '__main__':
    server = pywsgi.WSGIServer((IPADDRESS,PORT),source)
    server.serve_forever()

