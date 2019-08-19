import os
import time
import cv2
import numpy as np
from imageai.Detection import ObjectDetection
from imageai.Detection import VideoObjectDetection
from numpy import linalg as LA

execution_path = os.getcwd()
detector = ObjectDetection()

'''
Parameters:
-----------
Optimized for 1 frame every 2 seconds (.5 fps) videos typically time lapses  (this is the fastest it can run at real time)

mod: this is for how far around the previous boat detected to look to identify if it is the same boat from the previous frame
frame scope: how many frames to check previous from current frame to determine if it is the same boat as seen before
model: pretty much has to be 'y' (yolo) only model that consistently detect boats on water the best
speed: normal and fast work best changes how it segments the image to make the speed faster 

for frame rates at .5 fps (frame very 2 seconds) (tested)
mod = 2.6
frameScope = 30
model = 'y'
speed="normal"

for frame rate 1 every 30 seconds (experimental)
mod = 5
frameScope = 5
model = 'y'
speed="normal"

'''



mod = 5
frameScope = 5
model = 'y'
speed="normal"

numboats = 0
boatList = []

inVid = os.listdir(execution_path + '\src_vid')
inVid= './src_vid/'+ str(inVid[0])

findCamera = False
#if you want to use a camera, locate the camera address that cv will see it at
#VVVV UNCOMMENT THIS VVVV
#camera = cv2.VideoCapture(0)


#boat entity
class Boat:
    def __init__(self, center, rad, lastFrame):
        self.center = center
        self.rad = rad
        self.lastFrame = lastFrame


def defmodel():
    if model == 'r':
        detector.setModelTypeAsRetinaNet()
        detector.setModelPath(os.path.join(execution_path, "resnet50_coco_best_v2.0.1.h5"))
    elif model == 'y':
        detector.setModelTypeAsYOLOv3()
        detector.setModelPath(os.path.join(execution_path, "yolo.h5"))
    elif model == 'yl':
        detector.setModelTypeAsTinyYOLOv3()
        detector.setModelPath(os.path.join(execution_path, "yolo-tiny.h5"))
    detector.loadModel(detection_speed=str(speed))

#logic to see how close a boat was to it from the previous frame
def intercetC(boat1, boat2, mod):
    x1, y1 = boat1.center
    r1 = boat1.rad * mod
    x2, y2 = boat2.center
    r2 = boat2.rad * mod
    distSq = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
    radSumSq = (r1 + r2) * (r1 + r2);
    if (distSq < radSumSq):
        return 1
    else:
        return 0


#checks to see if it was a boat from before or new boat
def newBoatTest(bp, frame):
    new = True
    x1, y1, x2, y2 = bp
    a = np.array([x1, y1])
    b = np.array([x2, y2])
    r = (a - b) / 2
    c = b + r
    r = LA.norm(r)
    currBoat = Boat(c, r, frame)
    if len(boatList) == 0:
        boatList.append(currBoat)

    for boat in boatList:
        if intercetC(currBoat, boat, mod) and (currBoat.lastFrame - boat.lastFrame) < frameScope:
            boatList.remove(boat)
            boatList.append(currBoat)
            new = False

    if new:
        boatList.append(currBoat)


def forFrame(frame_number, output_array, output_count):
    print("boats detected: %d" % len(boatList))
    for detections in output_array:
        bp = detections['box_points']
        newBoatTest(bp, frame_number)


def findCam():
    camera0 = cv2.VideoCapture(0)
    camera1 = cv2.VideoCapture(1)
    camera2 = cv2.VideoCapture(2)
    camera3 = cv2.VideoCapture(3)
    camera4 = cv2.VideoCapture(4)
    camera5 = cv2.VideoCapture(5)
    camera6 = cv2.VideoCapture(6)
    for i in range(1000):
        retval, img = camera0.read()
        if retval:
            cv2.imshow("0", img)

        retval, img = camera1.read()
        if retval:
            cv2.imshow("1", img)

        retval, img = camera2.read()
        if retval:
            cv2.imshow("2", img)

        retval, img = camera3.read()
        if retval:
            cv2.imshow("3", img)

        retval, img = camera4.read()
        if retval:
            cv2.imshow("4", img)

        retval, img = camera5.read()
        if retval:
            cv2.imshow("5", img)

        retval, img = camera6.read()
        if retval:
            cv2.imshow("6", img)
        cv2.waitKey(1)


if findCamera:
    findCam()

start_time = time.time()
detector = VideoObjectDetection()
defmodel()
custom = detector.CustomObjects(boat=True)
video_path = detector.detectCustomObjectsFromVideo(custom_objects=custom,
                                                   input_file_path=os.path.join(execution_path,
                                                                                inVid),
                                                   output_file_path=os.path.join(execution_path, "video_out")
                                                   , per_frame_function=forFrame, save_detected_video=True,
                                                   minimum_percentage_probability=10, log_progress=True)

print(len(boatList))
print("--- %s seconds ---" % (time.time() - start_time))


