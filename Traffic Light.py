
# import the necessary packages

from project.detection import detect_people
import imutils
import cv2
import os
import time

# base path to YOLO directory
MODEL_PATH = "yolo-coco"

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([MODEL_PATH, "yolov3.weights"])
configPath = os.path.sep.join([MODEL_PATH, "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)


# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream and pointer to output video file
print("[INFO] accessing video stream...")
vs = cv2.VideoCapture(0)
writer = None

# loop over the frames from the video stream
while True:
    # read the next frame from the file
    (grabbed, frame) = vs.read()
    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break
    # resize the frame and then detect people (and only people) in it
    frame = imutils.resize(frame, width=700)
    results = detect_people(frame, net, ln,
        personIdx=LABELS.index("person"))      
    #the lock we make to separate between peroid cars and ppl
    bolt=0
    #to determine if there is ppl    
    if (len(results)>0) and (bolt == 0):
        for (i, (prob, bbox, centroid)) in enumerate(results):
        # extract the bounding box and centroid coordinates, then
        # initialize the color of the annotation
            (startX, startY, endX, endY) = bbox
            if len(results)>0:
                color = (0, 255, 0) 
        # draw (1) a bounding box around the person and (2) the
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)                
        # resize the frame and then detect people (and only people) in it
        frame = imutils.resize(frame, width=700)      
        text = "People can across the street: {}".format(len(results))
        cv2.putText(frame, text, (10, frame.shape[0] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 0), 3)

    # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
            
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

        counter =3
        while counter:
            time.sleep(1)
            counter-=1
            print('PEOPLE crossing')            
        bolt=1   
#     # to make the cars get through                  
    if (len(results)==0):
        # read the next frame from the file
#        (grabbed, frame) = vs.read()
#        # if the frame was not grabbed, then we have reached the end
#        # of the stream
#        if not grabbed:
#            break
#        # resize the frame and then detect people (and only people) in it
#        frame = imutils.resize(frame, width=700)  
        text = "People can't cross the street"
        cv2.putText(frame, text, (10, frame.shape[0] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 255), 3)
        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
        counter =4
        while counter:
            time.sleep(1)
            counter-=1
            print('Car crossing')            

       
# After the loop release the cap object
vs.release()
# Destroy all the windows
cv2.destroyAllWindows()        