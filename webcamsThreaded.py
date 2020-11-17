import cv2
import threading

# Start webcam captures
NUM_CAMERAS = 2
CAM_BUFFERS = [0]*NUM_CAMERAS

def threadedRetrieveCamFrame(id):
    print("Threaded webcam started with id: ", id)
    cap = cv2.VideoCapture(id)
    while True:
        ret, frame = cap.read()
        CAM_BUFFERS[id] = frame

for i in range(NUM_CAMERAS):
    t = threading.Thread(target=threadedRetrieveCamFrame, args=([i]), daemon=True)
    t.start()