import cv2
from cvzone.HandTrackingModule import HandDetector
import time
import numpy as np
import pandas as pd 

from utils import *
from Hands import Hands

class HandTracker:
    def __init__(self, piano, video_path, rot_deg):
        self.hands = Hands()
        self.piano = piano 
        self.video_path = video_path
        self.background_rot_deg = rot_deg
        self.record = []

    def run(self, sample_interval, output_path=None, debug=False, output_width=None):
        capture = cv2.VideoCapture(self.video_path)

        # get width and height of capture
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if self.background_rot_deg == 90 or self.background_rot_deg == 270:
            width, height = height, width
        
        if output_width != None:
            height = int(output_width * height / width)   
            width = output_width
                 

        print(f"width: {width}, height: {height}")

        fps = capture.get(cv2.CAP_PROP_FPS)
        print("fps", fps)
        
        cap_frame_interval = int(fps * sample_interval / 1000)
        frame_count = 0
        # print(f"cap_frame_interval: {cap_frame_interval}")

        if (output_path != None):
            output = cv2.VideoWriter(output_path,
                            cv2.VideoWriter_fourcc(*'MP4V'),
                            fps,
                            (width, height))

        detector = HandDetector(detectionCon=0.5, maxHands=2)
        
        self.key_finger_list = []

        while True:
            ret, img = capture.read()

            if not ret:
                print("read failed")
                break
                # continue

            # prepare landmarker inputs            
            det_hands, annotated_img = detector.findHands(img)
            
            self.hands.update_hands(det_hands, self.piano)
            
            # mark result
            if frame_count % cap_frame_interval == 0:
                keys = []
                key_finger = []
                for finger in self.hands.fingers:
                    key_finger.append([finger.handedness, finger.name_id, finger.on_note])
                    
                    if finger.on_note != None:
                        note = finger.on_note[0] + str(finger.on_note[1])
                        
                        keys.append(note)
                        
                self.record.append((frame_count / fps, keys))
                self.key_finger_list.append([frame_count / fps, key_finger])    # timestamp, [handedness, finger_id, on_note]
                
            frame_count += 1
            
            # plot output
            annotated_img = self.hands.plot_fingertip_coor(annotated_img)
            
            annotated_img = cv2.resize(annotated_img, (width, height))

            if output_path != None:
                ret = output.write(annotated_img)

            if debug:
                cv2.imshow("annotated", annotated_img)
                
            cv2.waitKey(1)

        # sort key_finger_list by timestamp
        self.key_finger_list.sort(key=lambda x: x[0])
        capture.release()
        if output_path != None:
            output.release()
        if debug:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    tracker = HandTracker 
    tracker.run()
