import cv2
import numpy as np
from typing import Mapping, Tuple

BLUE = (249, 209, 101)
WHITE = (255, 255, 255)

class Finger:
    def __init__(self, name, name_id, handedness):
        # identify finger
        self.name = name                # THUMB/INDEX/MIDDLE/RING/PINKY
        self.name_id = name_id          # 0-4
        self.handedness = handedness    # Left or Right

        # pitch related
        self.on_note = None 
        
        # finger state           
        self.on_screen = False          # present on screen?
        
        # position
        self.x, self.y = None, None
    
    # update finger present on screen
    def update(self, landmark, piano):
        self.on_screen = True 
        # update fingertip coordinates
        self.x, self.y = landmark[0], landmark[1]
        self.y = min(self.y, len(piano.key_segmentation)-1)
        self.x = min(self.x, len(piano.key_segmentation[0])-1)
        self.on_note = piano.key_segmentation[self.y][self.x]
        

# list of finger, utils for plotting
class Hands:
    def __init__(self):
        self.handedness_list = ["Left", "Right"]
        # finger objects
        self.fingers = []
        self.init_fingers()

        # formatting
        self.landmark_text_format = {
            "FONT_SIZE": 0.8,
            "FONT_THICKNESS": 1,
            "TEXT_COLOR": BLUE,
        }
    
    # create the 10 finger objects
    def init_fingers(self):
        finger_names = ["THUMB", "INDEX", "MIDDLE", "RING", "PINKY"]
        hands = ["Left", "Right"]
        for hand in hands:
            for finger_id, finger_name in enumerate(finger_names):
                finger = Finger(finger_name, finger_id, hand)
                self.fingers.append(finger)
    
    # interpret detection result
    def update_hands(self, hands, piano):
        if not hands:
            return 

        disp_map = {"Left": 0, "Right": 5}
        for hand in hands:
            start_id = disp_map[hand["type"]]
            landmarks = hand["lmList"]
            
            # for each finger in hand
            for idx in range(0, 5):
                finger = start_id + idx
                
                # retrieve landmark
                landmark_id = 4 * (idx + 1)
                landmark = landmarks[landmark_id]
                self.fingers[finger].update(landmark, piano)

    ######## DRAWING UTILITIES #########
    def draw_text(self, img, text, text_x, text_y):
        cv2.putText(
            img,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_DUPLEX,
            self.landmark_text_format["FONT_SIZE"],
            self.landmark_text_format["TEXT_COLOR"],
            self.landmark_text_format["FONT_THICKNESS"],
            cv2.LINE_AA,
        )
    
    def plot_fingertip_coor(self, img):
        # print("start plot")
        annotated_img = np.copy(img)
        
        text_start_pos = {"Left": 20, "Right": 600}
        finger_displacement = {"Left": 0, "Right": 5}
        
        for finger in self.fingers:
            if finger.on_screen:
                x, y = finger.x, finger.y
                cv2.circle(annotated_img, (x, y), 5, (0, 0, 255), -1)

        for handedness in self.handedness_list:
            text_x = text_start_pos[handedness]
            text_y = 40

            for finger_num in range(5):                         
                finger_id = finger_num + finger_displacement[handedness]
                finger = self.fingers[finger_id]
                
                text = f"{finger_id+1}: x {finger.x}, y {finger.y} ({finger.on_note});"

                self.draw_text(annotated_img, text, text_x, text_y)
                text_y += 25

       
        return annotated_img
