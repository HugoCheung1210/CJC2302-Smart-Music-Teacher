import cv2
import numpy as np 
import math 
import os 

class PianoKey:
    def __init__(self):
        self.color = None

        self.note = None
        self.octave = None

        self.x = None
        self.y = None
        self.w = None
        self.h = None

        self.centroid = None

    def init_from_box(self, box, color=None, note=None, octave=None):
        self.x, self.y, self.w, self.h, _ = box
        # convert all to int
        self.x, self.y, self.w, self.h = (
            int(self.x),
            int(self.y),
            int(self.w),
            int(self.h),
        )
        self.centroid = (self.x + self.w // 2, self.y + self.h // 2)

        self.color = color
        self.note = note
        self.octave = octave

    def plot(self, img, color=(255, 140, 0), plot_centroid=False, plot_note=False):
        # print(self.x, self.y, self.w, self.h, self.centroid)
        cv2.rectangle(
            img, (self.x, self.y), (self.x + self.w, self.y + self.h), color, 2
        )

        if plot_centroid:
            cv2.circle(img, self.centroid, 2, color, 1)

        if plot_note:
            if self.color == "black":
                h_displacement = self.h + 23
                note_text = f"{self.note[0]}'{self.octave}"
            else:
                h_displacement = self.h - 2
                note_text = f"{self.note}{self.octave}"
            
            cv2.putText(
                img,
                note_text,
                (self.x, self.y + h_displacement),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )


class Piano:
    def __init__(self, dir):
        self.dir = dir
        
        self.img = None
        self.height, self.width = None, None

        self.keyboard_top = None
        self.keyboard_bottom = None

        self.black_keys = []
        self.white_keys = []

    # import piano image
    def load_img(self, img_path, debug=False):
        MAX_WIDTH = 1000
        img = cv2.imread(img_path)

        # resize if width exceed max width
        # if img.shape[1] > MAX_WIDTH:
        #     new_h = int(MAX_WIDTH * img.shape[0] / img.shape[1])
        #     img = cv2.resize(img, (MAX_WIDTH, new_h))

        # new_h = int(MAX_WIDTH * img.shape[0] / img.shape[1])
        # img = cv2.resize(img, (MAX_WIDTH, new_h))

        self.img = img
        self.height, self.width = img.shape[:2]
        
        if debug:
            # cv2.imshow("original", self.img)
            print(f"image loaded from {img_path}, width {self.width}, height {self.height}")

    # hough transform: rotate image + find dom lines
    def do_hough(self, debug):
        self.gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        self.edge_img = self.edge(self.gray_img)
        if debug:
            tmp_img = cv2.resize(self.edge_img, (self.width // 2, self.height // 2))
            cv2.imshow("edge", tmp_img)

        self.edge_img = self.remove_small_regions(self.edge_img, threshold=200)

        """
            sobel_img = sobel(self.gray_img, ksize=1)
            sobel_binary = cv2.threshold(sobel_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            self.edge_img = remove_small_regions(sobel_binary, threshold=300)
        """
        # clean_edge_img = remove_small_regions(self.edge_img, threshold=100)

        self.dom_angle, self.hough_img, hough_lines = self.hough(
            self.edge_img, self.img, self.width
        )

        self.rot_matrix = self.get_rotation_matrix(self.img, self.dom_angle)

        # compute some rotated images
        self.rotate_img = self.rotate_image(self.img, self.rot_matrix)

        # cv2.imshow("gray", gray_img)
        # cv2.imshow("cleaned edge", self.edge_img)
        # cv2.imshow("clean edge", clean_edge_img)
        if debug:
            display_hough = cv2.resize(self.hough_img, (self.width // 2, self.height // 2))
            cv2.imshow("hough", display_hough)

        print("dominant angle:", self.dom_angle)

        self.hlines = self.get_hlines(
            hough_lines, self.rot_matrix, self.dom_angle, self.width
        )
        
        if debug:
            cv2.imshow("rotated", self.rotate_img)
            self.plot_hlines(self.rotate_img, self.hlines)

    def filter_keyboard(self, debug, main=False):
        # filter out white and black keys
        rotate_gray = self.rotate_image(self.gray_img, self.rot_matrix)
        _, white_th = cv2.threshold(rotate_gray, 127, 255, cv2.THRESH_BINARY)
        _, black_th = cv2.threshold(rotate_gray, 127, 255, cv2.THRESH_BINARY)
        black_th = cv2.bitwise_not(black_th)

        # cv2.imshow("white th", white_th)
        # cv2.imshow("black th", black_th)

        for top in self.hlines:
            for bottom in self.hlines:
                # filter out lines too close
                if bottom - top < self.height // 5:
                    continue

                # check if bottom 30% is mostly white
                U = int(bottom - (bottom - top) * 0.3)
                L = int(self.width * 0.1)
                R = int(self.width * 0.9)
                roi = white_th[U:bottom, L:R]
                white_count = cv2.countNonZero(roi)
                white_ratio = white_count / ((R - L) * (bottom - U))

                if debug:
                    print(f"top {top} and bottom {bottom}")
                    print(f"bottom white ratio {white_ratio}")
                if white_ratio < 0.9:
                    continue

                # check if top 60% is black + white
                D = int(top + (bottom - top) * 0.6)
                roi = white_th[top:D, L:R]
                white_count = cv2.countNonZero(roi)

                roi = black_th[top:D, L:R]
                black_count = cv2.countNonZero(roi)

                if debug:
                    print(f"white count {white_count}")
                    print(f"black count")
                white_ratio = white_count / ((R - L) * (D - top))
                black_ratio = black_count / ((R - L) * (D - top))

                if debug:
                    print(f"top white ratio {white_ratio}")
                    print(f"top black ratio {black_ratio}")

                if black_ratio < 0.3 or white_ratio < 0.4:
                    continue

                self.keyboard_top = top
                self.keyboard_bottom = bottom
                if debug:
                    print(f"line at {top} and {bottom}")
                break

        if self.keyboard_top == None:
            return

        # plot_hlines(self.rotate_img, [self.keyboard_top, self.keyboard_bottom])

        plot_img = self.rotate_img.copy()
        cv2.line(
            plot_img,
            (0, self.keyboard_top),
            (self.width, self.keyboard_top),
            (0, 255, 0),
            6,
        )
        cv2.line(
            plot_img,
            (0, self.keyboard_bottom),
            (self.width, self.keyboard_bottom),
            (0, 255, 0),
            6,
        )

        if debug:
            tmp_img = cv2.resize(plot_img, (self.width // 2, self.height // 2))
            cv2.imshow("keyboard", tmp_img)
        # cv2.imwrite(os.path.join(self.dir, "keyboard.jpg"), plot_img)

        crop_threshold = 5
        keyboard_img = self.rotate_img[
            self.keyboard_top - crop_threshold : self.keyboard_bottom + crop_threshold,
            :,
            :,
        ]
        if debug:
            # resize to 50% width of crop_keyboard_img
            tmp_img = cv2.resize(keyboard_img, (keyboard_img.shape[1] // 2, keyboard_img.shape[0] // 2))
            cv2.imshow("cropped keyboard", tmp_img)
            
        # crop the 20% to 80% of height for black key extraction
        crop_keyboard_img = keyboard_img[
            int(keyboard_img.shape[0] * 0.2) : int(keyboard_img.shape[0] * 0.8), :
        ]

        

        # do black threshold on crop img
        crop_gray = cv2.cvtColor(crop_keyboard_img, cv2.COLOR_BGR2GRAY)
        _, crop_black_th = cv2.threshold(crop_gray, 127, 255, cv2.THRESH_BINARY)
        crop_black_th = cv2.bitwise_not(crop_black_th)

        # cv2.imshow("filter black", crop_black_th)

        # apply dilation + erosion to smoothen
        kernel = np.ones((5, 5), np.uint8)
        crop_black_th = cv2.dilate(crop_black_th, kernel, iterations=1)
        # do erosion
        crop_black_th = cv2.erode(crop_black_th, kernel, iterations=1)
        # cv2.imshow("smoothened keyboard crop black", crop_black_th)

        # cv2.imshow("cropped black key", crop_black_th)

        # fill connected components
        # crop_black_th = fill_connected_components(crop_black_th)
        # cv2.imshow("filled cropped black key", crop_black_th)

        # apply erosion + dilation to remove noise
        kernel = np.ones((5, 5), np.uint8)
        crop_black_th = cv2.erode(crop_black_th, kernel, iterations=1)
        crop_black_th = cv2.dilate(crop_black_th, kernel, iterations=1)
        # cv2.imshow("denoised cropped black key", crop_black_th)

        # do region labelling
        black_key_boxes = self.region_label(crop_black_th)
        label_img = self.plot_box(black_key_boxes, crop_black_th)
        if debug:
            cv2.imshow("region label", label_img)

        black_key_boxes = self.filter_boxes(black_key_boxes)
        label_img = self.plot_box(black_key_boxes, crop_black_th)
        if debug:
            cv2.imshow("filtered key region", label_img)

        print(f"number of black keys {len(black_key_boxes)}")

        black_key_boxes = self.sort_boxes(black_key_boxes)

        for box in black_key_boxes:
            key = PianoKey()
            key.init_from_box(box, color="black")
            self.black_keys.append(key)

        self.label_black_keys(self.black_keys, crop_keyboard_img.shape[1], crop_keyboard_img)

        self.uncrop_boxes(
            self.keyboard_top - crop_threshold,
            self.keyboard_bottom + crop_threshold,
            self.black_keys,
        )
        
        self.compute_white_keys(self.black_keys, self.white_keys, keyboard_img.shape[0])

        # expand keys
        self.expand_keys(self.black_keys, top_expand_ratio=0.1, bottom_expand_ratio=0)
        self.expand_keys(self.white_keys, top_expand_ratio=0.1, bottom_expand_ratio=0.1)
        
        # white_keys_img = self.rotate_img.copy()
        keys_img = self.rotate_img.copy()
        for key in self.white_keys:
            key.plot(keys_img, color=(28, 96, 255), plot_centroid=True, plot_note=True)

        for key in self.black_keys:
            key.plot(keys_img, color=(255, 112, 41), plot_centroid=True, plot_note=True)

        cv2.imwrite(os.path.join(self.dir, "keyboard.jpg"), keys_img)
        
        if main:
            
            cv2.imwrite(f"{self.directory}\\{self.project_name}_keys.png", keys_img)

    # generate a w x h np array of key segments
    def gen_key_seg(self, debug, main=False):
        # gen np array of object dtype, dimension h x w
        key_seg_id = np.zeros((self.height, self.width))
        key_seg_map = []
        ct = 0
        for key in self.white_keys:
            x, y, w, h = key.x, key.y, key.w, key.h
            ct += 1
            key_seg_id[y : y + h, x : x + w] = ct
            key_seg_map.append((key.note, key.octave, key.color))
        if debug:
            print("#white keys =", ct)
        for key in self.black_keys:
            x, y, w, h = key.x, key.y, key.w, key.h
            ct += 1
            key_seg_id[y : y + h, x : x + w] = ct
            key_seg_map.append((key.note, key.octave, key.color))

        # rotate key seg id back to orignal image orientation
        self.reverse_rot_matrix = cv2.invertAffineTransform(self.rot_matrix)
        rotated_key_seg_id = self.rotate_image(key_seg_id, self.reverse_rot_matrix)
        if debug:
            print("rotation done, shape", rotated_key_seg_id.shape)

        # map id back to the tuple
        key_seg = [[None for x in range(self.width)] for y in range(self.height)]
        for y in range(self.height):
            for x in range(self.width):
                id = int(rotated_key_seg_id[y, x])
                if id == 0:
                    key_seg[y][x] = None
                else:
                    key_seg[y][x] = key_seg_map[id - 1]

        self.key_segmentation = key_seg
        plot = self.plot_keyseg(self.img, self.key_segmentation, transparency=0.5)
        if debug:
            tmp_plot = cv2.resize(plot, (self.width // 2, self.height // 2))
            cv2.imshow("key seg", tmp_plot)

    def detect_keys(self, debug=False, main=False):
        # print(f"start detecting keys, debug = {debug}")
        self.do_hough(debug)
        self.filter_keyboard(debug, main=main)
        self.gen_key_seg(debug, main=main)
            
        # try:
            
        # except:
        #     print("detect keys failed")
        #     return False

        return True 

    ###### UTILS FUNCTION ######
    ################### FOR IMAGE #########################
    def region_label(self, img):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img)

        print("num labels", num_labels)
        return stats


    def fill_connected_components(self, img):
        # fill out all the black holes in a connected component
        # find contours
        contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # draw contours
        filled_img = img.copy()
        filled_img = cv2.cvtColor(filled_img, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(filled_img, contours, -1, (0, 0, 255), 1)
        # fill contours
        for contour in contours:
            cv2.fillPoly(filled_img, pts=[contour], color=(255, 255, 255))
        # convert back to grayscale
        filled_img = cv2.cvtColor(filled_img, cv2.COLOR_BGR2GRAY)
        return filled_img

    # rotate image along center point
    def rotate_image(self, img, rot_matrix):
        height, width = img.shape[:2]
        rotated_img = cv2.warpAffine(img, rot_matrix, (width, height))
        return rotated_img


    ################### FOR PLOTTING #########################
    def plot_hlines(self, img, hlines):
        plot_img = np.copy(img)
        print(plot_img.shape)
        width = img.shape[1]
        for line in hlines:
            # print(f"plot height {line}")
            cv2.line(plot_img, (0, line), (width, line), (0, 255, 0), 2)
        cv2.imshow("hline on img", plot_img)

    def plot_box(self, stats, img):
        label_img = img.copy()
        # convert from binary to RGB if image is binary
        if len(label_img.shape) == 2:
            label_img = cv2.cvtColor(label_img, cv2.COLOR_GRAY2BGR)

        num_labels = len(stats)

        # plot each region on image
        for i in range(0, num_labels):
            # get region stats
            x, y, w, h, area = stats[i]
            cx, cy = x + w // 2, y + h // 2

            # draw bounding box
            cv2.rectangle(label_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # print(f"region {i} at {x}, {y}, {w}, {h}, {area}, centroid {cx}, {cy}")
            # draw centroid
            cv2.circle(label_img, (int(cx), int(cy)), 2, (255, 0, 0), 1)

        return label_img


    def plot_keyseg(
        self, img, key_seg, white_color=(255, 0, 0), black_color=(0, 255, 0), transparency=0.4
    ):
        height, width = img.shape[:2]
        overlay_img = np.zeros_like(img)
        for y in range(height):
            for x in range(width):
                if key_seg[y][x] == None:
                    continue

                if key_seg[y][x][2] == "white":
                    overlay_img[y, x] = white_color
                elif key_seg[y][x][2] == "black":
                    overlay_img[y, x] = black_color

        plot_img = cv2.addWeighted(img, 1 - transparency, overlay_img, transparency, 0)
        return plot_img


    ################### FOR FILTERING #########################
    def absolute_thresholding(self, img):
        ret, white_th = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        ret, black_th = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        return white_th, black_th


    ############## FOR HOUGH TRANSFORM ################
    def edge(self, img):
        # canny edge for now, may change
        t1 = 50
        t2 = 150
        edge_img = cv2.Canny(img, t1, t2)
        return edge_img


    def remove_small_regions(self, binary_image, threshold=10):
        # Apply region labeling to the binary image
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_image
        )

        # Loop through all connected components and remove those whose area is less than a specified threshold
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < threshold:
                labels[labels == i] = 0

        # Use the cv2.bitwise_and function to remove the small regions from the binary image
        result = cv2.bitwise_and(binary_image, binary_image, mask=labels.astype("uint8"))

        return result


    def hough(self, canny_img, raw_img, MAX_WIDTH=1000, hough_threshold=200):
        hough_img = np.copy(raw_img)

        lines = cv2.HoughLines(canny_img, 1, np.pi / 180, hough_threshold, None, 0, 0)

        # find dominating angle
        angles = []
        line_lengths = []
        for line in lines:
            rho, theta = line[0]
            # angle = float(round(theta * 180 / np.pi))
            angle = theta * 180 / np.pi
            angles.append(angle)

        # find higest occurence in range 45 to 135
        filtered_angles = [angle for angle in angles if angle>=45 and angle<=135]
        # Count the occurrences of each unique angle
        # from collections import Counter
        # angle_counts = Counter(angles)

        # Sort the unique angles based on their occurrence count in descending order
        # sorted_angles = sorted(set(angles), key=lambda angle: angle_counts[angle], reverse=True)

        # Print the unique angles in descending order of occurrence
        # for angle in sorted_angles:
        #     print(angle)
    
        dom_angle = max(set(filtered_angles), key=filtered_angles.count)
        # dom_angle = max(set(angles), key=angles.count)
        # Bin angles into a histogram with 36 bins
        # hist, bin_edges = np.histogram(angles, bins=36)

        # # Find bin with highest count in histogram
        # dominant_bin = np.argmax(hist)

        # # Calculate dominant angle as center of bin with highest count
        # dom_angle = (bin_edges[dominant_bin] + bin_edges[dominant_bin+1]) / 2

        for line in lines:
            rho, theta = line[0]
            angle = theta * 180 / np.pi
            a, b = math.cos(theta), math.sin(theta)
            x0, y0 = a * rho, b * rho

            pt1 = (int(x0 + MAX_WIDTH * (-b)), int(y0 + MAX_WIDTH * (a)))
            pt2 = (int(x0 - MAX_WIDTH * (-b)), int(y0 - MAX_WIDTH * (a)))

            color = (0, 255, 0)
            # color = (0, 0, 255) if (abs(angle - dom_angle) <= 0) else (255, 0, 0)

            cv2.line(hough_img, pt1, pt2, color, 2, cv2.LINE_AA)

        return dom_angle, hough_img, lines


    # represent each horizontal line as height
    def get_hlines(self, lines, rot_matrix, dom_angle, MAX_WIDTH=1000):
        hlines = []
        threshold = 0

        for line in lines:
            rho, theta = line[0]
            angle = theta * 180 / np.pi
            if abs(angle - dom_angle) <= threshold:
                a, b = math.cos(theta), math.sin(theta)
                x0, y0 = a * rho, b * rho

                pt1 = (int(x0 - MAX_WIDTH * b), int(y0 + MAX_WIDTH * (a)))
                pt2 = (int(x0 + MAX_WIDTH * b), int(y0 - MAX_WIDTH * (a)))

                # calculate its height
                (new_x, new_y) = cv2.transform(np.array([(pt1, pt2)]), rot_matrix)[0][0]

                height = new_y
                # height = int(pt1[0] * math.sin(rot_angle) + pt1[1] * math.cos(rot_angle))

                # add to hlines
                hlines.append(height)

        # make distinct then sort
        hlines = list(set(hlines))
        hlines.sort()
        return hlines


    def get_rotation_matrix(self, img, dom_angle):
        rot_angle = dom_angle - 90
        height, width = img.shape[:2]

        # Calculate the rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), rot_angle, 1)
        return rotation_matrix


    ############## FOR KEY DETECTION ################
    def filter_boxes(self, stats):
        # get the dominant width and height of the regions in the width center 50% of image
        widths = []
        bottoms = []
        for x, y, w, h, area in stats:
            if (x > 1000 * 0.25) and (x + w < 1000 * 0.75):
                widths.append(w)
                bottoms.append(y + h)
        # get the dominant width = average of widths
        dominant_width = np.mean(widths)

        # get the dominant bottom = average of bottoms
        dominant_bottom = np.mean(bottoms)

        print(f"dominant width {dominant_width}, dominant bottom {dominant_bottom}")

        # filter out regions that are over threshold diff with dominant width and bottom
        filtered_stats = []
        for x, y, w, h, area in stats:
            if (abs(w - dominant_width) < dominant_width) and (
                abs(y + h - dominant_bottom) < 20
            ):
                filtered_stats.append((x, y, w, h, area))
            else:
                # print(f"filtered out {x}, {y}, {w}, {h}, {area}")
                pass
        return filtered_stats


    def uncrop_boxes(self, crop_top, crop_bottom, keys):
        # crop algo: crop_top to crop_bottom, then get 20% to 80% of height
        # get the total height being cropped
        crop_height = int((crop_bottom - crop_top) * 0.2 + crop_top)
        for key in keys:
            key.y = int(key.y + crop_height - (crop_bottom - crop_top) * 0.1)
            key.h = int(key.h + (crop_bottom - crop_top) * 0.1)

            # update centroid
            key.centroid = (key.x + key.w // 2, key.y + key.h // 2)

            key.area = int(key.w) * int(key.h)

    # expand height of keys
    def expand_keys(self, keys, top_expand_ratio=0, bottom_expand_ratio=0):
        for key in keys:
            key.y = int(key.y - key.h * top_expand_ratio)
            key.h = int(key.h * (1 + top_expand_ratio + bottom_expand_ratio))
            key.centroid = (key.x + key.w // 2, key.y + key.h // 2)
    
    def sort_boxes(self, boxes):
        # sort by x, then y
        boxes = sorted(boxes, key=lambda x: (x[0], x[1]))
        return boxes


    def get_note(self, note, octave, diff, mode="full"):
        if mode == "full":
            notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        elif mode == "black":
            notes = ["C#", "D#", "F#", "G#", "A#"]
        elif mode == "white":
            notes = ["C", "D", "E", "F", "G", "A", "B"]

        note_id = notes.index(note)
        new_note_id = (note_id + diff) % len(notes)

        new_octave = octave + (note_id + diff) // len(notes)
        return notes[new_note_id], new_octave


    def label_black_keys(self, black_keys, img_width, img):
        CD_pairs = []
        # find all pairs of black keys where distance between them is less than distance between their other neighbours
        for i in range(1, len(black_keys) - 2):
            key1 = black_keys[i]
            key2 = black_keys[i + 1]

            # if distance between them is less than distance between their other neighbours
            dist_self = key2.x - key1.x
            dist_left = key1.x - black_keys[i - 1].x
            dist_right = black_keys[i + 2].x - key2.x

            if dist_self * 1.3 < dist_left and dist_self * 1.3 < dist_right:
                CD_pairs.append(((key1.centroid[0] + key2.centroid[0]) // 2, i, i + 1))

        # plot detected pairs
        # tmp_plot = img.copy()
        # for pair in CD_pairs:
        #     black_keys[pair[1]].plot(tmp_plot, color=(0, 255, 0))
        #     black_keys[pair[2]].plot(tmp_plot, color=(0, 255, 0))
        # cv2.imshow("CD pair plot", tmp_plot)

        # find the pair nearest to center of image
        min_dist = img_width
        min_pair = None
        for pair in CD_pairs:
            dist = abs(img_width // 2 - pair[0])
            if dist < min_dist:
                min_dist = dist
                min_pair = pair

        # label the pair as C#4 and D#4
        black_keys[min_pair[1]].note = "C#"
        black_keys[min_pair[1]].octave = 4
        black_keys[min_pair[2]].note = "D#"
        black_keys[min_pair[2]].octave = 4

        # label the remaining black keys based on their relative order with C#4 and D#4
        cur_note = ("C#", 4)
        for i in range(min_pair[1] - 1, -1, -1):
            new_note = self.get_note(cur_note[0], cur_note[1], -1, "black")
            black_keys[i].note = new_note[0]
            black_keys[i].octave = new_note[1]
            cur_note = new_note

        cur_note = ("D#", 4)
        for i in range(min_pair[2] + 1, len(black_keys)):
            new_note = self.get_note(cur_note[0], cur_note[1], 1, "black")
            black_keys[i].note = new_note[0]
            black_keys[i].octave = new_note[1]
            cur_note = new_note


    def compute_white_keys(self, black_keys, white_keys, keyboard_height):
        # build the white keys before current black key
        for i in range(1, len(black_keys)):
            if black_keys[i].note in ["C#", "F#"]:
                left_x = (black_keys[i - 1].centroid[0] + black_keys[i].centroid[0]) // 2
            else:
                left_x = black_keys[i - 1].centroid[0]

            right_x = black_keys[i].centroid[0]

            y = black_keys[i].y
            h = (keyboard_height - 10) * 0.9
            w = right_x - left_x

            note, octave = self.get_note(black_keys[i].note, black_keys[i].octave, -1, "full")
            white_key = PianoKey()
            white_key.init_from_box(
                (left_x, y, w, h, 0), color="white", note=note, octave=octave
            )

            white_keys.append(white_key)

        # build the white keys after current black key
        for i in range(0, len(black_keys) - 1):
            if black_keys[i].note not in ["A#", "D#"]:
                continue
            right_x = (black_keys[i + 1].centroid[0] + black_keys[i].centroid[0]) // 2
            left_x = black_keys[i].centroid[0]

            y = black_keys[i].y
            h = (keyboard_height - 10) * 0.9
            w = right_x - left_x

            note, octave = self.get_note(black_keys[i].note, black_keys[i].octave, 1, "full")
            white_key = PianoKey()
            white_key.init_from_box(
                (left_x, y, w, h, 0), color="white", note=note, octave=octave
            )

            white_keys.append(white_key)

if __name__ == "__main__":
    
    directory = "E:\CUHK\FYP\code\\videos\\test_key_seg"
    files = os.listdir(directory)
    
    for file in ["studio_case.png"]:
        print(f"process {file}")
        piano = Piano()
        piano.directory = directory
        piano_path = piano.directory + "\\" + file
        piano.project_name = file[:-4]
        piano.load_img(piano_path, debug=False)
        piano.detect_keys(debug=False, main=True)
        cv2.waitKey(0)
        cv2.destroyAllWindows()