import os
import argparse
import matplotlib.pyplot as plt
import subprocess

from FingeringAnalyser import FingeringAnalyser
from utils import *
from style_classification import *


## interface script for piece analysis (input midi, output fingering + style transfer)
class PieceAnalyser:
    def __init__(self):
        self.init()

    def init(self):
        parser = argparse.ArgumentParser(description="perform style transfer")
        parser.add_argument("--dir", type=str, help="directory to analyze")
        parser.add_argument("--musescorePath", type=str, help="path to musescore")
        parser.add_argument("--mode", type=str, help="analyze or generate")
        parser.add_argument("--sourceStyle", type=str, help="source style")
        parser.add_argument("--targetStyle", type=str, help="target style")
        
        # parse the arguments
        args = parser.parse_args()
        self.args = args

        self.dir = args.dir
        self.musescore_path = args.musescorePath
        self.mode = args.mode
        print(f"dir: {self.dir}, musescore_path: {self.musescore_path}, mode: {self.mode}")

        file_name = "input.mid"
        filepath = os.path.join(self.dir, file_name)

        self.input_path = filepath
        self.output_path = os.path.join(self.dir, "output.mid")

    def generate_fingering(self, note_time):
        ct = len(note_time)
        left_seq_abs = []
        right_seq_abs = []
        correct = []
        prev_time = 0
        for i in range(ct):
            cur_time = float(note_time[i][1])
            print("cur_time: ", cur_time)
            
            # find max midi num in the note
            max_midi = 0
            for midi_note in note_time[i][0]:
                max_midi = max(max_midi, librosa.note_to_midi(midi_note.replace("♯", "#")))
            
            for midi_note in note_time[i][0]:
                correct.append([1, midi_note])
                
                midi_note = midi_note.replace("♯", "#")
                # Change pitch into midi number
                midi_num = librosa.note_to_midi(midi_note)
                
                cur_key_color = 0
                if midi_num % 12 in [1, 3, 6, 8, 10]:
                    cur_key_color = 1
                
                if midi_num < 60 or max_midi - midi_num > 12:
                    left_seq_abs.append([midi_num, 0, cur_key_color, cur_time])
                else:
                    right_seq_abs.append([midi_num, 0, cur_key_color, cur_time])
                    
        prev_time = 0
        prev_pitch = 0
        prev_key = 0
        
        left_seq, right_seq = [], []
        for i in range(len(left_seq_abs)):
            left_seq.append([left_seq_abs[i][3]-prev_time, prev_key, left_seq_abs[i][2],  (left_seq_abs[i][0]-prev_pitch)*-1])
            
            prev_time = left_seq_abs[i][3]
            prev_pitch = left_seq_abs[i][0]
            prev_key = left_seq_abs[i][2]
        
        prev_time = 0
        prev_pitch = 0
        prev_key = 0
        for i in range(len(right_seq_abs)):
            right_seq.append([right_seq_abs[i][3]-prev_time, prev_key, right_seq_abs[i][2], right_seq_abs[i][0]-prev_pitch])
            
            prev_time = right_seq_abs[i][3]
            prev_pitch = right_seq_abs[i][0]
            prev_key = right_seq_abs[i][2]
        
        self.fingering_analyser = FingeringAnalyser()
        
        # print("left_seq: ", left_seq)
        left_finger = self.fingering_analyser.inference(left_seq)
        # print("left_finger: ", left_finger)
        
        # print("right_seq: ", right_seq)
        if len(right_seq) > 128:
            # send it in multiple parts of 128
            right_finger = []
            for i in range(0, len(right_seq), 128):
                right_finger += self.fingering_analyser.inference(right_seq[i:i+128])                
        # right_finger = self.fingering_analyser.inference(right_seq)
        
        print("len left_seq: ", len(left_seq))
        print("len right_seq: ", len(right_seq))
        # right_finger = [5, 4, 1, 2, 3, 2, 4, 2, 2, 2, 2, 4, 4, 3, 2, 1, 2, 3, 3, 3, 3, 2, 2, 2, 1, 3]
        # left_finger = [1, 3, 1, 3, 3, 2, 1, 4, 1, 4, 1, 3, 3, 1, 5]
        print("len left_finger: ", len(left_finger))
        print("len right_finger: ", len(right_finger))
        print("correct ct: ", len(correct))
        combined_finger = []
        hands = []
        left_ct, right_ct = 0, 0
        for i in range(ct):
            max_midi = 0
            for midi_note in note_time[i][0]:
                max_midi = max(max_midi, librosa.note_to_midi(midi_note.replace("♯", "#")))
                
            for midi_note in note_time[i][0]:
                midi_note = midi_note.replace("♯", "#")
                # Change pitch into midi number
                midi_num = librosa.note_to_midi(midi_note)
                
                if midi_num < 60 or max_midi - midi_num > 12:
                    combined_finger.append(left_finger[left_ct])
                    hands.append("Left")
                    left_ct += 1
                else:
                    if right_ct == len(right_finger):
                        combined_finger.append(0)
                        hands.append("Right")
                        print("right finger out of range", right_ct)
                    combined_finger.append(right_finger[right_ct])
                    hands.append("Right")
                    right_ct += 1
        
        return combined_finger, hands, correct
    
    # analyze music and classify style
    def analyze(self):
        print("start analyze")
        # parse input midi file and generate score
        self.input_note_time = read_midi_note_time(self.input_path)
        # print("note time: ", self.input_note_time)
        
        combined_finger, hands, correct = self.generate_fingering(self.input_note_time)
                    
        # generate score from input midi
        note_time2score(
            perf_note_time=self.input_note_time,
            musescore_path=self.musescore_path,
            output_xml_path=os.path.join(self.dir, "input_score.xml"),
            output_image_path=os.path.join(self.dir, "input_score.png"),
            fingerings=combined_finger,
            hand=hands,
            correct=correct
        )
        
        print("score generated")
        
        # generate audio from input midi
        subprocess.run([self.musescore_path, "-o", os.path.join(self.dir, "input_score.wav"), self.input_path])
        
        # style_classification(self.audio_path)

        
        # *** must print the type of the piece at last output line ***
        # one of ["Baroque", "Classical", "Romantic", "Modern"];
        print("Baroque")
    
    def tmp_gen(self):
        print("start analyze")
        # parse output midi file and generate score
        self.output_note_time = read_midi_note_time(self.output_path)
        # print("note time: ", self.output_note_time)
        
        combined_finger, hands, correct = self.generate_fingering(self.output_note_time)
        
        # generate score from output midi
        note_time2score(
            perf_note_time=self.output_note_time,
            musescore_path=self.musescore_path,
            output_xml_path=os.path.join(self.dir, "output_score.xml"),
            output_image_path=os.path.join(self.dir, "output_score.png"),
            fingerings=combined_finger,
            hand=hands,
            correct=correct
        )
        
        print("score generated")
        
        # generate audio from output midi
        subprocess.run([self.musescore_path, "-o", os.path.join(self.dir, "output_score.wav"), self.output_path])
        
    # generate music
    def generate(self):
        source_style = self.args.sourceStyle
        target_style = self.args.targetStyle
        
        
        output_midi_path = os.path.join(self.dir, "output.mid")
        
        ### call style transfer model, output midi
        # output_midi_path = self.input_path

        # Subprocess call to style transfer model
        subprocess.run(["python ./style_transfer/style_transfer.py ", 
                        "--input_path" , self.input_path,
                        "--output_path", output_midi_path,
                        "--sourceStyle", source_style, 
                        "--targetStyle", target_style])


        ### end style transfer model
        
        self.output_note_time = read_midi_note_time(output_midi_path)
        
        # generate fingering
        
        output_xml_path = os.path.join(self.dir, "output_score.xml")
        output_image_path = os.path.join(self.dir, "output_score.png")
        output_wav_path = os.path.join(self.dir, "output_score.wav")
        
        # generate score 
        note_time2score(
            perf_note_time=self.output_note_time,
            musescore_path=self.musescore_path,
            output_xml_path=output_xml_path,
            output_image_path=output_image_path,
        )
        
        # generate audio
        subprocess.run([self.musescore_path, "-o", output_wav_path, output_midi_path])
        print("generation end")


if __name__ == "__main__":
    analyser = PieceAnalyser()
    if analyser.mode == "analyze":
        # print("Modern")
        # exit()
        # analyser.tmp_gen()
        analyser.analyze()
    
    elif analyser.mode == "transfer":
        # analyser.tmp_gen()
        analyser.generate()