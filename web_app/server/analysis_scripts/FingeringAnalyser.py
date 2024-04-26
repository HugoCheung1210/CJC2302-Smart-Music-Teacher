import numpy as np
import ast
import os
import pandas as pd
import matplotlib.pyplot as plt
import librosa
from utils import *
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import (
    Dense,
    LSTM,
    Dropout,
    Input,
    TimeDistributed,
    Masking,
    Bidirectional,
)

class FingeringAnalyser:
    def __init__(self):      
        # pairwise fingering cost table
        self.spans = {
            (1, 2): [[1, 5], [-3, 8], [-5, 10]],
            (1, 3): [[3, 7], [-2, 10], [-3, 12]],
            (1, 4): [[5, 9], [-1, 12], [-3, 14]],
            (1, 5): [[7, 10], [1, 13], [-1, 15]],
            (2, 3): [[1, 2], [1, 3], [1, 5]],
            (2, 4): [[3, 4], [1, 5], [1, 7]],
            (2, 5): [[5, 6], [2, 8], [2, 10]],
            (3, 4): [[1, 2], [1, 2], [1, 4]],
            (3, 5): [[3, 4], [1, 5], [1, 7]],
            (4, 5): [[1, 2], [1, 3], [1, 5]],
            (1, 1): [[0, 0], [0, 0], [0, 0]],
            (2, 2): [[0, 0], [0, 0], [0, 0]],
            (3, 3): [[0, 0], [0, 0], [0, 0]],
            (4, 4): [[0, 0], [0, 0], [0, 0]],
            (5, 5): [[0, 0], [0, 0], [0, 0]],
        }
        
        self.left_spans = {
            (1, 2): [[-5, -1], [-8, 3], [-10, 5]],
            (1, 3): [[-7, -3], [-10, 2], [-12, 3]],
            (1, 4): [[-9, -5], [-12, 1], [-14, 3]],
            (1, 5): [[-10, -7], [-13, 1], [-15, -1]],
            (2, 3): [[-2, -1], [-3, -1], [-5, -1]],
            (2, 4): [[-4, -3], [-5, -1], [-7, -1]],
            (2, 5): [[-6, -5], [-8, -2], [-10, -2]],
            (3, 4): [[-2, -1], [-2, -1], [-4, -1]],
            (3, 5): [[-4, -3], [-5, -1], [-7, -1]],
            (4, 5): [[-2, -1], [-3, -1], [-5, -1]],
            (1, 1): [[0, 0], [0, 0], [0, 0]],
            (2, 2): [[0, 0], [0, 0], [0, 0]],
            (3, 3): [[0, 0], [0, 0], [0, 0]],
            (4, 4): [[0, 0], [0, 0], [0, 0]],
            (5, 5): [[0, 0], [0, 0], [0, 0]],
        }
        
    # load midi file and convert to feature sequence
    def load_midi(self, midi_path):
        self.midi_path = midi_path
        self.note_times = read_midi_note_time(self.midi_path)

        print("self.note_times", self.note_times)
        raw_sequence = []
        for note_time in self.note_times:
            for note in note_time[0]:
                note_on = float(note_time[1])
                midi_number = librosa.note_to_midi(note)
                # is black keys?
                if midi_number % 12 in [1, 3, 6, 8, 10]:
                    is_black = 1
                else:
                    is_black = 0
                raw_sequence.append([note_on, is_black, midi_number])

        print("raw sequence", raw_sequence)
        # time_diff, start_bw, end_bw, rel_pitch
        sequence = []
        for i in range(len(raw_sequence)):
            if i == 0:
                time_diff = 0
                start_bw = raw_sequence[i][1]
                rel_pitch = 0
            else:
                time_diff = raw_sequence[i][0] - raw_sequence[i - 1][0]
                start_bw = raw_sequence[i - 1][1]
                rel_pitch = raw_sequence[i][2] - raw_sequence[i - 1][2]
            end_bw = raw_sequence[i][1]

            # normalize rel_pitch
            rel_pitch = (rel_pitch + 12) / 24.0

            sequence.append([time_diff, start_bw, end_bw, rel_pitch])
            
        self.sequence = sequence
        
        self.padded_sequence = pad_sequences(
            sequence, maxlen=256, padding="post", value=-1, dtype=float
        )

    def load_fingering(self, time, left_fingering, right_fingering, pitch):
        left_seq = {
            "time": [],
            "fingering": [],
            "pitch": [],
        }
        
        right_seq = {
            "time": [],
            "fingering": [],
            "pitch": [],
        }
        
        n = len(time)
        for i in range(n):
            if left_fingering[i]:
                left_seq["time"].append(time[i])
                left_seq["fingering"].append(left_fingering[i])
                left_seq["pitch"].append(pitch[i])
                
            if right_fingering[i]:
                right_seq["time"].append(time[i])
                right_seq["fingering"].append(right_fingering[i])
                right_seq["pitch"].append(pitch[i])
        
        self.left_seq = left_seq
        self.right_seq = right_seq
    
    # sequence to sequence model architecture
    def lstm_model(self, maxlen, num_classes):
        model = Sequential(name='LSTM_model')
        model.add(Input(shape=(maxlen, 4), name='Input_Layer'))
        
        # masking layer to omit paddings
        model.add(Masking(mask_value=-1, name='Masking_Layer'))
        
        model.add(Bidirectional(LSTM(1024, return_sequences=True), name='LSTM_1'))
        model.add(Dropout(0.2, name='Dropout_1'))
        
        model.add(Bidirectional(LSTM(512, return_sequences=True), name='LSTM_2'))
        model.add(Dropout(0.2, name='Dropout_2'))
        
        model.add(TimeDistributed(Dense(128, activation='relu'), name='Dense_1'))  # added for v6
        model.add(TimeDistributed(Dense(num_classes, activation='softmax'), name='Output_Layer'))
        
        return model

    # normalize and pad sequences
    def preprocess_data(self, sequences, fingerings=None, maxlen=128):
        # standardize rel_pitch from -12 to +12 to 0 to 1
        scaled_sequences = []
        for seq in sequences:
            scaled_seq = []
            for note in seq:
                if note[3] > 12:
                    note[3] = 12
                if note[3] < -12:
                    note[3] = -12
                    
                scaled_note = [note[0], note[1], note[2], (note[3] + 12) / 24.0]
                scaled_seq.append(scaled_note)
            scaled_sequences.append(scaled_seq)
        
        # pad sequences
        padded_sequences = pad_sequences(scaled_sequences, padding='post', value=-1, dtype=float, maxlen=maxlen)
        
        if fingerings:
            padded_fingerings = pad_sequences(fingerings, padding='post', value=0, maxlen=maxlen)
        else:
            padded_fingerings = None
        return padded_sequences, padded_fingerings

    def inference(self, seq=None):
        # current directory
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        model_dir = "fingering/models/lstm_v2/kaggle_v9"
        model_abs_dir = os.path.join(cur_dir, model_dir)
        model = self.lstm_model(128, 5)
        model.load_weights(os.path.join(model_abs_dir, "model.weights.h5"))
        
        print("model loaded")
        # print("model structure", model.summary())
        
        if not seq:
            seq = [[0.75, 0, 0, -48], [1.5, 0, 0, 5], [1.5, 0, 0, -5], [2.25, 0, 0, 5], [0.75, 0, 0, 0], [2.25, 0, 0, -4], [0.75, 0, 0, -1], [2.25, 0, 0, 5], [0.75, 0, 0, -5], [1.5, 0, 0, 5], [1.5, 0, 0, -5], [1.5, 0, 0, 5], [1.5, 0, 0, 0], [1.5, 0, 0, -4], [3.75, 0, 0, -1]]
        
        print("seq type", type(seq))
        # print("seq", seq)
        
        proc_seq, _ = self.preprocess_data([seq])
        
        # print("seq after preprocessing", seq)
        # print("seq shape", seq.shape)
        # print("seq type", type(seq))
        
        pred = model.predict(proc_seq, verbose=0)
        
        # print argmax of pred
        fingering = np.argmax(pred, axis=-1)
        print("fingering", fingering)
        fingering = fingering[0][:len(seq)]
        fingering = [ fing + 1 for fing in fingering ]
        
        
        # print("seq len", len(seq))
        print(fingering)
        print("fingering done")
        return fingering
    
    def inference_from_midi(self, midi_path):
        print("start inference")
        self.load_midi(midi_path)
        print("loaded midi")
        self.load_model()
        print("loaded model")
        self.inference()
        print("inference done")
        
        
    
    
    ############################## cost function ################################
    # left hand: swap pitch direction
    def single_cost(self, pitch, finger):
        # weak finger rule 
        if finger == 4 or finger == 5:
            return 1
        else :
            return 0
    
    # compute cost function
    def pair_cost(self, start_pitch, end_pitch, start_finger, end_finger, is_left=False):
        
        # thumb passing rule
        if end_finger == 1 and end_pitch % 12 in [1, 3, 6, 8, 10] and start_finger > 1 and start_pitch % 12 in [0, 2, 4, 5, 7, 9, 11]:
            cost += 2
        
        if start_finger > end_finger:
            start_finger, end_finger = end_finger, start_finger
            start_pitch, end_pitch = end_pitch, start_pitch
        
        pitch_diff = end_pitch - start_pitch
        
        span = self.spans[(start_finger, end_finger)]
        if is_left:
            span = self.left_spans[(start_finger, end_finger)]
        
        cost = 0
        # out of relax zone
        if pitch_diff < span[0][0]:
            cost += span[0][0] - pitch_diff
        
        if pitch_diff > span[0][1]:
            cost += pitch_diff - span[0][1]
        
        # double cost for non thumb
        if start_finger != 1 and end_finger != 1:
            cost *= 2
                
        # out of comfortable zone (small/large span rule)
        if pitch_diff < span[1][0]:
            cost += span[1][0] - pitch_diff
        
        if pitch_diff > span[1][1]:
            cost += pitch_diff - span[1][1]

        # three to four rule
        if start_finger == 3 and end_finger == 4:
            cost += 1
            
            # four on black rule
            if start_pitch % 12 in [0, 2, 4, 5, 7, 9, 11] and end_pitch % 12 in [1, 3, 6, 8, 10]:
                cost += 1
        
        # thumb on black
        if start_finger == 1 and start_pitch % 12 in [1, 3, 6, 8, 10]:
            cost += 1
            
            if end_pitch % 12 in [0, 2, 4, 5, 7, 9, 11]:
                cost += 2
        
        if end_finger == 1 and end_pitch % 12 in [1, 3, 6, 8, 10]:
            cost += 1
            
            if start_pitch % 12 in [0, 2, 4, 5, 7, 9, 11]:
                cost += 2
        
        # five on black rule
        if start_finger == 5 and start_pitch % 12 in [1, 3, 6, 8, 10]:
            if end_pitch % 12 in [0, 2, 4, 5, 7, 9, 11]:
                cost += 2
        
        if end_finger == 5 and end_pitch % 12 in [1, 3, 6, 8, 10]:
            if start_pitch % 12 in [0, 2, 4, 5, 7, 9, 11]:
                cost += 2
                
        # finger passing rule
        if pitch_diff < 0:
            cost += 1
            
        return cost 

    def triple_cost(self, start_pitch, mid_pitch, end_pitch, start_finger, mid_finger, end_finger, is_left=False):
        if start_finger > end_finger:
            start_finger, end_finger = end_finger, start_finger
            start_pitch, end_pitch = end_pitch, start_pitch
        
        cost = 0
        
        # position change count rule
        pitch_diff = end_pitch - start_pitch
        # check_change
        span = self.spans[(start_finger, end_finger)]
        if is_left:
            span = self.left_spans[(start_finger, end_finger)]
            
        # change occur
        if pitch_diff > span[1][1] or pitch_diff < span[1][0]:
            cost += 1
            # full change
            if mid_finger == 1 and mid_pitch >= start_pitch and mid_pitch <= end_pitch and (end_pitch - start_pitch < span[2][0] or end_pitch - start_pitch > span[2][1]):
                cost += 1
        
        # position change size rule
        if pitch_diff < span[1][0]:
            cost += span[1][0] - pitch_diff
        
        if pitch_diff > span[1][1]:
            cost += pitch_diff - span[1][1]
        
        # three_four_five rule 
        # if 345 appears in all, in any order
        if min(start_finger, mid_finger, end_finger) == 3 and max(start_finger, mid_finger, end_finger) == 5 and start_finger + mid_finger + end_finger == 12:
            cost += 1
            
        return cost 
    
    def compute_monophonic_cost(self, pitches, fingerings):
        n = len(pitches)
        cost = 0
        
        for i in range(n):
            cost += self.single_cost(pitches[i], fingerings[i])
        
        for i in range(n-1):
            cost += self.pair_cost(pitches[i], pitches[i+1], fingerings[i], fingerings[i+1])
        
        for i in range(n-2):
            cost += self.triple_cost(pitches[i], pitches[i+1], pitches[i+2], fingerings[i], fingerings[i+1], fingerings[i+2])

        return cost
        
    # chord: [ [[pitch1, pitch2, pitch3], [f1, f2, f3]], ...], pitch is midi number
    def compute_chord_cost(self, chords):
        n = len(chords)
        cost = 0
        
        # for each chord
        for i in range(n):
            pitches = chords[i][0]
            fingerings = chords[i][1]
            
            for j in range(len(pitches)):
                cost += self.single_cost(pitches[j], fingerings[j])
            
            # compute max pair of vertical cost 
            pair_costs = []
            for j in range(len(pitches)-1):
                for k in range(j+1, len(pitches)):
                    pair_costs.append(self.pair_cost(pitches[j], pitches[k], fingerings[j], fingerings[k]))
            cost += max(pair_costs)
        
        # for each pair of consecutive chords
        for i in range(n-1):
            m = len(chords[i][0])
            l = len(chords[i+1][0])
            
            start_pitches = chords[i][0]
            end_pitches = chords[i+1][0]
            start_fingerings = chords[i][1]
            end_fingerings = chords[i+1][1]
            
            pair_costs = []
            
            # compute the max pair of cost
            for j in range(m):
                for k in range(l):
                    pair_costs.append(self.pair_cost(start_pitches[j], end_pitches[k], start_fingerings[j], end_fingerings[k]))
                    
            cost += max(pair_costs)

        # extract note with smallest fingering number for each chord
        min_finger = []
        min_finger_pitch = [] 
        
        for i in range(n):
            pitches = chords[i][0]
            fingerings = chords[i][1]
            
            min_finger.append(min(fingerings))
            for j in range(len(pitches)):
                if fingerings[j] == min_finger[-1]:
                    min_finger_pitch.append(pitches[j])
                    break
        
        
        # for each triple of consecutive chords (extracted note)
        for i in range(n-2):
            cost += self.triple_cost(min_finger_pitch[i], min_finger_pitch[i+1], min_finger_pitch[i+2], min_finger[i], min_finger[i+1], min_finger[i+2])

            
            
            
    # def compute_chords(self):
    #     # convert into a list of chords: [ [[pitch1, pitch2, pitch3], [f1, f2, f3]], ...]
    #     n = len(time)
    #     left_chords, right_chords = [], []
    #     left_chord, right_chord = [[], []], [[], []]
    #     prev_time = 0
    #     for i in range(n):
    #         # new chord
    #         if time[i] != prev_time:
    #             # append if non-empty 
    #             if left_chord[0]:
    #                 left_chords.append(left_chord)
    #                 left_chord = [[], []]
                
    #             if right_chord[0]:
    #                 right_chords.append(right_chord)
    #                 right_chord = [[], []]
            
    #         # append to current chord
    #         if left_fingering[i]:
    #             left_chord[0].append(pitch[i])
    #             left_chord[1].append(left_fingering[i])
                
    #         if right_fingering[i]:
    #             right_chord[0].append(pitch[i])
    #             right_chord[1].append(right_fingering[i])
                    
if __name__ == "__main__":
    analyser = FingeringAnalyser()
    
    print("now testing inference")
    # analyser.inference_from_midi(".\\analysis_scripts\\fingering/fingering_dataset/midi_files/twinkle.mid")
    analyser.inference()
    exit()
    
    pitches = [60, 62, 64, 65, 67, 69, 71]
    fingerings = [1, 2, 3, 4, 5, 1, 2]
    
    print(analyser.compute_monophonic_cost(pitches, fingerings))
    
    
    fingerings = [1, 2, 3, 1, 2, 3, 4]
    print(analyser.compute_monophonic_cost(pitches, fingerings))