import os 
import numpy as np
import pandas as pd 
from hmmlearn import hmm
import json 
import ast 

finger_files_dir = "D:\CUHK\FYP\code\piano_fingering\PianoFingeringDataset_v1.2\FingeringFiles"
all_features_filename = "fingering_dataset/features.csv"

# convert spelled pitch to midi note
def convert_spelled_pitch(spelled_pitch):
    note = spelled_pitch[:-1]
    octave = int(spelled_pitch[-1])
    
    note_to_midi = {
        "C": 0,
        "C#": 1,
        "Db": 1,
        "D": 2,
        "D#": 3,
        "Eb": 3,
        "E": 4,
        "F": 5,
        "F#": 6,
        "Gb": 6,
        "G": 7,
        "G#": 8,
        "Ab": 8,
        "A": 9,
        "A#": 10,
        "Bb": 10,
        "B": 11
    }
    
    return int(12 + 12 * octave + note_to_midi[note])

def import_data():
    sequences = []
    fingerings = []
    
    filenames = os.listdir(finger_files_dir)
    for filename in filenames:
        if filename.endswith(".txt"):
            data = pd.read_csv(os.path.join(finger_files_dir, filename), sep='\t', skiprows=1, header=None, names=["id", "onset_time", "offset_time", "spelled_pitch", "onset_velocity", "offset_velocity", "channel", "finger"], index_col=0)
            # print(data)
            
            # sequence = [ [onset_time, offset_time, spelled_pitch] ]
            sequence = []
            for index, row in data.iterrows():
                sequence.append([float(row["onset_time"]), float(row["offset_time"]), convert_spelled_pitch(row["spelled_pitch"])])
            
            sequences.append(np.array(sequence))
            
            fingerings.append(data["finger"].values)
            # break 
    
    # print(sequences)
    # print(fingerings)
    
    return sequences, fingerings

def preprocess_data(sequences, fingerings):
    # separate left and right fingers
    left_sequences = []
    right_sequences = []
    left_fingerings = []
    right_fingerings = []
    
    for sequence, fingering in zip(sequences, fingerings):
        # if fingering is string, for fingering with s_e, make it to s
        if type(fingering[0]) == str:        
            fingering = np.array([int(finger.split("_")[0]) for finger in fingering])
        
        # separate left and right hand
        negative_indices = np.where(fingering < 0)
        
        # all values < 0 are left hand
        left_sequence = sequence[negative_indices]
        left_fingering = fingering[negative_indices]
        
        # all values >= 0 are right hand
        positive_indices = np.where(fingering > 0)
        right_sequence = sequence[positive_indices]
        right_fingering = fingering[positive_indices]
        
        left_sequences.append(left_sequence)
        right_sequences.append(right_sequence)
        left_fingerings.append(left_fingering)
        right_fingerings.append(right_fingering) 

        # print("left sequence", left_sequence)
    
    # compute relative pitch 
    left_rel_sequences = []
    right_rel_sequences = []
    
    # from back to front cur_rel_pitch = cur_pitch - prev_pitch
    # handle left sequence
    for left_sequence in left_sequences:
        left_rel_sequence = [0]
        
        prev_pitch = left_sequence[0][2]
        for i in range(1, len(left_sequence)):
            left_rel_sequence.append(left_sequence[i][2] - prev_pitch)
            prev_pitch = left_sequence[i][2]
        
        left_rel_sequences.append(np.array(left_rel_sequence))
    
    # handle right sequence
    for right_sequence in right_sequences:
        right_rel_sequence = [0]
        
        prev_pitch = right_sequence[0][2]
        for i in range(1, len(right_sequence)):
            right_rel_sequence.append(right_sequence[i][2] - prev_pitch)
            prev_pitch = right_sequence[i][2]
        
        right_rel_sequences.append(np.array(right_rel_sequence))
    
    # return all
    return left_sequences, right_sequences, left_fingerings, right_fingerings, left_rel_sequences, right_rel_sequences
       
def generate_processed_data():
    sequences, fingerings = import_data()
    left_sequences, right_sequences, left_fingerings, right_fingerings, left_rel_sequences, right_rel_sequences = preprocess_data(sequences, fingerings)
    
    print(len(sequences))
    
    # convert all to string representation
    sequences = [json.dumps(sequence.tolist()) for sequence in sequences]
    fingerings = [json.dumps(fingering.tolist()) for fingering in fingerings]
    left_sequences = [json.dumps(sequence.tolist()) for sequence in left_sequences]
    right_sequences = [json.dumps(sequence.tolist()) for sequence in right_sequences]
    left_fingerings = [json.dumps(fingering.tolist()) for fingering in left_fingerings]
    right_fingerings = [json.dumps(fingering.tolist()) for fingering in right_fingerings]
    left_rel_sequences = [json.dumps(sequence.tolist()) for sequence in left_rel_sequences]
    right_rel_sequences = [json.dumps(sequence.tolist()) for sequence in right_rel_sequences]
    
    
    
    # put all these into a dataframe and save
    data = {
        "sequence": sequences,
        "fingerings": fingerings,
        "left_sequence": left_sequences,
        "right_sequence": right_sequences,
        "left_fingering": left_fingerings,
        "right_fingering": right_fingerings,
        "left_rel_sequence": left_rel_sequences,
        "right_rel_sequence": right_rel_sequences
    }
    # save to features.csv
    df = pd.DataFrame(data)
    df.to_csv(all_features_filename, index=False, sep=";")

def generate_LR_data():
    df = pd.read_csv(all_features_filename, delimiter=";")
    # print(df.head())
    
    left_df = df[["left_sequence", "left_fingering", "left_rel_sequence"]]
    right_df = df[["right_sequence", "right_fingering", "right_rel_sequence"]]
    
    
    # change name to sequence, fingering, rel_sequence
    left_df.columns = ["sequence", "fingering", "rel_sequence"]
    right_df.columns = ["sequence", "fingering", "rel_sequence"]
    
    # save to right_features.csv
    right_df.to_csv("fingering_dataset/right_features.csv", index=False, sep=";")
    
    # process left df
    # multiply fingering & relative pitch by -1 (aka flip left hand to right hand)
    # left_df["fingering"] = list(left_df["fingering"])
    print("left df head", left_df.head())
    print("left df fingering", left_df["fingering"].head())
    
    # for each row, multiply fingering by -1
    for index, row in left_df.iterrows():
        # print("row fingering", row["fingering"])
        
        list_row = ast.literal_eval(str(row["fingering"]))
        list_row = [val * -1 for val in list_row]
        row["fingering"] = list_row
        
        list_row = ast.literal_eval(str(row["rel_sequence"]))
        # print("rel_sequence list row", list_row)
        list_row = [int(val) * -1 for val in list_row]
        row["rel_sequence"] = list_row
        # row["rel_sequence"] *= -1
        # print("row fingering", row["fingering"])

    # print("left df fingering", left_df["fingering"].head())
    print("left df head", left_df.head())
    
    # save to left_features.csv 
    left_df.to_csv("fingering_dataset/left_features.csv", index=False, sep=";")
    
    # concat into one dataframe (mixing two data)
    mixed_df = pd.concat([left_df, right_df])
    mixed_df.to_csv("fingering_dataset/mixed_features.csv", index=False, sep=";")
    
if __name__ == "__main__":
    # generate_processed_data()
    generate_LR_data() 