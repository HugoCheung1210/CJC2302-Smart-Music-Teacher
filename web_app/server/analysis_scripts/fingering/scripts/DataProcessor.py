import numpy as np
import pandas as pd
import ast 

class DataProcessor:
    def __init__(self, data, maxlen):
        self.data = data
        self.max_length = maxlen

    def get_sequences(self):
        sequences = []
        fingerings = []

        for seq in self.data:
            sequence = seq["sequence"]
            fingering = seq["fingering"]

            sequences.append(sequence)
            fingerings.append(fingering)

        return sequences, fingerings

    def get_max_sequence_length(self):
        max_length = 0

        for seq in self.data:
            sequence = seq["sequence"]
            length = sequence.shape[0]

            if length > max_length:
                max_length = length

        return max_length

    def preprocess_sequences(self, sequences, fingerings):
        maxlen = self.max_length
        
        padded_sequences = []
        padded_fingerings = []

        for seq, fingering in zip(sequences, fingerings):
            if seq.shape[0] < maxlen:
                pad_length = maxlen - seq.shape[0]

                seq_pad = np.vstack([np.zeros(3)] * pad_length)
                padded_seq = np.concatenate((seq, seq_pad), axis=0)

                fing_pad = np.zeros(pad_length)
                padded_fingering = np.concatenate((fingering, fing_pad))
            else:
                padded_seq = np.array(seq[:maxlen])
                padded_fingering = fingering[:maxlen]

            padded_sequences.append(padded_seq)
            padded_fingerings.append(padded_fingering)

        padded_sequences = np.stack(padded_sequences, axis=0)
        padded_fingerings = np.stack(padded_fingerings, axis=0)

        return padded_sequences, padded_fingerings

    def preprocess_data(self):
        sequences, fingerings = self.get_sequences()

        # add padding
        padded_sequences, padded_fingerings = self.preprocess_sequences(sequences, fingerings)

        return padded_sequences, padded_fingerings

    def get_data(self):
        return self.preprocess_data()

def generate_v2_data():
    csv_path = "fingering_dataset/mixed_features.csv"
    
    df = pd.read_csv(csv_path, delimiter=";")
    
    sequences = df["sequence"]
    rel_pitch = df["rel_sequence"]
    fingerings = df["fingering"]
    
    n_sample = len(sequences)
    time_diffs = []
    start_bws, end_bws = [], []
    trim_rel = []
    
    black_keys = [1, 3, 6, 8, 10]
    
    for i in range(n_sample):
        seq = ast.literal_eval(sequences[i])
        rel = ast.literal_eval(rel_pitch[i])
        fing = ast.literal_eval(fingerings[i])
        
        seq_len = len(seq)
        
        # compute new features
        time_diff = []
        start_bw, end_bw = [], []
        for j in range(seq_len):
            prev = seq[j-1] if j > 0 else [0, seq[j][1], seq[j][2]]
            
            time_diff.append(seq[j][0] - prev[0])
            
            if (int(prev[1]) % 12) in black_keys:
                start_bw.append(1)
            else:
                start_bw.append(0)
                
            if (int(seq[j][1]) % 12) in black_keys:
                end_bw.append(1)
            else:
                end_bw.append(0)
        
        # for rel pitch, if exceed 12 then set as 12
        for j in range(seq_len):
            if rel[j] > 12:
                rel[j] = 12
            elif rel[j] < -12:
                rel[j] = -12
        
        trim_rel.append(rel)
        time_diffs.append(time_diff)
        start_bws.append(start_bw)
        end_bws.append(end_bw)
    
    new_df = pd.DataFrame({
        "rel_pitch": trim_rel,
        "time_diff": time_diffs,
        "start_bw": start_bws,
        "end_bw": end_bws,
        "fingering": fingerings,
    })
    
    new_csv_path = "fingering_dataset/mixed_features_v2.csv"
    
    new_df.to_csv(new_csv_path, sep=";", index=False)
    
if __name__ == "__main__":
    generate_v2_data()
    print("Data generated")