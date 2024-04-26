import os 
import matplotlib.pyplot as plt
import pandas as pd
import ast
import numpy as np
from keras.models import load_model
import tensorflow as tf

model_dir = 'models/bilstm'
model_run = '0408-2214'
model_name = 'model.keras'

model_path = os.path.join(model_dir, model_run, model_name)
maxlen = 128

data_path = os.path.join('test_sequences', 'c_major.txt')

def load_test_data():
    # read note data 
    with open(data_path, 'r') as f:
        data = f.read()
    data = data.split('\n')
    print("data", data)
    
    data = [ast.literal_eval(d) for d in data]
    
    features = []
    
    for i in range(len(data)):
        feature = []
        for j in range(len(data[i])):
            # compute relative pitch
            if j == 0:
                rel_pitch = 0
            else:
                rel_pitch = data[i][j][0] - data[i][j-1][0]
            
            # (note onset, note offset, relative pitch)
            feature.append(np.array([data[i][j][0], data[i][j][1], rel_pitch]))
            
        if len(feature) < maxlen:
            pad_length = maxlen - len(feature)
            seq_pad = [np.zeros(3)] * pad_length
            feature.extend(seq_pad)
            print("padded feature length", len(feature))
        
        print(len(feature))
        features.append(np.array(feature))
                
    features = np.array(features)
    return features

def load_training_data():
    df = pd.read_csv("fingering_dataset/mixed_features.csv", delimiter=";")
    
    # print("df head\n", df.head())
    
    sequences = df["sequence"]
    rel_pitch = df["rel_sequence"]
    fingerings = df["fingering"]

    # df data count
    n_samples = len(sequences)
    
    # formulate sequences
    combined_sequences = []
    
    for i in range(n_samples):
        seq_list = ast.literal_eval(sequences[i])
        seq_len = len(seq_list)
        rel_pitch_list = ast.literal_eval(rel_pitch[i])
        
        combiend_seq = [ np.array([seq_list[j][0], seq_list[j][1], rel_pitch_list[j]]) for j in range(seq_len) ]
        
        combined_sequences.append(np.array(combiend_seq))
    
    fingerings = [ast.literal_eval(fingering) for fingering in fingerings]
    
    # map class
    fingerings =[ [fing - 1 for fing in fingering] for fingering in fingerings]
    
    # print("sequences")
    # print(combined_sequences[0])
    # print("fingerings", fingerings[0])
    
    return combined_sequences, fingerings

# pad sequences to max length
def preprocess_sequences(sequences, fingerings, maxlen):
    padded_sequences = []
    padded_fingerings = []

    for seq, fingering in zip(sequences, fingerings):
        if seq.shape[0] < maxlen:
            pad_length = maxlen - seq.shape[0]
            
            seq_pad = np.vstack([np.zeros(3)] * pad_length)
            # print("seq_pad", seq_pad)
            # print("seq shape", seq.shape, "seq_pad shape", seq_pad.shape)
            
            padded_seq = np.concatenate((seq, seq_pad), axis=0)
            # convert to numpy array
            # padded_seq = np.array(padded_seq)
            
            fing_pad = np.zeros(pad_length)
            # print("fing_pad", fing_pad)
            padded_fingering = np.concatenate((fingering, fing_pad))
            # padded_fingering = np.array(padded_fingering)
        else:
            padded_seq = np.array(seq[:maxlen])
            padded_fingering = fingering[:maxlen]

        padded_sequences.append(padded_seq)
        padded_fingerings.append(padded_fingering)

    # converrt to numpy array
    padded_sequences = np.stack(padded_sequences, axis=0)
    padded_fingerings = np.stack(padded_fingerings, axis=0)
    
    return padded_sequences, padded_fingerings

def _load_model():
    saved_model_dir = os.path.join(model_dir, model_run, "model")
    model = tf.saved_model.load(saved_model_dir)
    # model = load_model(model_path)
    return model

def inference():
    features = load_test_data()
    # print("features", features)
    
    model = _load_model()
    print("model", model)
    
    print(features.shape)    # (256, 3)
    logits = model(features)
    print
    # logits = model.predict(features, batch_size=features.shape[0]) 
    # max index of logits
    fingering = np.argmax(logits, axis=-1)
    fingering = fingering + 1
    
    for i in range(min(5, len(fingering))):
        print("fingering", fingering[i])
        plt.plot(fingering[i])
        plt.show()
    
def evaluate():
    model = _load_model()
    
    # load entire dataset
    sequences, fingerings = load_training_data()
    padded_sequences, padded_fingerings = preprocess_sequences(sequences, fingerings, maxlen)
    
    print("padded_sequences", padded_sequences.shape)
    print("padded_fingerings", padded_fingerings.shape)
    
    # evaluate
    # loss, acc = model.evaluate(padded_sequences, padded_fingerings, batch_size=padded_fingerings.shape[0])
    
    # print("loss", loss)
    # print("acc", acc)
    
    for i in range(5):
        print("padded_fingerings", padded_fingerings[i])
        logits = model.predict(padded_sequences[i].reshape(1, -1, 3), batch_size=1)
        fingering = np.argmax(logits, axis=-1)
        fingering = fingering + 1
        print("fingering", fingering)
        
        plt.plot(padded_fingerings[i])
        plt.plot(fingering[0])
        plt.show()

if __name__ == '__main__':
    inference()
    # evaluate()