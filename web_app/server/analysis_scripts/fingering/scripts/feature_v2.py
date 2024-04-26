import keras
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import ast
import time
import datetime
import joblib 

from sklearn.preprocessing import StandardScaler
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Input, TimeDistributed, Masking, Bidirectional
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

class_to_idx = [0, 0, 1, 2, 3, 4]
idx_to_class = [1, 2, 3, 4, 5]
num_features = 4

# load from mix_features_v2.csv, [[feature]], [fingerings]
def load_data():
    df = pd.read_csv("fingering_dataset/mixed_features_v2.csv", delimiter=";")

    time_diff = df["time_diff"]
    start_bw = df["start_bw"]
    end_bw = df["end_bw"]
    rel_pitch = df["rel_pitch"]
    fingerings = df["fingering"]

    rel_pitch = [ast.literal_eval(rel) for rel in rel_pitch]
    start_bw = [ast.literal_eval(start) for start in start_bw]
    end_bw = [ast.literal_eval(end) for end in end_bw]
    time_diff = [ast.literal_eval(time) for time in time_diff]
    fingerings = [ast.literal_eval(fingering) for fingering in fingerings]
    
    # map class
    fingerings = [[class_to_idx[fing] for fing in fingering] for fingering in fingerings]
    
    combined_sequences = []
    for i in range(len(rel_pitch)):
        seq = []
        for j in range(len(rel_pitch[i])):
            seq.append([time_diff[i][j], start_bw[i][j], end_bw[i][j], rel_pitch[i][j]])
        combined_sequences.append(seq)
    
    return combined_sequences, fingerings

# pad sequences to max length
def old_pad_sequences(sequences, fingerings, maxlen):
    padded_sequences = []
    padded_fingerings = []

    for seq, fingering in zip(sequences, fingerings):
        if len(seq) < maxlen:
            pad_length = maxlen - len(seq)
            
            seq_pad = np.zeros(pad_length)
            padded_seq = np.concatenate((seq, seq_pad))
            
            fing_pad = np.zeros(pad_length)
            padded_fingering = np.concatenate((fingering, fing_pad))
            # padded_fingering = np.array(padded_fingering)
        else:
            padded_seq = seq[:maxlen]
            padded_fingering = fingering[:maxlen]

        padded_sequences.append(padded_seq)
        padded_fingerings.append(padded_fingering)

    # converrt to numpy array
    padded_sequences = np.stack(padded_sequences, axis=0)
    padded_fingerings = np.stack(padded_fingerings, axis=0)
    
    return padded_sequences, padded_fingerings

# normalize and pad sequences
def preprocess_data(sequences, fingerings, model_dir, maxlen=128):
    # standardize rel_pitch from -12 to +12 to 0 to 1
    scaled_sequences = []
    for seq in sequences:
        scaled_seq = []
        for note in seq:
            scaled_note = [note[0], note[1], note[2], (note[3] + 12) / 24.0]
            scaled_seq.append(scaled_note)
        scaled_sequences.append(scaled_seq)
    
    # pad sequences
    padded_sequences = pad_sequences(scaled_sequences, padding='post', value=-1, dtype=float, maxlen=maxlen)
    padded_fingerings = pad_sequences(fingerings, padding='post', value=0, maxlen=maxlen)
    
    return padded_sequences, padded_fingerings

# sequence to sequence model architecture
def lstm_model(maxlen, num_classes):
    model = Sequential(name='LSTM_model')
    model.add(Input(shape=(maxlen, num_features), name='Input_Layer'))
    
    # masking layer to omit paddings
    model.add(Masking(mask_value=-1, name='Masking_Layer'))
    
    model.add(Bidirectional(LSTM(1024, return_sequences=True), name='LSTM_1'))
    model.add(Dropout(0.2, name='Dropout_1'))
    
    model.add(Bidirectional(LSTM(512, return_sequences=True), name='LSTM_2'))
    model.add(Dropout(0.2, name='Dropout_2'))
    
    model.add(TimeDistributed(Dense(128, activation='relu'), name='Dense_1'))  # added for v6
    model.add(TimeDistributed(Dense(num_classes, activation='softmax'), name='Output_Layer'))
    
    return model

def inference():
    model_dir = "models/lstm_v2/kaggle_v9"
    model = lstm_model(128, 5)
    model.load_weights(os.path.join(model_dir, "model.weights.h5"))
    
    data_id = 0
    seqs = [[[0.75, 0, 0, -48], [1.5, 0, 0, 5], [1.5, 0, 0, -5], [2.25, 0, 0, 5], [0.75, 0, 0, 0], [2.25, 0, 0, -4], [0.75, 0, 0, -1], [2.25, 0, 0, 5], [0.75, 0, 0, -5], [1.5, 0, 0, 5], [1.5, 0, 0, -5], [1.5, 0, 0, 5], [1.5, 0, 0, 0], [1.5, 0, 0, -4], [3.75, 0, 0, -1]], [[0.38, 0, 1, 70], [0.37, 1, 0, -10], [0.18999999999999995, 0, 0, 12], [1.12, 0, 1, -4], [0.18999999999999995, 1, 0, -8], [0.18999999999999995, 0, 1, 3], [2.02, 1, 0, 1], [0.040000000000000036, 0, 0, 0], [1.42, 0, 1, 6], [0.08000000000000007, 1, 0, -6], [0.71, 0, 1, 6], [0.0, 1, 1, -2], [0.040000000000000036, 1, 0, -6], [0.71, 0, 1, 8], [0.040000000000000036, 1, 0, -8], [0.7100000000000009, 0, 1, 8], [1.5399999999999991, 1, 0, -6], [1.1300000000000008, 0, 0, 3], [0.7399999999999984, 0, 0, 0], [1.08, 0, 0, 5], [0.75, 0, 0, -10], [1.1300000000000008, 0, 0, 0], [0.5600000000000005, 0, 1, 1], [0.5599999999999987, 1, 0, 2], [1.240000000000002, 0, 1, 5], [0.8200000000000003, 1, 0, -10], [0.9399999999999977, 0, 0, 2], [0.7100000000000009, 0, 0, 9], [0.03999999999999915, 0, 0, -2], [0.379999999999999, 0, 0, -5], [0.370000000000001, 0, 0, -2], [0.7100000000000009, 0, 0, 5], [2.210000000000001, 0, 0, -7], [0.0, 0, 0, 5]],
            [[0.38, 0, 0, -53], [3.37, 0, 1, -5], [6.75, 1, 0, 5], [0.5600000000000005, 0, 0, 0], [1.2299999999999986, 0, 1, 2], [2.66, 1, 1, 0], [0.5600000000000005, 1, 1, -7], [6.000000000000002, 1, 0, 5], [0.18999999999999773, 0, 0, -4]],
            [[0.75, 0, 0, -12], [1.5, 0, 0, 5], [1.5, 0, 0, -5], [2.25, 0, 0, 5], [0.75, 0, 0, 0], [2.25, 0, 0, -4], [0.75, 0, 0, -1], [2.25, 0, 0, 5], [0.75, 0, 0, -5], [1.5, 0, 0, 5], [1.5, 0, 0, -5], [1.5, 0, 0, 5], [1.5, 0, 0, 0], [1.5, 0, 0, -4], [3.75, 0, 0, -1]], \
        [[0.75, 0, 0, 12], [0.75, 0, 0, -2], [0.75, 0, 0, -2], [0.75, 0, 0, 2], [0.75, 0, 0, 2], [0.75, 0, 0, 0], [1.5, 0, 0, 0], [0.75, 0, 0, -2], [0.75, 0, 0, 0], [1.5, 0, 0, 0], [0.75, 0, 0, 2], [0.75, 0, 0, 3], [1.5, 0, 0, 0], [0.75, 0, 0, -3], [0.75, 0, 0, -2], [0.75, 0, 0, -2], [0.75, 0, 0, 2], [0.75, 0, 0, 2], [0.75, 0, 0, 0], [0.75, 0, 0, 0], [0.75, 0, 0, 0], [0.75, 0, 0, -2], [0.75, 0, 0, 0], [0.75, 0, 0, 2], [0.75, 0, 0, -2], [3.0, 0, 0, -2]], \
            [[0.5, 0, 0, 2], [0.5, 0, 0, 2], [0.5, 0, 0, 2], [0.5, 0, 0, 2], [0.5, 0, 0, 2]], \
           [0, -4, 4, -7, 7, -12, 1, 2, 2, -2, 2, 2, 1, -1, 1, 2, 2, -4, -3, 3, -8, 5, -9, -3, 2, -2, 3, -3, 2, -2, 3, -3, 2, -2, 7, -4, 2, -2, 4, -4, 2, -2, 4, -4, 2, -2, 7, -3, 1, -1, 3, -3, 1, -1, 3, -3, 1, -1, 6, 4, -5, 5, -9, 4, -5, -2, 2, 1, 2, -2, 2, 2, 1, -1, 1, 2, 2, -5, 1, 2, -2, 2, 2, 2, -2, 2, 1, 2, -2, 2, 2, -7, 2, 1, -1], \
           [0, -2, -2, 2, 2, 0, 0, -1, 0, 0, 0, 1, 0]]
    
    labels = [[1, 2, 3, 4, 5], \
        [5, 3, 5, 1, 5, 1, 2, 3, 4, 1, 2, 3, 4, 3, 1, 3, 4, 2, 1, 5, 1, 5, 3, 1, 2, 1, 3, 1, 2, 1, 3, 1, 2, 1, 3, 1, 2, 1, 3, 1, 2, 1, 3, 1, 2, 1, 3, 1, 2, 1, 3, 1, 2, 1, 3, 1, 2, 1, 3, 5, 2, 5, 1, 3, 2, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 4, 3, 4, 5, 1, 2, 3, 2, 1, 5, 5, 3], \
            [3, 2, 1, 2, 3, 3, 3, 2, 2, 2, 2, 3, 3]]
    
    # sequences, fingerings = load_data()
    
    # label = fingerings[0]
    # seq = sequences[0]
    seq = seqs[0]
    label = labels[0]
    
    print("seq", seq)
    
    seq, label = preprocess_data([seq], [label], model_dir)
    
    # print("seq", seq)
    
    pred = model.predict(seq)
    
    # print argmax of pred
    fingering = np.argmax(pred, axis=-1)
    fingering = [ fing + 1 for fing in fingering ]
    print("seq len", len(seqs[0]))
    print(list(fingering[0][:len(seqs[0])]))
    return 
    label = label[0] + 1
    print(label)
    
    
    mismatch = [1 if l != p else 0 for l, p in zip(label, fingering[0])]
    print("accuracy", 1 - sum(mismatch) / len(mismatch))
    x_axis = range(0, len(seq))
    plt.plot(label, label="truth")
    plt.plot(fingering[0], label="predict")
    
    plt.legend(loc="upper left")
    plt.show()

if __name__ == "__main__":
    inference()
    exit()
    
    model_dir = "models/lstm_v2"
    
    # create run name using current date time
    run_name = datetime.datetime.now().strftime("%m%d-%H%M")
    model_dir = os.path.join(model_dir, run_name)
    os.makedirs(model_dir, exist_ok=True)
    
    epochs = 50
    num_classes = 5
    batch_size = 32
    
    sequences, fingerings = load_data()
    
    # preprocess data
    padded_sequences, padded_fingerings = preprocess_data(sequences, fingerings, model_dir)
    
    maxlen = padded_sequences.shape[1]
    
    print("padded_sequences", padded_sequences.shape)
    print("padded_fingerings", padded_fingerings.shape)
    
    # print before and after padding values
    # print("before padding", sequences[0][:10])
    # print("after padding", padded_sequences[0][:10])
    
    X_train, X_test, y_train, y_test = train_test_split(
        padded_sequences, padded_fingerings, test_size=0.2
    )
    
    print("x_train shape", X_train.shape)
    # print("y_train shape", y_train.shape, "sample", y_train[0][:10])
    # print("x_test shape", X_test.shape, "sample", X_test[0][:10])
    # print("y_test shape", y_test.shape, "sample", y_test[0][:10])
    
    # print the four shapes
    print("X_train shape", X_train.shape)
    print("y_train shape", y_train.shape)
    print("X_test shape", X_test.shape)
    print("y_test shape", y_test.shape)
    
    print("train test split done")
    
    # exit()
    lstm_model = lstm_model(maxlen, num_classes)
    print("lstm model built")
    
    # learning rate decay
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-2,
        decay_steps=10000,
        decay_rate=0.9
    )
    
    lstm_model.compile(
        loss="sparse_categorical_crossentropy", 
        # optimizer=Adam(learning_rate=lr_schedule), 
        optimizer=Adam(),
        metrics=["accuracy"]
    )
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            os.path.join(model_dir, "model.keras"),
            save_best_only=True,
            monitor="val_loss"
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5
        )
    ]
    
    # print datatype of X_train, y_train
    print("X_train dtype", X_train.dtype)
    print("y_train dtype", y_train.dtype)

    history = lstm_model.fit(
        X_train, y_train, 
        epochs=epochs, 
        batch_size=batch_size, 
        callbacks=callbacks,
        validation_data=(X_test, y_test)
    )
    
    # save model
    # lstm_model.save(os.path.join(model_dir, "model.hdf5"))
    lstm_model.save_weights(os.path.join(model_dir, "model.weights.h5"))
    
    # plot loss and accuracy
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    # savefig
    plt.savefig(os.path.join(model_dir, "loss.png"))
    plt.clf()
    
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    # savefig
    plt.savefig(os.path.join(model_dir, "accuracy.png"))
