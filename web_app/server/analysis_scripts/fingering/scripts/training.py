import keras 
import numpy as np
import pandas as pd
from BiLSTM import BiLSTM
from DataProcessor import DataProcessor

import os
import ast 
import datetime
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class_to_idx = [0, 0, 1, 2, 3, 4]
idx_to_class = [1, 2, 3, 4, 5]

epochs = 80
maxlen = 128
num_classes = 5
batch_size = 32
    
# load from mix_features.csv
def load_data():
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

        combiend_seq = [
            np.array([seq_list[j][0], seq_list[j][1], rel_pitch_list[j]])
            for j in range(seq_len)
        ]

        combined_sequences.append(np.array(combiend_seq))

    fingerings = [ast.literal_eval(fingering) for fingering in fingerings]

    # map class
    fingerings = [
        [class_to_idx[fing] for fing in fingering] for fingering in fingerings
    ]

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

def plot_history(hist):
    plt.plot(hist.history["loss"], label="train_loss")
    plt.plot(hist.history["val_loss"], label="val_loss")
    plt.plot(hist.history["accuracy"], label="train_acc")
    plt.plot(hist.history["val_accuracy"], label="val_acc")
    plt.legend()
    plt.show()

def main():
    model_dir = "models/bilstm"

    # create run name using current date time
    run_name = datetime.datetime.now().strftime("%m%d-%H%M")
    model_dir = os.path.join(model_dir, run_name)
    os.makedirs(model_dir, exist_ok=True)

    # load data
    sequences, fingerings = load_data()

    # preprocess sequences
    padded_sequences, padded_fingerings = preprocess_sequences(sequences, fingerings, maxlen)

    X_train, X_test, y_train, y_test = train_test_split(
        padded_sequences, padded_fingerings, test_size=0.2
    )
    
    # train model
    model = BiLSTM(maxlen, num_classes)

    # compile model
    model.compile(
        optimizer="adam", 
        loss="sparse_categorical_crossentropy", 
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
    hist = model.fit(
        X_train, y_train, 
        epochs=epochs, 
        batch_size=batch_size, 
        validation_data=(X_test, y_test)
    )

    # save model weights
    model.save_weights(os.path.join(model_dir, "model.weights.h5"))
    
    # plot history
    plot_history(hist)

if __name__ == "__main__":
    main()