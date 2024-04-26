import keras
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import ast
import time
import datetime

from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Input, TimeDistributed, Masking, Bidirectional
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

class_to_idx = [0, 0, 1, 2, 3, 4]
idx_to_class = [1, 2, 3, 4, 5]

# load from mix_features.csv
def load_data():
    df = pd.read_csv("fingering_dataset/mixed_features.csv", delimiter=";")
    
    # print("df head\n", df.head())
    
    sequences = df["sequence"]
    rel_pitch = df["rel_sequence"]
    fingerings = df["fingering"]

    rel_pitch = [ast.literal_eval(rel) for rel in rel_pitch]
    fingerings = [ast.literal_eval(fingering) for fingering in fingerings]
    
    # map class
    fingerings = [[class_to_idx[fing] for fing in fingering] for fingering in fingerings]
    
    # print("sequences")
    # print(combined_sequences[0])
    # print("fingerings", fingerings[0])
    
    return sequences, rel_pitch, fingerings

# pad sequences to max length
def pad_sequences(sequences, fingerings, maxlen):
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

# sequence to sequence model architecture
def lstm_model(maxlen, num_classes):
    model = Sequential(name='LSTM_model')
    model.add(Input(shape=(maxlen, 1), name='Input_Layer'))
    
    # masking layer to omit paddings
    model.add(Masking(mask_value=0., name='Masking_Layer'))
    
    model.add(Bidirectional(LSTM(512, return_sequences=True), name='LSTM_1'))
    model.add(Dropout(0.2, name='Dropout_1'))
    
    model.add(Bidirectional(LSTM(512, return_sequences=True), name='LSTM_2'))
    model.add(Dropout(0.2, name='Dropout_2'))
    
    model.add(TimeDistributed(Dense(num_classes, activation='softmax'), name='Output_Layer'))
    
    
    
    return model

def inference():
    model_dir = "models/rel_seq_lstm/0410-0244"
    model = lstm_model(128, 5)
    model.load_weights(os.path.join(model_dir, "model.weights.h5"))
    
    data_id = 0
    seqs = [[2, 2, 2, 2, 2], \
           [0, -4, 4, -7, 7, -12, 1, 2, 2, -2, 2, 2, 1, -1, 1, 2, 2, -4, -3, 3, -8, 5, -9, -3, 2, -2, 3, -3, 2, -2, 3, -3, 2, -2, 7, -4, 2, -2, 4, -4, 2, -2, 4, -4, 2, -2, 7, -3, 1, -1, 3, -3, 1, -1, 3, -3, 1, -1, 6, 4, -5, 5, -9, 4, -5, -2, 2, 1, 2, -2, 2, 2, 1, -1, 1, 2, 2, -5, 1, 2, -2, 2, 2, 2, -2, 2, 1, 2, -2, 2, 2, -7, 2, 1, -1], \
           [0, -2, -2, 2, 2, 0, 0, -1, 0, 0, 0, 1, 0]]
    
    labels = [[1, 2, 3, 4, 5], \
        [5, 3, 5, 1, 5, 1, 2, 3, 4, 1, 2, 3, 4, 3, 1, 3, 4, 2, 1, 5, 1, 5, 3, 1, 2, 1, 3, 1, 2, 1, 3, 1, 2, 1, 3, 1, 2, 1, 3, 1, 2, 1, 3, 1, 2, 1, 3, 1, 2, 1, 3, 1, 2, 1, 3, 1, 2, 1, 3, 5, 2, 5, 1, 3, 2, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 4, 3, 4, 5, 1, 2, 3, 2, 1, 5, 5, 3], \
            [3, 2, 1, 2, 3, 3, 3, 2, 2, 2, 2, 3, 3]]
    
    label = labels[data_id]
    
    
    seq = seqs[data_id]
    pred = model.predict(np.array([seq]))
    
    # print argmax of pred
    fingering = np.argmax(pred, axis=-1)
    fingering = [ fing + 1 for fing in fingering ]
    print(fingering)
    
    
    label = label[:len(seq)]
    
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
    
    model_dir = "models/rel_seq_lstm"
    
    # create run name using current date time
    run_name = datetime.datetime.now().strftime("%m%d-%H%M")
    model_dir = os.path.join(model_dir, run_name)
    os.makedirs(model_dir, exist_ok=True)
    
    epochs = 100
    maxlen = 128
    num_features = 128  # number of midi notes
    num_classes = 5
    batch_size = 32
    
    sequences, fingerings = load_data()
    # print max sequence length
    
    # print("sequences", sequences[0])
    # print("fingerings", fingerings[0])
    
    padded_sequences, padded_fingerings = preprocess_sequences(sequences, fingerings, maxlen)
    
    # print("padded_sequences", padded_sequences[0])
    # print("padded_fingerings", padded_fingerings[0])
    
    
    X_train, X_test, y_train, y_test = train_test_split(
        padded_sequences, padded_fingerings, test_size=0.2
    )
    # print("x_train shape", X_train.shape, "sample", X_train[0][:10])
    # print("y_train shape", y_train.shape, "sample", y_train[0][:10])
    # print("x_test shape", X_test.shape, "sample", X_test[0][:10])
    # print("y_test shape", y_test.shape, "sample", y_test[0][:10])
    
    # print the four shapes
    print("X_train shape", X_train.shape)
    print("y_train shape", y_train.shape)
    print("X_test shape", X_test.shape)
    print("y_test shape", y_test.shape)
    
    print("train test split done")
    
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
