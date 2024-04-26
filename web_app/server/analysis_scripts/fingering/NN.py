import keras
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import ast
import time
import datetime

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

    return sequences, rel_pitch, fingerings

def preprocess_sequences(sequences, rel_pitch, fingerings, maxlen):
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
    fingerings = [[class_to_idx[fing] for fing in fingering] for fingering in fingerings]
    
    # print("sequences")
    # print(combined_sequences[0])
    # print("fingerings", fingerings[0])
    
    return combined_sequences, fingerings 

# pad sequences to max length
def pad_sequences(sequences, fingerings, maxlen):
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

# sequence to sequence model architecture
def lstm_model(maxlen, num_classes):
    model = Sequential(name='LSTM_model')
    model.add(Input(shape=(maxlen, 3), name='Input_Layer'))
    
    # masking layer to omit paddings
    model.add(Masking(mask_value=0., name='Masking_Layer'))
    
    model.add(Bidirectional(LSTM(1024, return_sequences=True), name='LSTM_1'))
    model.add(Dropout(0.2, name='Dropout_1'))
    # model.add(Bidirectional(LSTM(1024, return_sequences=True), name='LSTM_2'))
    # model.add(Dropout(0.2, name='Dropout_2'))
    model.add(Bidirectional(LSTM(512, return_sequences=True), name='LSTM_3'))
    model.add(Dropout(0.2, name='Dropout_3'))
    
    model.add(TimeDistributed(Dense(256, activation='relu'), name='Dense_1'))
    model.add(TimeDistributed(Dense(num_classes, activation='softmax'), name='Output_Layer'))
    
    # learning rate decay
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-2,
        decay_steps=10000,
        decay_rate=0.9
    )
    
    model.compile(
        loss="sparse_categorical_crossentropy", 
        # optimizer=Adam(learning_rate=lr_schedule), 
        optimizer=Adam(),
        metrics=["accuracy"]
    )
    
    model.build(input_shape=(None, maxlen, 3))
    
    return model


if __name__ == "__main__":
    model_dir = "models/lstmv3"
    
    # create run name using current date time
    run_name = datetime.datetime.now().strftime("%m%d-%H%M")
    model_dir = os.path.join(model_dir, run_name)
    os.makedirs(model_dir, exist_ok=True)
    
    epochs = 1
    maxlen = 128
    num_features = 128  # number of midi notes
    num_classes = 5
    batch_size = 32
    
    sequences, fingerings = load_data()
    
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
    
    print("model summary", lstm_model.summary())
    
    # print model summary to model_dir
    with open(os.path.join(model_dir, "model_summary.txt"), "w") as f:
        f.write(str(lstm_model.summary()))
    
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
    tf.saved_model.save(lstm_model, os.path.join(model_dir, "model"))
    
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
