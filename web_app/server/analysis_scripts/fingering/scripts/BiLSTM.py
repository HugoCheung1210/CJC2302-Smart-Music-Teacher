from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Input, TimeDistributed, Masking, Bidirectional

# sequence to sequence model architecture
def BiLSTM(maxlen, num_classes):
    model = Sequential(name='LSTM_model')
    model.add(Input(shape=(maxlen, 3), name='Input_Layer'))
    
    # masking layer to omit paddings
    model.add(Masking(mask_value=0., name='Masking_Layer'))
    
    model.add(Bidirectional(LSTM(1024, return_sequences=True), name='LSTM_1'))
    model.add(Dropout(0.2, name='Dropout_1'))
    model.add(Bidirectional(LSTM(1024, return_sequences=True), name='LSTM_2'))
    model.add(Dropout(0.2, name='Dropout_2'))
    model.add(Bidirectional(LSTM(512, return_sequences=True), name='LSTM_3'))
    model.add(Dropout(0.2, name='Dropout_3'))
    
    model.add(TimeDistributed(Dense(256, activation='relu'), name='Dense_1'))
    model.add(TimeDistributed(Dense(num_classes, activation='softmax'), name='Output_Layer'))
    
    return model