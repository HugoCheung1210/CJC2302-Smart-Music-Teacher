import vamp
import librosa
import vampyhost
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)  

    def forward(self, x):
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device)  
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))  
        out = self.fc(out[:, -1, :])  
        prob = F.softmax(out, dim=1)  

        return out,prob

    
class CNN_BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(CNN_BiLSTM, self).__init__()
        # CNN layers
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(16, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)  

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Reshape for CNN (batch_size, input_size, sequence_length)
        batch_size, input_size, sequence_length = x.size()
        x = x.view(-1, input_size, sequence_length)  # Reshape to apply CNN independently to each feature sequence
        cnn_output = self.cnn(x)
        cnn_output = cnn_output.view(batch_size, sequence_length, -1)  # Reshape back to original size        
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device)  
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(cnn_output, (h0, c0))  
        out = self.fc(out[:, -1, :])  
        prob = F.softmax(out, dim=1)  

        return out,prob
    

def load_audio(audio_path):
    if sys.platform.startswith('win'):
        # Windows-specific code
        os.environ['VAMP_PATH'] = "./style_transfer/Vamp_Window"


    elif sys.platform.startswith('darwin'):
        # macOS-specific code
        os.environ['VAMP_PATH'] = "./style_transfer/Vamp_Mac"

    data, rate = librosa.load(audio_path)
    plugin_key = "nnls-chroma:nnls-chroma"

    parameters = {
        "useNNLS": 1.0,  # Toggle approximate transcription (NNLS)
        "chromanormalize": 0.0  # Chroma normalization
    }


    step_size = int(rate/10)

    results = vamp.process_audio(data, rate, plugin_key, output="chroma",block_size=8192, step_size=step_size, parameters=parameters)

    ct = 1 

    data = []
    # Retrieve the results
    for features in results:
        arr = np.array(features["values"])
        time = 0.1*ct
        data.append(np.insert(arr, 0, time))    
        ct+=1

    data_100 = []
    num_chunks = len(data)//100
    for i in range(num_chunks):
        data_100.append(data[i*100:(i+1)*100])

    return data_100



def style_classification(audio_path):
    features = load_audio(audio_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNN_BiLSTM(input_size=13,hidden_size=16,num_layers=2,num_classes=5).to(device)

    check_pt = "./style_transfer/classification_checkpt/trained_100_2.pt"

    model.load_state_dict(torch.load(check_pt, map_location=device))  
    with torch.no_grad():
        model.eval()
        # for i in range(len(features)):
        chromadata = torch.tensor(features, dtype=torch.float32).to(device)
        # chromadata = chromadata.unsqueeze(0)
        out, prob = model(chromadata)
        prediction = torch.argmax(prob, dim=1)
        unique, counts = torch.unique(prediction,return_counts=True)
        label = unique[counts.argmax()]

        label_map = {'romantic': 0, 'baroque': 1, 'modern': 2, 'addon': 3, 'classical': 4}
        label = list(label_map.keys())[list(label_map.values()).index(label)]
        if label == "addon":
            label = "modern"
        print(label)









# print(chromadata[0])