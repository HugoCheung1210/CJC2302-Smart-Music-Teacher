import os 
import argparse
from moviepy.editor import AudioFileClip
import matplotlib.pyplot as plt
import xgboost as xgb
from pymusickit.key_finder import KeyFinder
import librosa
import numpy as np
import os

## dummy interface script for emotion analysis

def extract_feature(audio_path):    
    y_org, sr = librosa.load(audio_path)
    y, index = librosa.effects.trim(y_org)

    # Find the key of the song
    song = KeyFinder(audio_path)

    key = song.key_primary

    scale = key[-5:]
    pitch = key[:-6]

    # map key into number
    key_map = {
        'C': 0,
        'C#': 1,
        'D': 2,
        'D#': 3,
        'E': 4,
        'F': 5,
        'F#': 6,
        'G': 7,
        'G#': 8,
        'A': 9,
        'A#': 10,
        'B': 11
    }

    if scale == "minor":
        scale_num = 0
    else:
        scale_num = 1

    key_num = key_map[pitch]



    # Find mean and variance of the tempo
    
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)

    global_tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env)

    dtempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr,
                                aggregate=None)
    tempo_mean = global_tempo
    tempo_var = np.var(dtempo)

    # Find mean and variance of the amplitude in db
    S = np.abs(librosa.stft(y_org))
    db = librosa.power_to_db(S**2)

    amp_mean = np.mean(db)
    amp_var = np.var(db)
    
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_stft_mean = np.mean(chroma_stft)
    chroma_stft_var = np.var(chroma_stft)

    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    mfcc_mean = np.mean(mfcc)
    mfcc_var = np.var(mfcc)

    rms = librosa.feature.rms(y=y)
    rms_mean = np.mean(rms)
    rms_var = np.var(rms)

    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    cent_mean = np.mean(cent)
    cent_var = np.var(cent)

    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))


    har_y = librosa.effects.harmonic(y)
    tonnetz = np.mean(librosa.feature.tonnetz(y=har_y, sr=sr))


    feature = np.array([chroma_stft_mean, chroma_stft_var, mfcc_mean, mfcc_var, spectral_contrast, 
                        zcr, cent_mean, cent_var, rms_mean, rms_var, tonnetz, key_num, scale_num, 
                        tempo_mean, tempo_var, amp_mean, amp_var])
    np.save(os.path.join("./", audio_path[:-4]), feature)

    return feature


def predict_emotion(data, model_path, output_path):

    model = xgb.XGBClassifier(
    learning_rate =0.01,
    n_estimators=1000,
    objective="multi:softprob")


    model.load_model(model_path)


    prob_list = []
    data = np.array(data).reshape(1, -1)
    # Predict 100 times
    for i in range(100):
        np.random.seed(i)
        prediction = model.predict_proba(data)[0]
        new_prediction = prediction + np.random.exponential(scale=0.05, size=prediction.shape)
        # Find predicted label
        
        prob_list.append(new_prediction)



    x_s = []
    y_s = []
    weight = [1,1,1,1]
    ct = 0
    for prob in prob_list:

        x = 100*prob[0]* weight[0] - 100*prob[1]* weight[1] - 100*prob[2]* weight[2] + 100*prob[3]* weight[3]
        x_s.append(x)
        y = 100*prob[0]* weight[0] + 100*prob[1]* weight[1] - 100*prob[2]* weight[2] - 100*prob[3]* weight[3]
        y_s.append(y)
        ct += 1
        weight = [1,1,1,1]


    plt.scatter(x_s, y_s, marker='o')
    plt.xticks([])  # Remove x-axis ticks
    plt.yticks([])  # Remove y-axis ticks

    # adding vertical line in data co-ordinates 
    plt.axvline(0, c='black', ls='-') 
    
    # adding horizontal line in data co-ordinates 
    plt.axhline(0, c='black', ls='-') 

    plt.xlabel('Valence')
    plt.ylabel('Arousal')
    plt.xlim([-100, 100])
    plt.ylim([-100, 100])

    plt.title('Valence Arousal Graph')
    plt.savefig(output_path)



def main():
    parser = argparse.ArgumentParser(description="predict emotion from wav file")
    # add --file argument
    parser.add_argument("--dir", type=str, help="directory to analyze")
    # parse the arguments
    args = parser.parse_args()
    
    print("args", args)
    file_name = os.listdir(args.dir)[0]
    filepath = os.path.join(args.dir, file_name)
    # if file type is video, extract audio
    if filepath.endswith(".mp4"):
        audio_path = filepath.replace(".mp4", ".wav")
        audio = AudioFileClip(filepath)
        audio.write_audiofile(audio_path)
        
        os.remove(filepath)
    else:
        audio_path = filepath
    
    print("audio path", audio_path)
    
    plot_path = os.path.join(args.dir, "emotion_plot.png")
    # process and output stuff 
    data = extract_feature(audio_path)
    model_path = "./emotion_checkpt/emotion_3.json"
    predict_emotion(data, model_path, plot_path)
    
if __name__ == "__main__":
    main()