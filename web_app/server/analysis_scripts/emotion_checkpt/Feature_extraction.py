from pymusickit.key_finder import KeyFinder
import librosa
import numpy as np
import os
import math
import soundfile as sf

# loop over all the audio files in directory
ct = 0

mfcc_arr = []
max_len = 0
feature_arr = []
for file in os.listdir("Q4"):
    try:
        audio_path = os.path.join("./Q4", file)
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
        #save the feature
        np.save(os.path.join("./Q4_npy", file[:-4]), feature)
        # ft = np.load(os.path.join("./Q4_npy", file[:-4]+".npy"))
        # print(ft)

    except:
        print("Error in reading file")
        # remove the file
        print("Removing file: ", audio_path)
        os.remove(audio_path)










