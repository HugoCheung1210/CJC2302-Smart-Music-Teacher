# import crepe
import copy
import librosa
import librosa.display
import mido 
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy 
import math
import pandas as pd

from mido import MidiFile
from music21 import stream, note, chord, converter
from PIL import Image
from scipy.signal import find_peaks

from utils import read_midi_note_time, note_time2score, note_time2midi

class AudioAnalyser():
    def __init__(self, audio_path, midi_path, project_path, musescore_path):
        self.audio_path = audio_path
        self.midi_path = midi_path
        self.project_path = project_path
        self.musescore_path = musescore_path
        
        self.output_graphs = {}
    
    ### Tempo Analysis ###
    def tempo_analysis(self, output_path):
        X, sr = librosa.load(self.audio_path)

        # X_t, index = librosa.effects.trim(X, top_db= 10)


        onset_env = librosa.onset.onset_strength(y=X, sr=sr)
        pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sr)

        beats_plp = np.flatnonzero(librosa.util.localmax(pulse))

        tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env)
        beats_plp = np.flatnonzero(librosa.util.localmax(pulse))
        times = librosa.times_like(onset_env, sr=sr)


        times = librosa.times_like(pulse, sr=sr)
        
        plt.figure(figsize=(18,6))
        plt.plot(times, librosa.util.normalize(pulse),
            label='PLP')
        plt.vlines(times[beats_plp], 0, 1, alpha=0.5, color='r',
                linestyle='--', label='PLP Beats')
        plt.legend()
        plt.ylabel('PLP')
        plt.xlabel('Time (s)')
    #     plt.xaxis.set_major_formatter(librosa.display.TimeFormatter())
    #     plt.show()

        self.output_graphs["PLP"] = {
            "xAxis": times.tolist(), 
            "yAxis": librosa.util.normalize(pulse).tolist(),
            "verticalLines": times[beats_plp].tolist()
        }
    
        plt.savefig(output_path, bbox_inches='tight')

    ### Parse MIDI ### 
    def read_midi_note_time(self):
        mid = MidiFile(self.midi_path)

        note_time=[]
        time= 0
        note_unit =[]

        for msg in mid:
            # print(msg)
            note_unit_time = "%.2f" % time
            if msg.type == 'set_tempo':
                tempo = mido.tempo2bpm(msg.tempo)
                # Calculate the time duration of a quarter note
                quarter_note_duration = float(f"%.2f" % (60 / tempo))

            if msg.type == "note_on":
                time+=msg.time
                
                duration = float(f"%.2f" % msg.time)
                if duration / quarter_note_duration == 0.25:
                    time_symbol = 0.25
                elif duration / quarter_note_duration == 0.5:
                    time_symbol = 0.5
                elif duration / quarter_note_duration == 1:
                    time_symbol = 1
                elif duration / quarter_note_duration == 2:
                    time_symbol = 2
                elif duration / quarter_note_duration == 4:
                    time_symbol = 4

                if msg.velocity !=0:
                    note_unit.append((librosa.midi_to_note(msg.note)))
                # if msg.velocity ==0:
                #     note_unit.append((librosa.midi_to_note(msg.note)))
            if note_unit_time != "%.2f" % time:
                note_time.append((note_unit,"%.2f" % time, time_symbol))
                note_unit =[]

        return note_time
    
    def onset_detection(self, output_path):
        y, sr = librosa.load(self.audio_path)
        y_t, index = librosa.effects.trim(y, top_db= 20)
        
        self.music_start = index[0]/sr
        self.music_end = index[1]/sr
        
        # note_time = self.read_midi_note_time()
        note_time = read_midi_note_time(self.midi_path)

        o_env = librosa.onset.onset_strength(y=y, sr=sr)
        times = librosa.times_like(o_env, sr=sr)
        for i in range(0, 300, 1):
                i = i/300
                onset = librosa.onset.onset_detect(y=y_t, sr=sr, units='time', delta = i)
                # onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr, delta = i)
                if len(onset) < len(note_time):
                       break
        
        new_onset =[]
        for i in range(len(onset)):
               new_onset.append(onset[i]+index[0]/sr)

        D = np.abs(librosa.stft(y))
        fig, ax = plt.subplots(nrows=2, sharex=True)
        librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
                                x_axis='time', y_axis='log', ax=ax[0])
        ax[0].set(title='Power spectrogram')
        ax[0].label_outer()
        ax[1].plot(times, o_env, label='Onset strength')
        ax[1].vlines(new_onset, 0, o_env.max(), color='r', alpha=0.9,
                linestyle='--', label='Onsets')
        ax[1].set_ylabel('Onset strength')
        ax[1].set_xlabel('Time (s)')
        ax[1].legend()
        # plt.show()
        
        # print(f"new onset: {new_onset}")
        # print(f"time shape: {times.shape}")
        self.output_graphs["onset"] = {
            "xAxis": times.tolist(),
            "yAxis": o_env.tolist(),
            "verticalLines": new_onset,
        }
        
        plt.savefig(output_path, bbox_inches='tight')
        
        return new_onset
    
    def local_tempo_analysis(self, audio_path):
        y, sr = librosa.load(audio_path)
        y, index = librosa.effects.trim(y)
        
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        
        global_tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env)
        tempo_arr =[]
        timestamps = []
        
        # find the time of audio in second
        times = y.shape[0] / sr
        frame_duration = 6
        frame_size = 0.5    # frame every 0.5s
        n_frames = int(times/frame_size)
        print("Number of frames: ", n_frames-1)
        print("Global Tempo: ", global_tempo)
        
        for i in range(0, n_frames):
            s = int(i*sr*frame_size)
            e = s + frame_duration*sr
            onset_env = librosa.onset.onset_strength(y=y[s:e], sr=sr)
            tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env)
            # print("Tempo: " +  str(tempo) + " at " + str((i+1)*frame_size) + " seconds")
            
            tempo_arr.append(tempo)
            timestamps.append((i+1)*frame_size)
        
        self.output_graphs["tempo"] = {
            "xAxis": timestamps,
            "yAxis": tempo_arr,
        }
        # print("Tempo Array: ", tempo_arr)
        return tempo_arr
    
    ### Dynamics Analysis ###
    def dynamics_changes(self, second):
        cur_amp = second[0]
        threshold = np.array(second).mean()/4

        trend = "asc"

        dyn_changes = []
        for i in range(len(second)):
            if trend == "asc":
                if cur_amp - second[i] < threshold:
                    # print("Dscending")
                    trend = "dsc"
                    dyn_changes.append((i,trend))

                else:
                    continue

            elif trend == "dsc":
                if second[i] - cur_amp > threshold:
                    # print("Ascending")
                    trend = "asc"
                    dyn_changes.append((i,trend))


                else:
                    continue
            cur_amp = second[i]
        return dyn_changes

    def dynamics_analysis(self, output_path):
        y, sr = librosa.load(self.audio_path)

        second = []
        second_mean =[]
        half_second = int(sr/2)
        for s in range(0,len(y),sr):
            second.append( np.abs(y[s:s+half_second]).max())
            second_mean.append( np.abs(y[s:s+half_second]).mean())

        changes = self.dynamics_changes(second)

        peak, prop = scipy.signal.find_peaks(second)
        # prepend 0 to the peak array
        peak = np.insert(peak, 0, 0)
        plt.figure(figsize=(15,6))
        plt.plot(second, label = "Amplitude")
        plt.plot(peak, np.array(second)[peak], label = "General changes")
        plt.legend()
        plt.ylabel("Amplitude")
        plt.xlabel("Time")
        plt.title("Dynamic Change")
        # plt.show()
        
        self.output_graphs["dynamics"] = {
            "xAxis": np.arange(len(second)).tolist(),
            "yAxis": second        
        }        
        
        self.output_graphs["generalDynamics"] = {
            "xAxis": peak.tolist(),
            "yAxis": np.array(second)[peak].tolist()
        }
        
        plt.savefig(output_path)

    ### Pitch Detection ###
    '''def monophonic_pitch_detection(self):
        sr, audio = wavfile.read(self.audio_path)

        # Step size is 10 ms
        time, frequency, confidence, activation = crepe.predict(audio, sr, viterbi=True, step_size=100)

        # Extract only the notes with confidence > 0.7
        pitch = frequency[np.where(confidence >0.7)]

        # Convert the frequency to note
        note = librosa.hz_to_note(pitch)
        ct = 0
        detected_note_time = []

        for i in range(len(time)):
            if confidence[i] >0.7:
                print("Time: "+ str(time[i]) + " Note: "+ str(note[ct]))
                detected_note_time.append((note[ct],time[i]))
                ct+=1
        return detected_note_time
        '''
    
    
    def polyphonic_pitch_detection(self, onset_note, window_duration):
        # Load the audio file
        y, sr = librosa.load(self.audio_path)

        # Calculate the duration of each spectrogram window (in seconds)
        # Calculate the number of samples per window
        samples_per_window = int(sr * window_duration/2)


        # Calculate the number of windows
        # num_windows = len(y) // samples_per_window

        # Compute the spectrogram for each window
        # significant_frequencies = []

        detected_notes_time = []

        # note_time = self.read_midi_note_time()
        note_time = read_midi_note_time(self.midi_path)

        S = np.abs(librosa.stft(y))**2

        pitches, magnitudes = librosa.piptrack(S = S, sr=sr,
                                                    fmin=librosa.note_to_hz("C1"), fmax=librosa.note_to_hz("C7"))
        
        # print(magnitudes[np.nonzero(magnitudes)].max())

        max_val = math.ceil(magnitudes[np.nonzero(magnitudes)].max())
        step_size = int(max_val / 10)

        # print("Max Val: " + str(max_val))
        # print("Step Size: " + str(step_size))   
        for threshold in range(max_val,0-step_size, -step_size):
            detected_notes_time = []

            for i in range(len(onset_note)):
                # Extract the current window
                window_start = int(onset_note[i]*sr-samples_per_window)
                window_end = int(onset_note[i]*sr+samples_per_window)
                window = y[window_start:window_end]

                # Compute the spectrogram of the window
                D = np.abs(librosa.stft(window))**2
                S = librosa.feature.melspectrogram(S=D, sr=sr)

                pitches, magnitudes = librosa.piptrack(S = D, sr=sr,
                                                    fmin=librosa.note_to_hz("C1"), fmax=librosa.note_to_hz("C7"))
                # print(magnitudes[np.nonzero(magnitudes)])

                pitches = pitches[np.nonzero(pitches)]
                #filter out pitches with low magnitudes
                pitches = pitches[magnitudes[np.nonzero(magnitudes)] > threshold]
            
                f0_new = pitches[~np.isnan(pitches)]
                note = []
                power =[]
                if len(f0_new) ==0:
                    note.append("NULL")
                else:
                    note.append(librosa.hz_to_note(f0_new))
                    power.append(magnitudes)

                note = np.unique(note)
                # print("Second: " + str(i*window_duration) + "  Pitches: " + str(note))
                if "NULL" in note:
                    continue
                detected_notes_time.append((note,onset_note[i]))

            # print("Threshold: " + str(threshold) + "  Detected Notes: " + str(len(detected_notes_time)))
            if len(note_time) > len(detected_notes_time):
                continue
            else:
                return detected_notes_time
            
        if len(note_time) > len(detected_notes_time):
            for i in range(len(note_time)-len(detected_notes_time)):
                detected_notes_time.append((["NULL"],0))
            return detected_notes_time

    
    
    ### Accuracy ### 
    def check_accuracy(self, detected_note, onset_time, vision_data, window_duration, output_txt_path, output_overall_txt_path, output_csv_path):
        note_time = read_midi_note_time(self.midi_path)
        # note_time = self.read_midi_note_time()
                
        # Delete the onset time before the first detected note
        detected = []
        dum = True
        while dum:
            for i in range(len(onset_time)):
                if onset_time[i] < (detected_note[0][1]-0.1):
                    onset_time = np.delete(onset_time,i)
                    break
                else:
                    dum = False

        ct = 0
        print(f"length of note time: {len(note_time)}")
        print(f"length of detected_note: {len(detected_note)}")
       
        # Detect Notes
        for i in detected_note:
            for j in range(len(onset_time)):
                if i[1]+window_duration > onset_time[j]:
                    # Find vision data contain what notes
                    for k in vision_data:
                        if float(k[0]) > onset_time[j]:
                            filter_notes = k[1]
                            break
                    # Filter away the notes that are not in the vision data
                    correct_note = []
                    for note in i[0]:
                        if note in filter_notes:
                            correct_note.append(note)
                    

                    # Add the note that contain octave error
                    for note in i[0]:
                        if note not in correct_note:
                            if (len(note_time) < ct):
                                print(f"ct: {ct} len(note_time): {len(note_time)}")
                                break 
                            for filter_note in note_time[ct][0]:
                                if note[:-1] == filter_note[:-1]:
                                    if filter_note not in correct_note:
                                        correct_note.append(filter_note)

                    ct+=1
                    detected.append((correct_note, onset_time[j]))
                    onset_time = np.delete(onset_time,j)
                    break
                
        new_detected = copy.deepcopy(detected)
        for i in range(1,len(detected)):
            for note in detected[i][0]:
                if note in detected[i-1][0] and note not in note_time[i][0]:
                    new_detected[i][0].remove(note)

        for i in range(len(new_detected)):
            for j in range(len(new_detected[i][0])):
                new_detected[i][0][j] = new_detected[i][0][j].replace("♯", "#")
        
        with open(output_txt_path, "w") as f:
            for i in range(len(new_detected)):                
                f.write(str(new_detected[i][0]) + " " + str(new_detected[i][1]) + "\n")
                
        # for i in range(len(new_detected)):
        #     print("Detected notes: " + str(new_detected[i][0]) + " MIDI notes: " + str(note_time[i][0]) + " Detected time: " + str(new_detected[i][1]))

        # Count accuracy
        wrong_ct = 0
        note_ct = 0
        wrong_note = []
        for i in range(ct):
            note_ct+= len(note_time[i][0])
            for midi_note in note_time[i][0]:
                if midi_note not in new_detected[i][0]:
                    midi_note = midi_note.replace("♯", "#")
                    wrong_note.append((midi_note, np.round(new_detected[i][1],1)))
                    wrong_ct+=1
        acc = (note_ct-wrong_ct)/note_ct*100
        print("Ratio of correct note: " + str(f"%d" % (note_ct-wrong_ct))+ "/" + str(f"%d" % note_ct) + " (" + str(f"%.2f" % acc)+"%)")

        self.pitch_accuracy = acc

        # OVERALL OUTPUT
        with open(output_overall_txt_path, "w") as f:
            f.write("Ratio of correct note: " + str(f"%d" % (note_ct-wrong_ct))+ "/" + str(f"%d" % note_ct) + " (" + str(f"%.2f" % acc)+"%)\n")
            
            f.write(str(wrong_note))
        
        pitch = []
        # correct = 0, detected but not in MIDI = 1, not detected = 2
        class_list = []
        time_stamp = []
        midi_nums =[]
        for i in range(ct):
            for midi_note in note_time[i][0]:
                if midi_note in new_detected[i][0]:
                    if "♯" in midi_note:
                        midi_note = midi_note.replace("♯", "#")
                    # Change pitch into midi number
                    midi_num = librosa.note_to_midi(midi_note)
                    midi_nums.append(midi_num) 
                    pitch.append(midi_note)
                    class_list.append(0)
                    time_stamp.append(np.round(new_detected[i][1],1))
                elif midi_note not in new_detected[i][0]:
                    if "♯" in midi_note:
                        midi_note = midi_note.replace("♯", "#")
                    midi_num = librosa.note_to_midi(midi_note)
                    midi_nums.append(midi_num) 
                    pitch.append(midi_note)
                    class_list.append(2)
                    time_stamp.append(np.round(new_detected[i][1],1))
            for detected_note in new_detected[i][0]:
                if detected_note not in note_time[i][0]:
                    if "♯" in detected_note:
                        detected_note = detected_note.replace("♯", "#")
                    midi_num = librosa.note_to_midi(detected_note)
                    midi_nums.append(midi_num) 
                    pitch.append(detected_note)
                    class_list.append(1)
                    time_stamp.append(np.round(new_detected[i][1],1))
        
        df = pd.DataFrame({'Time': time_stamp, 'Pitch': pitch, 'Midi Number': midi_nums,'Class': class_list})
        df.columns = ['Time', 'Pitch', 'Midi Number', 'Class']

        df.to_csv(output_csv_path,index=False)
            
        return new_detected

    def read_vision_data(self, path):
        vision_data = []
        with open(path, 'r') as f:
            data = f.readlines()

            for line in data:
                line = line.replace('(','').replace(')','')
                line = line.replace('\n','').split(',',1)
                vision_data.append(np.array(line))

            
            f.close()
        
        return vision_data


    def run(self, audio_type="poly", window_duration=0.2):
        self.onset_time = self.onset_detection(os.path.join(self.project_path, "onset_detection.png"))
                    
        # if audio_type == "mono":
        #     self.detected_note = self.monophonic_pitch_detection()
        # else:
        self.detected_note = self.polyphonic_pitch_detection(self.onset_time, window_duration)
            
        self.vision_data = self.read_vision_data(os.path.join(self.project_path,  "hand_detection.txt"))
        
        self.note_time_data = self.check_accuracy(self.detected_note, self.onset_time, self.vision_data, window_duration, os.path.join(self.project_path, "pitch_detection.txt"), os.path.join(self.project_path, "overall_accuracy.txt"), os.path.join(self.project_path, "note_info.csv"))
        
        # self.note_time2score(self.note_time_data, os.path.join(self.project_path, "output_score.xml"), os.path.join(self.project_path, "output_score.png"))
        self.piece_note_time = read_midi_note_time(self.midi_path)
        note_time2midi(self.note_time_data, os.path.join(self.project_path, "output.mid"), piece_note_time=self.piece_note_time)
        self.tempo_analysis(os.path.join(self.project_path, "tempo_analysis.png"))
        self.local_tempo_analysis(self.audio_path)
        
        self.dynamics_analysis(os.path.join(self.project_path, "dynamics_analysis.png"))
        
if __name__ == "__main__":
    project_name = "elise_band"
    project_path = f"E:\CUHK\FYP\code\\results\{project_name}\\"
    midi_path = "E:\CUHK\FYP\code\\videos\pieces\elise.mid"
    analyser = AudioAnalyser(audio_path=project_path + "audio.wav", midi_path=midi_path, project_path=project_path)
    analyser.run(window_duration=0.3)
    