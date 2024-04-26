import argparse
import requests
import pandas as pd
import os
import subprocess
from moviepy.editor import AudioFileClip
import json
import scipy
import numpy as np
import librosa

from utils import *
from Piano import Piano
from Handtracker import HandTracker
from AudioAnalyser import AudioAnalyser
from FingeringAnalyser import FingeringAnalyser

class Analyser:
    def __init__(self):
        self.video_path = None
        self.background_path = None

        self.load_args()
        self.print_args()
        self.init_path()

        self.debug = False
        self.piano = None
        self.handtracker = None

    def load_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--dir", type=str, help="directory to analyze")
        parser.add_argument("--videoRotation", type=int, help="rotation of video")
        parser.add_argument("--backgroundTime", type=float, help="background time")
        parser.add_argument("--recordingId", type=int, help="recording id")
        parser.add_argument("--midiPath", type=str, help="midi filepath")
        parser.add_argument("--musescorePath", type=str, help="musescore path")

        self.args = parser.parse_args()
        self.dir = self.args.dir
        self.video_rotation = self.args.videoRotation
        self.background_time = self.args.backgroundTime
        self.recordingId = self.args.recordingId
        self.midi_path = self.args.midiPath
        self.musescore_path = self.args.musescorePath

    def print_args(self):
        # print all arguments
        print("directory", self.args.dir)
        print("rotation", self.args.videoRotation)
        print("background time", self.args.backgroundTime)
        print("recording id", self.args.recordingId)
        print("midi path", self.args.midiPath)

    def init_path(self):
        # setup video path
        for file in os.listdir(self.dir):
            if "raw_video" in file:
                self.video_path = os.path.join(self.dir, file)
                break

        # if video not mp4, change video to mp4
        if not self.video_path.endswith(".mp4"):
            mp4_path = self.video_path[:-4] + ".mp4"

            # convert to mp4 using ffmpeg
            command = ["ffmpeg", "-i", self.video_path, mp4_path]
            subprocess.run(command, check=True)

            os.remove(self.video_path)
            self.video_path = mp4_path

        self.background_path = os.path.join(self.dir, "background.jpg")

        self.audio_path = os.path.join(self.dir, "audio.wav")

    # rotate video and capture background
    def preprocess_video(self):
        # do rotation
        if self.video_rotation in [90, 180, 270]:
            transpose_cmds = {
                90: "transpose=1",
                180: "transpose=2,transpose=2",
                270: "transpose=2",
            }
            transpose_cmd = transpose_cmds[self.video_rotation]

            print("self.video_path", self.video_path)
            tmp_path = self.video_path[:-4] + "_tmp.mp4"

            # remove tmp file if exists
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

            # rename video file to avoid overwriting
            os.rename(self.video_path, tmp_path)

            # rotation using ffmpeg
            command = [
                "ffmpeg",
                "-i",
                tmp_path,
                "-vf",
                f"{transpose_cmd}",
                self.video_path,
            ]
            print("command", command)
            print("video path", self.video_path)
            print("tmp path", tmp_path)
            subprocess.run(command, check=True)

            # remove tmp file
            os.remove(tmp_path)

        # capture background
        frame = capture_background(self.video_path, self.background_time)
        # save frame to background.jpg
        cv2.imwrite(self.background_path, frame)

    def run_key_detection(self):
        self.piano = Piano(self.dir)
        self.piano.load_img(self.background_path, debug=self.debug)
        _ = self.piano.detect_keys(debug=self.debug)

        if not _:
            print("key detection failed, please reload project")
            return False

        return True

    def run_hand_tracking(self):
        self.handtracker = HandTracker(
            piano=self.piano, video_path=self.video_path, rot_deg=0
        )

        if self.debug:
            output_path = self.dir + "hand_detection.mp4"
            output_width = 1080
        else:
            output_path = None
            output_width = None

        self.handtracker.run(
            sample_interval=100,
            output_path=output_path,
            debug=self.debug,
            output_width=output_width,
        )
        print("hand detection done")

        # print as txt
        def export_hand_detection_result():
            filename = os.path.join(self.dir, "hand_detection.txt")
            with open(filename, "w") as f:
                for i in range(len(self.handtracker.record)):
                    f.write(str(self.handtracker.record[i]) + "\n")
            print("hand detection results exported")

        export_hand_detection_result()
        return True

    def postprocess_charts(self):
        # reduce chart size
        chart = self.audio_analyser.output_graphs

        # get audio length of self.audio_path
        y, sr = librosa.load(self.audio_path)
        audio_length = librosa.get_duration(y=y, sr=sr)
        time_count = int(audio_length * 10)

        for key in chart:
            print("key", key)
            print("xaxis len", len(chart[key]["xAxis"]))
            print("max time", max(chart[key]["xAxis"]))

            ori_xaxis = chart[key]["xAxis"]
            ori_yaxis = chart[key]["yAxis"]

            # round down ori_xaxis[0] to 0.1
            ori_xaxis[0] = np.floor(ori_xaxis[0] * 10) / 10
            # round up ori_xaxis[-1] to 0.1
            ori_xaxis[-1] = np.ceil(ori_xaxis[-1] * 10) / 10

            max_time = max(ori_xaxis)

            # convert sample rate to 0.1s
            new_xaxis = np.arange(ori_xaxis[0], max_time, 0.1)
            new_yaxis = scipy.interpolate.interp1d(ori_xaxis, ori_yaxis)(new_xaxis)

            print("new xaxis len", len(new_xaxis))
            print("new yaxis len", len(new_yaxis))

            # if new_xaxis is shorter than time_count, pad with 0
            if len(new_xaxis) < time_count:
                # pad front from 0 to new_xaxis[0]
                pad_length = int(new_xaxis[0] * 10)
                new_xaxis = np.append(np.arange(0, new_xaxis[0], 0.1), new_xaxis)
                new_yaxis = np.append(np.zeros(pad_length), new_yaxis)

                # pad end from new_xaxis[-1] to time_count
                pad_length = time_count - len(new_xaxis)
                new_xaxis = np.append(
                    new_xaxis,
                    np.arange(
                        max(new_xaxis) + 0.1,
                        max(new_xaxis) + 0.1 * (pad_length + 1),
                        0.1,
                    ),
                )
                new_yaxis = np.append(new_yaxis, np.zeros(pad_length))

            # if new_xaxis longer than time_count, trim to time_count
            if len(new_xaxis) > time_count:
                new_xaxis = new_xaxis[:time_count]
                new_yaxis = new_yaxis[:time_count]

            chart[key]["xAxis"] = new_xaxis.tolist()
            chart[key]["yAxis"] = new_yaxis.tolist()

        self.charts = chart

    def run_audio_analysis(self):
        self.audio_analyser = AudioAnalyser(
            audio_path=self.audio_path,
            midi_path=self.midi_path,
            project_path=self.dir,
            musescore_path=self.musescore_path,
        )
        self.audio_analyser.run()

    def gen_fingering(self, left_notes, right_notes):
        # finger id, time, midi number, index
        
        # [rel_time, prev_key_color, self_key_color, rel_pitch]
        left_seq = []
        right_seq = []
        
        prev_time = 0
        prev_key_color = 0
        prev_pitch = 0
        
        for note in left_notes:
            cur_key_color = note[2] % 12 in [1, 3, 6, 8, 10]
            left_seq.append([note[1]-prev_time, prev_key_color, cur_key_color, (note[2]-prev_pitch)*-1])
            
            prev_time = note[1]
            prev_pitch = note[2]
            prev_key_color = cur_key_color
        
        prev_time = 0
        prev_key_color = 0
        prev_pitch = 0
        for note in right_notes:
            cur_key_color = note[2] % 12 in [1, 3, 6, 8, 10]
            right_seq.append([note[1]-prev_time, prev_key_color, cur_key_color, note[2]-prev_pitch])
            
            prev_time = note[1]
            prev_pitch = note[2]
            prev_key_color = cur_key_color
        
        self.fingering_analyser = FingeringAnalyser()
        
        if len(left_seq) > 128:
            left_finger = [] 
            for i in range(0, len(left_seq), 128):
                left_finger += self.fingering_analyser.inference(left_seq[i:i+128]) 
        else:
            left_finger = self.fingering_analyser.inference(left_seq)

        if len(right_seq) > 128:
            right_finger = [] 
            for i in range(0, len(right_seq), 128):
                right_finger += self.fingering_analyser.inference(right_seq[i:i+128])
        else:
            right_finger = self.fingering_analyser.inference(right_seq)
        
        print("left seq: ", left_seq)
        print("gen left_finger: ", left_finger)
        
        print("right seq: ", right_seq)
        print("gen right_finger: ", right_finger)
        
        combined_finger = [None for _ in range(len(self.dummy_correct))]
        
        print("len(combined_finger): ", len(combined_finger))
        
        for i in range(len(left_notes)):
            # print("left_notes[i][3]: ", left_notes[i][3])
            # print("left_finger[i]: ", left_finger[i])
            combined_finger[left_notes[i][3]] = left_finger[i]
            
        for i in range(len(right_notes)):
            combined_finger[right_notes[i][3]] = right_finger[i]
        
        self.generated_fingering = combined_finger
        
    def run_fingering_analysis(self):
        # [timestamp, [[handedness, finger_id, on_note]]]
        key_finger_map = self.handtracker.key_finger_list
        with open(os.path.join(self.dir, "key_finger_map.txt"), "w") as f:
            for key_finger in key_finger_map:
                f.write(str(key_finger) + "\n")

        note_df = pd.read_csv(os.path.join(self.dir, "note_info.csv"))

        new_note_df = map_finger_to_key(note_df, key_finger_map)
        new_note_df.to_csv(os.path.join(self.dir, "note_info_finger.csv"), index=False)
        
        # extract fingering from df 
        # for each row 
        left_notes, right_notes = [], []
        fingering = []
        hand = []
        correct = []
        correct_dummy = []
        time = []
        note_ct = 0
        
        for index, row in new_note_df.iterrows():
            if row["Class"] == 2:
                continue
            
            if row["Class"] == 1:
                correct.append((False, row["Pitch"]))
            else:
                correct.append((True, row["Pitch"]))
            
            correct_dummy.append((True, row["Pitch"]))
            time.append(row["Time"])
            
            if row["finger_id"] == None or np.isnan(row["finger_id"]):
                fingering.append(None)
                hand.append(None)
                continue
            
            hand.append(row["hand"])
            fingering.append(int(row["finger_id"] + 1))
            
            if row["hand"] == "Left":
                left_notes.append([int(row["finger_id"] + 1), row["Time"], row["Midi Number"], note_ct])
            else:
                right_notes.append([int(row["finger_id"] + 1), row["Time"], row["Midi Number"], note_ct])
            
            note_ct += 1

        print(fingering)
        print(correct)
        
        # data for transcription
        self.detected_fingering = fingering
        self.detected_hand = hand
        self.detected_correct = correct
        
        # cost computation 
        
        # generate suggested fingering
        self.dummy_correct = correct_dummy
        self.gen_fingering(left_notes, right_notes)
        
    def compute_sequence_stability(self, seq, max_std=None):
        start_time = self.audio_analyser.music_start
        end_time = self.audio_analyser.music_end
        
        # trim to start_time, end_time
        start_idx = int(start_time * 10)
        end_idx = int(end_time * 10)
        
        trim_seq = seq[start_idx:end_idx]
        
        # max of trim_seq
        if not max_std:
            max_std = np.max(trim_seq)
        
        # inverse of standard deviation
        std = np.std(trim_seq)
        print(f"std: {std}, max_std: {max_std}")
        
        score = (max_std - std) / max_std * 100
        
        # trim to 0, 100
        score = max(0, score)
        score = min(100, score)
        
        return score
    
    def compute_score(self):
        score = {}
        score["pitch_acc"] = self.audio_analyser.pitch_accuracy
        
        chart = self.audio_analyser.output_graphs
        
        # generalDynamics
        general_dynamics = np.array(chart["generalDynamics"]["yAxis"])       
        score["dyn_cons"] = self.compute_sequence_stability(general_dynamics)
        
        # tempoStability
        tempo = np.array(chart["tempo"]["yAxis"])
        score["tempo_stab"] = self.compute_sequence_stability(tempo, max_std=80)
        
        score["overall"] = np.mean([score["pitch_acc"], score["dyn_cons"], score["tempo_stab"]])
        
        # TODO
        score["tempo_acc"] = 0
        score["dyn_range"] = 0
        score["finger"] = 80
        
        self.score = score
    
    def generate_comment(self):
        comment = {}
        overall_comments = [
            ["Work harder!", "You can do better!", "Keep practicing!", "You can improve!"],
            ["Good effort!", "You're getting there!", "Keep it up!", "You're on the right track!"],
            ["Well done!", "You're doing great!", "Good work!", "You're almost there!"],
            ["Excellent!", "Amazing job!", "You're doing amazing!", "You're a star!"],
            ["Perfect!", "You're a pro!", "You're a star!", "You're a genius!"]
        ]
        # randomly select a comment from the range
        score_range = int(min(4, self.score["overall"] // 20))
        comment["overall"] = overall_comments[score_range][int(np.random.randint(0, 4))]
        
        pitch_acc_comments = [
            ["Your pitch accuracy needs improvement. Focus on hitting the correct notes consistently.", "It seems you're struggling with pitch accuracy. Keep practicing and pay attention to the correct pitch of each note.", "There is room for improvement in your pitch accuracy. Work on listening closely and matching the pitches accurately."],
            ["Your pitch accuracy is showing some progress, but there is still work to be done. Keep practicing and strive for more precise pitch control.", "You're making efforts to improve pitch accuracy, but it still needs more attention. Practice scales and exercises to enhance your pitch recognition and control.", "Your pitch accuracy is improving, but be mindful of occasional inconsistencies. Focus on ear training and continue practicing challenging passages to refine your pitch precision."],
            ["Your pitch accuracy is good, and you're generally hitting the correct notes. Keep up the consistent effort and aim for even greater accuracy.", "You're doing well with pitch accuracy, showing good control over the notes. Keep practicing with attention to detail to maintain your accuracy consistently.", "Your pitch accuracy is getting solid, demonstrating improved control over the notes. Work on refining your intonation in challenging sections for a more polished performance."],
            ["Your pitch accuracy is great! You consistently hit the correct notes with precision. Maintain this level of accuracy and continue refining your pitch control.", "Excellent pitch accuracy! You have a strong command over the notes and maintain consistent intonation. Keep up the impressive work!", "Your pitch accuracy is impressive, resulting in a captivating performance. Maintain your exceptional precision and strive for even greater subtleties in pitch expression."],
            ["Your pitch accuracy is outstanding! You have a remarkable ability to hit every note precisely. Your attention to detail in pitch control sets you apart as a musician.", "Perfect pitch accuracy! Your performance showcases flawless note recognition and control. Continue to explore nuances in pitch expression to elevate your artistry.", "Your pitch accuracy is exceptional, demonstrating mastery of intonation. Your precise notes contribute to a truly professional and captivating performance."]
        ]
        score_range = int(min(4, self.score["pitch_acc"] // 20))
        comment["pitch"] = pitch_acc_comments[score_range][int(np.random.randint(0, 3))]
        
        tempo_stability_comments = [
            ["You need to keep your tempo more consistent!", "The tempo needs to be more stable!", "You need to work on your tempo stability!", "You need to keep your tempo more consistent!"],
            ["Your tempo stability is improving!", "You're getting better at tempo stability!", "You're making progress with tempo stability!", "You're getting there with tempo stability!"],
            ["Your tempo stability is good!", "You're doing well with tempo stability!", "You're doing great with tempo stability!", "You're almost there with tempo stability!"],
            ["Your tempo stability is great!", "You're doing amazing with tempo stability!", "Your tempo stability is amazing!", "You're almost perfect with tempo stability!"],
            ["Your tempo stability is perfect!", "Your tempo stability is spot on!", "Your tempo stability is amazing!", "Your tempo stability is perfect!"]
        ]
        
        comment["tempo"] = "Remember to slow down a bit at the end"
        comment["dynamics"] = "Generally good, you can try to make a stronger contrast"
        comment["finger"] = "great work!"
        
        self.comments = comment

    def success_response(self):
        url = "http://localhost:3001/recordings/" + str(self.recordingId)

        # dummy JSON data result
        data = {
            "score": self.score,
            "comment": self.comments,
            "charts": self.charts,
        }

        print("Sending response to server")

        response = requests.put(url, json=data)
        print("response code:", response.status_code)
        if response.status_code == 200:
            print("Response sent successfully")
        else:
            print("Response failed:", response.text)

    def fail_response(self):
        url = "http://localhost:3001/recordings/" + str(self.recordingId)
        response = requests.put(url, json={"error": "Analysis failed"})
        if response.status_code == 200:
            print("Response sent successfully")

    def run(self):      
        self.preprocess_video()
        _ = self.run_key_detection()

        if not _:
            self.fail_response()
            return

        self.run_hand_tracking()

        # clip audio from video
        clip = AudioFileClip(self.video_path)
        clip.write_audiofile(self.audio_path)

        self.run_audio_analysis()

        self.run_fingering_analysis()

        performance_note_time = self.audio_analyser.note_time_data
        piece_note_time = read_midi_note_time(self.midi_path)

        # todo: add fingering to it
        note_time2score(
            performance_note_time,
            musescore_path=self.musescore_path,
            output_xml_path=os.path.join(self.dir, "output_score.xml"),
            output_image_path=os.path.join(self.dir, "output_score.png"),
            piece_note_time=piece_note_time,
            fingerings=self.detected_fingering,
            hand=self.detected_hand,
            correct=self.detected_correct,
        )
        
        # generated suggested fingering
        note_time2score(
            performance_note_time,
            musescore_path=self.musescore_path,
            output_xml_path=os.path.join(self.dir, "output_gen_score.xml"),
            output_image_path=os.path.join(self.dir, "output_gen_score.png"),
            piece_note_time=piece_note_time,
            fingerings=self.generated_fingering,
            hand=self.detected_hand,
            correct=self.dummy_correct,
        )

        self.postprocess_charts()

        # add score computation
        self.compute_score()

        # add comment description
        self.generate_comment()
        
        # send response
        self.success_response()


if __name__ == "__main__":
    analyser = Analyser()
    analyser.run()
