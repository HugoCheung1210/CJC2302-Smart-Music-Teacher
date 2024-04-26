import cv2
from mido import MidiFile
import mido
import librosa
from music21 import chord, stream, converter, articulations, midi, clef, note
from PIL import Image
import subprocess
import os
import shutil

def frame_to_time(fps, frame_cnt):
    return frame_cnt / fps


def time_to_frame(fps, time):
    print(f"fps: {fps}, time: {time}")
    return int(fps * time)


def capture_background(video_path, time):
    cap = cv2.VideoCapture(video_path)
    frame_num = time_to_frame(cap.get(cv2.CAP_PROP_FPS), time)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    return frame


### Parse MIDI ###
def read_midi_note_time(midi_path):
    mid = MidiFile(midi_path)

    note_time = []
    time = 0
    note_unit = []

    for msg in mid:
        # print(msg)
        note_unit_time = "%.2f" % time
        if msg.type == "set_tempo":
            tempo = mido.tempo2bpm(msg.tempo)
            # Calculate the time duration of a quarter note
            quarter_note_duration = float(f"%.2f" % (60 / tempo))

        if msg.type == "note_on":
            time += msg.time

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
            else:
                time_symbol = 0.25

            if msg.velocity != 0:
                note_unit.append((librosa.midi_to_note(msg.note)))
            # if msg.velocity ==0:
            #     note_unit.append((librosa.midi_to_note(msg.note)))
        if note_unit_time != "%.2f" % time:
            note_time.append((note_unit, "%.2f" % time, time_symbol))
            note_unit = []

    return note_time


def split_score_hand(src_xml_path, output_xml_path, hand, correct, fingerings=None):
    print("length of hand", len(hand))
    print("length of correct", len(correct))
    print("length of fingerings", len(fingerings))
    
    s = converter.parse(src_xml_path)
    s2 = stream.Score()

    right_part = stream.Part()
    right_part.append(clef.TrebleClef())
    left_part = stream.Part()
    left_part.append(clef.BassClef())

    note_ct = 0

    # for each measure in score
    for i in range(len(s.parts[0].getElementsByClass("Measure"))):
        left = stream.Measure()
        right = stream.Measure()

        for element in s.parts[0].getElementsByClass("Measure")[i].elements:
            if isinstance(element, note.Note):
                ori_chord_odj = chord.Chord(
                    [element.pitch.nameWithOctave.replace("♯", "#")]
                )
                ori_chord_odj.quarterLength = element.quarterLength
            elif isinstance(element, chord.Chord):
                ori_chord_odj = element
            else:
                continue
            
            # print all pitches in chord
            print(ori_chord_odj.pitches)

            # left, right chord object
            chord_odj = [chord.Chord(), chord.Chord()]

            for k in range(len(ori_chord_odj.pitches)):
                if note_ct >= len(hand):
                    print("note_ct out of range", note_ct)
                    continue
                
                pitch = correct[note_ct][1].replace("♯", "#")
                hand_id = 0 if hand[note_ct] == "Left" else 1

                if fingerings is not None and fingerings[note_ct] is not None:
                    art = articulations.Fingering(fingerings[note_ct])
                    chord_odj[hand_id].articulations.append(art)

                chord_odj[hand_id].add(pitch)

                if correct[note_ct] is not None and not correct[note_ct][0]:
                    chord_odj[hand_id].setColor("red", pitch)

                note_ct += 1

            # get ori_chord_odj quarterLength
            chord_odj[0].quarterLength = ori_chord_odj.quarterLength
            chord_odj[1].quarterLength = ori_chord_odj.quarterLength
            
            # if empty, append rest
            if len(chord_odj[0].pitches) == 0:
                left.append(note.Rest(quarterLength=ori_chord_odj.quarterLength))
            else:
                left.append(chord_odj[0])
                
            if len(chord_odj[1].pitches) == 0:
                right.append(note.Rest(quarterLength=ori_chord_odj.quarterLength))
            else:
                right.append(chord_odj[1])
            
            # print("left",chord_odj[0].pitches)
            # print("right",chord_odj[1].pitches)

        # append measure to part
        left_part.append(left)
        right_part.append(right)

    s2.insert(0, right_part)
    s2.insert(0, left_part)
    
    s2.write("musicxml", output_xml_path)


def note_time2score(
    perf_note_time,
    musescore_path,
    output_xml_path,
    output_image_path,
    piece_note_time=None,
    fingerings=None,
    hand=None,
    correct=None,
):
    if piece_note_time is None:
        piece_note_time = perf_note_time

    # Create a stream
    s = stream.Stream()

    # Add notes with specific pitch and duration
    for i in range(len(perf_note_time)):

        for j in range(len(perf_note_time[i][0])):
            perf_note_time[i][0][j] = perf_note_time[i][0][j].replace("♯", "#")

        chord_odj = chord.Chord(perf_note_time[i][0])

        chord_odj.quarterLength = piece_note_time[i][2]
        s.append(chord_odj)

    # check if s contain measure - no
    # number_of_measures = len([element for element in s if isinstance(element, stream.Measure)])
    # print("Number of measures: " + str(number_of_measures))

    s.write("musicxml", output_xml_path)

    if hand is not None:
        split_score_hand(
            src_xml_path=output_xml_path,
            output_xml_path= output_xml_path[:-4] + "_LR.xml",
            hand=hand,
            correct=correct,
            fingerings=fingerings,
        )
    else:
        shutil.copy(output_xml_path, output_xml_path[:-4] + "_LR.xml")

    # Parse the XML file using music21
    score = converter.parse(output_xml_path)
    # find number of bars in score
    num_bars = len(score.parts[0].getElementsByClass("Measure"))

    print("Number of bars: " + str(num_bars))

    subprocess.run([musescore_path, "-o", output_image_path, output_xml_path])

    # Open the original PNG image
    img_path = output_image_path[:-4]
    img_filename = img_path + "-1.png"

    image = Image.open(img_filename)

    # Crop the image based on the bounding box coordinates
    # Get the height of the image
    image_height = image.height

    # Calculate the crop size based on a percentage of the height
    if num_bars <= 7:
        crop_percentage = 0.275  # Adjust this value as needed
    elif num_bars <= 15:
        crop_percentage = 0.425
    elif num_bars <= 23:
        crop_percentage = 0.53
    elif num_bars <= 31:
        crop_percentage = 0.675
    elif num_bars <= 39:
        crop_percentage = 0.8
    else:
        crop_percentage = 0.925
    crop_height = int(image_height * crop_percentage)

    crop_top = int(image_height * 0.12)
    # Crop the image
    cropped_image = image.crop((0, crop_top, image.width, crop_height))

    # Save the cropped image
    cropped_image.save(output_image_path)
    os.remove(img_filename)


def map_finger_to_key(note_df, key_finger_map):
    search_id = 0
    hand, finger_id = [], []
    # for each note in note_df
    for index, row in note_df.iterrows():
        time = row["Time"]
        pitch = row["Pitch"]

        if row["Class"] == 2:
            hand.append(None)
            finger_id.append(None)
            continue

        # search for timestamp that appears right after time in key_finger_map
        while search_id < len(key_finger_map) and key_finger_map[search_id][0] < time:
            search_id += 1

        time_notes = key_finger_map[search_id][1]

        found = False
        hand_val = None
        finger_id_val = None
        # search for note in time_notes
        for time_note in time_notes:
            if time_note[2] == None:
                continue

            note_pitch = time_note[2][0] + str(time_note[2][1])
            if note_pitch == pitch:
                hand_val = time_note[0]
                finger_id_val = time_note[1]
                found = True
                break

        if not found:
            print("note not found", time, pitch)
            # do fuzzy matching and find +- 2 notes
            for i in range(1, 3):
                if search_id + i < len(key_finger_map):
                    time_notes = key_finger_map[search_id + i][1]
                    for time_note in time_notes:
                        if time_note[2] == None:
                            continue

                        note_pitch = time_note[2][0] + str(time_note[2][1])
                        if note_pitch == pitch:
                            hand_val = time_note[0]
                            finger_id_val = time_note[1]
                            found = True
                            break

                if not found and search_id - i >= 0:
                    time_notes = key_finger_map[search_id - i][1]
                    for time_note in time_notes:
                        if time_note[2] == None:
                            continue

                        note_pitch = time_note[2][0] + str(time_note[2][1])
                        if note_pitch == pitch:
                            hand_val = time_note[0]
                            finger_id_val = time_note[1]
                            found = True
                            break

                if found:
                    break

        if not found:
            print("note still not found", time, pitch)

        # check if value is nan
        # if hand_val and hand_val == "":
        #     hand_val = None
        # if finger_id_val and finger_id_val.isnan():
        #     finger_id_val = None

        hand.append(hand_val)
        finger_id.append(finger_id_val)

    # add finger as new column in note_df
    note_df["hand"] = hand
    note_df["finger_id"] = finger_id
    return note_df


def note_time2midi(perf_note_time, output_midi_path, piece_note_time=None):
    if piece_note_time is None:
        piece_note_time = perf_note_time

    # Create a stream
    s = stream.Stream()
    # Add notes with specific pitch and duration
    for i in range(len(perf_note_time)):
        for j in range(len(perf_note_time[i][0])):
            perf_note_time[i][0][j] = perf_note_time[i][0][j].replace("♯", "#")

        chord_odj = chord.Chord(perf_note_time[i][0])
        chord_odj.quarterLength = piece_note_time[i][2]
        s.append(chord_odj)

    mf = midi.translate.streamToMidiFile(s)
    mf.open(output_midi_path, "wb")
    mf.write()
    mf.close()

if __name__ == "__main__":
    midi_path = os.path.join("data", "scoers", "Ode_to_Joy.mid")