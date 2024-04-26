import os
import pandas as pd
import numpy as np
import ast
from music21 import midi


class midiParser:
    def __init__(self, midi_path):
        self.midi_path = midi_path

    # Function to extract note on, note off, and midi pitch from a MIDI file
    def extract_notes_from_midi(file_path):
        # Read the MIDI file
        mf = midi.MidiFile()
        mf.open(file_path)
        mf.read()
        mf.close()

        # Create a list to hold note data
        note_data = []

        for track in mf.tracks:
            for event in track.events:
                if isinstance(event, midi.MidiEvent):
                    if event.type == "NOTE_ON" and event.velocity > 0:
                        note_data.append(["note_on", event.pitch])
                    elif (
                        event.type == "NOTE_ON" and event.velocity == 0
                    ) or event.type == "NOTE_OFF":
                        note_data.append(["note_off", event.pitch])

        # Filter out the note_on/note_off events to pair them
        paired_note_data = []
        active_notes = {}

        for data in note_data:
            note_type, pitch = data
            if note_type == "note_on":
                if pitch not in active_notes:
                    active_notes[pitch] = []
                active_notes[pitch].append("note_on")
            elif note_type == "note_off":
                if pitch in active_notes and "note_on" in active_notes[pitch]:
                    active_notes[pitch].remove("note_on")
                    paired_note_data.append(
                        [pitch]
                    )  # Assuming we just want to record the pitch

        return paired_note_data

    


midi_path = os.path.join("fingering_dataset", "midi_files", "c_major_scale.mid")
parser = midiParser(midi_path)
