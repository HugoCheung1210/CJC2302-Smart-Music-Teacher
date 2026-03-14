# Web-based Piano Music Tutoring using Artificial Intelligence

This project focuses on building a web application that would help users improve their
piano-playing ability and musicality through performing analysis on different piano pieces or
performances and then presenting the analytics in an understandable and meaningful way.
The application has three major functionalities: performance analysis, emotion analysis and
music piece analysis.


The performance analysis function is designed to provide feedback on users’ piano practices.
It transcribes users’ performances from practice recordings, compares them against the
music score, and offers scoring and improvement suggestions. The emotion analysis function
examines the emotional changes in a music recording, utilizing a valence-arousal graph. As
for the music piece analysis functions, they classify the music style, recreate it in a different
style, and generate piano fingerings to facilitate users in playing the piece effectively.

This Paper was published at the UbiMus2024 Conference

## Novelty functions
### Emotion Analysis
This component visualizes the feeling conveyed throughout the recording to nurture musicality
and appreciation. The valence-arousal model is used thanks to its continuous
intensity scale. The 4Q audio emotion dataset [Panda et al. 2018], annotated into 4 quadrants
according to Russell’s circumplex model [Feldman Barrett and Russell 1998], was
used. Referencing ER and ESIN [Er and Esin 2021], 1×17 audio feature vectors were
generated and marked with emotion labels Q1, Q2, Q3, Q4. Comparing Random Forest,
Support Vector Machine, and Gradient Boosting Decision Trees (GBDT), GBDT
achieved the best overall accuracy of 71.21% with a 0.005 learning rate and 200 estimators.
The model predicts 5-second segments every 0.5 seconds, plotting emotion changes
on the 4-quadrant valence-emotion graph.

### Piece Analysis
The music era classification function categorizes user input into one of four musical eras,
supporting the style transfer function. Our model uses time series of chroma features from
audio for training. We extract spatial features from 12 chroma classes using a CNN model,
followed by a Bi-LSTM to preserve past and future information. The best CNN+Bi-
LSTM model, trained with a batch size of 32 and a learning rate of 0.0001, achieved a
validation accuracy of 57%, outperforming the baseline Bi-LSTM model.


For music era style transfer, we aimed to transfer audio from one era to another,
offering users a way to experiment with different musical styles. The Giant-MIDI Piano
dataset [Kong et al. 2022] was used, with the musical era labelled based on composer
birth years. Referencing CycleGAN [Zhu et al. 2017], we trained 20 epochs for each pair
of models using processed pianoroll data. The best result was obtained from the Baroque
and Modern eras CycleGAN model with a total loss of 1.27. The classification and target
transfer era are used for style transfer


For piano fingering generation, a long short-term memory (LSTM) model was
trained to generate fingerings for a sequence of notes. It was trained on a piano fingering
dataset prepared by [Nakamura et al. 2020]. Basing on previous work [Guan et al. 2021],
we introduced new features (onset time difference, note distance, previous and current
key color) to achieve polyphonic piano fingering prediction for both hands.

## Repository structure

- `web_app/` — Web application (client + server)
- `Web-based Piano Music Tutoring using Artificial Intelligence_Full_Version.pdf` — Project report (full version)

## Getting started

### Prerequisites
- **Node.js + npm** (for the web client/server)
- (Optional / if you use the AI components) **Python 3.x**

### Clone
```bash
git clone https://github.com/HugoCheung1210/Web-based-Piano-Music-Tutoring-using-Artificial-Intelligence.git
cd Web-based-Piano-Music-Tutoring-using-Artificial-Intelligence
```

## Run the web app

The `web_app/README.md` currently documents starting the server and client with `npm start`.

### Start Server
```bash
cd web_app/server
npm start
```

### Start Client
```bash
cd web_app/client
npm start
```

Then open:
- `http://localhost:3000`

## Documentation

- See the full paper: `Web-based Piano Music Tutoring using Artificial Intelligence_Full_Version.pdf`

## Tech stack

This is a web application built using the MERN stack (MongoDB, Express, React, Node),
while the analysis and AI-related features are performed by Python scripts triggered at
the back-end server. The system consists of three main components, namely performance
analysis, emotion analysis, and music piece analysis.

