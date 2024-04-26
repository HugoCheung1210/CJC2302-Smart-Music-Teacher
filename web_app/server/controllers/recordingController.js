const Recording = require("../models/recordingModel");
const Piece = require("../models/pieceModel");
const multer = require("multer");
const path = require("path");
const fs = require("fs");
const { exec } = require("child_process");

const { getIo } = require("../io");
require("dotenv").config();


// get all recordings
const getAllRecordings = (req, res) => {
  Recording.find({ visible: true, pieceId: req.params.pieceId })
    .sort({ datetime: -1 })
    .then((recordings) => {
      res.status(200).json(recordings);
    })
    .catch((err) => {
      res.status(400).json({ error: err });
    });
};

// get all hidden recordings
const getAllHiddenRecordings = (req, res) => {
  Recording.find({ visible: false, pieceId: req.params.pieceId})
    .then((recordings) => {
      res.status(200).json(recordings);
    })
    .catch((err) => {
      res.status(400).json({ error: err });
    });
}

// get recording by recordingId
const getRecordingById = (req, res) => {
  Recording.findOne({ recordingId: req.params.recordingId })
    .then((recording) => {
      res.status(200).json(recording);
    })
    .catch((err) => {
      res.status(400).json({ error: err });
    });
};

// add new recording
const addNewRecording = (req, res) => {
  // console.log(req.body.dateTime);
  const dateObj = new Date(req.body.dateTime);

  // console.log("date object", dateObj)
  const newRecording = new Recording({
    pieceId: req.body.pieceId,
    datetime: dateObj,
  });

  newRecording
    .save()
    .then(() => {
      // return recording Id
      res.status(201).json({ recordingId: newRecording.recordingId });
    })
    .catch((err) => {
      console.log(err + "error");
      res.status(400).json({ error: err });
    });
};

// update recording
const updateRecording = (req, res) => {
  // console.log(req.body);
  console.log(req.params.recordingId, "update")
  Recording.findOne({ recordingId: req.params.recordingId })
    .then((recording) => {
      if (req.body.score) {
        recording.score = req.body.score;
      }
      if (req.body.comment) {
        recording.comment = req.body.comment;
      }
      if (req.body.charts) {
        recording.charts = req.body.charts;
      }
      recording.visible = true;

      recording
        .save()
        .then(() => {
          res.status(200).json({ message: "Recording updated successfully" });
          console.log("Recording updated successfully");
        })
        .catch((err) => {
          res.status(400).json({ error: err });
          console.log(err);
        });
    })
    .catch((err) => {
      res.status(400).json({ error: err });
    });
};


// --------- Upload recording ---------
// Configure storage for Multer
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    const recordingPath = path.join(
      "assets",
      "recordings",
      req.params.recordingId
    );

    // Check if the directory exists
    fs.access(recordingPath, (error) => {
      if (error) {
        // Directory does not exist, so create it
        fs.mkdir(recordingPath, { recursive: true }, (error) => {
          if (error) {
            cb(error);
          } else {
            cb(null, recordingPath);
          }
        });
      } else {
        // Directory exists, pass it to the callback
        cb(null, recordingPath);
      }
    });
  },
  filename: function (req, file, cb) {
    cb(null, "raw_video" + path.extname(file.originalname));
  },
});

const upload = multer({ storage: storage }).single("video");

async function findMidiName (pieceId){
  // get from database piece midi_name
  const result = await Piece.findOne({ pieceId: pieceId })
  console.log("found midi", result.midi_name);
  return result.midi_name;

  // Piece.findOne({ pieceId: pieceId})
  //   .then((piece) => {
  //     console.log("found midi", piece.midi_name);
  //     return piece.midi_name;
  //   })
  //   .catch((err) => {
  //     console.log(err);
  //     return null;
  //   });
}

// to trigger client to reload page
function triggerClientReload() {
  const io = getIo();
  io.sockets.emit('reloadPage');
}

// trigger python script to run analysis
async function runAnalysis(req) {
  const recordingId = req.params.recordingId
  const backgroundTime = req.body.backgroundTime;
  const rotationAngle = req.body.rotationAngle;
  const pieceId = req.body.pieceId;

  const condaPath = process.env.CONDA_PATH;
  // console.log("conda path", condaPath);
  const scriptPath = path.normalize(path.join("analysis_scripts", "start_analysis.py"));

  const recordingDir = path.normalize(path.join("assets", "recordings", recordingId));

  // get from database piece midi_name
  // todo: set to wait for findMidiName
  console.log("piece id", pieceId)
  const midi_name = await findMidiName(pieceId);
  console.log("midi name", midi_name);
  const midi_path = path.normalize(path.join("assets", "scores", midi_name));

  const musescore_path = process.env.MUSESCORE_PATH;

  // check if condapath exist
  if (!fs.existsSync(condaPath)) {
    console.error("Error: Conda path does not exist");
    return;
  }

  // check if scriptPath exist
  if (!fs.existsSync(scriptPath)) {
    console.error("Error: Script path does not exist");
    return;
  }

  // Construct the command to execute
  const command = `${condaPath} ${scriptPath} --dir ${recordingDir} --videoRotation ${rotationAngle} --backgroundTime ${backgroundTime} --recordingId ${recordingId} --midiPath ${midi_path} --musescorePath "${musescore_path}"`;

  // Run the Python script using exec
  console.log("starting python script")
  exec(command, (error, stdout, stderr) => {
    console.log("Python script executed");
    console.log(`Stdout: ${stdout}`);

    if (error) {
      console.error(`Error: ${error.message}`);
      return;
    }

    triggerClientReload();
    console.log("reload event sent");
    // if (stderr) {
    //   console.error(`Stderr: ${stderr}`);
    //   return;
    // }
  });

}

const uploadRecording = (req, res) => {
  // console.log(req.params.recordingId);
  res.status(200).json({ message: "File uploaded successfully" });

  runAnalysis(req);
};

const testPython = (req, res) => {
  runAnalysis(req);
  res.status(200).json({ message: "Python script executed successfully" });
}

module.exports = {
  getAllRecordings,
  getAllHiddenRecordings,
  getRecordingById,
  addNewRecording,
  updateRecording,
  upload,
  uploadRecording,

  testPython,
};
