// handle CRUD for events
const express = require("express");
const recordingController = require("../controllers/recordingController");
const router = express.Router();

// get all recordings by pieceid (visible and hidden)
router.get("/:pieceId/visible", recordingController.getAllRecordings);
router.get("/:pieceId/hidden", recordingController.getAllHiddenRecordings);

// add new recording (pieceId, dateTime)
router.post("/", recordingController.addNewRecording);

// get recording by recordingId
router.get("/:recordingId", recordingController.getRecordingById);

// add score and comments to recording
router.put("/:recordingId", recordingController.updateRecording);

// upload recording video
router.post("/:recordingId/upload", recordingController.upload, recordingController.uploadRecording);

router.post("/:recordingId/testpy", recordingController.testPython);

module.exports.recordingRouter = router;
