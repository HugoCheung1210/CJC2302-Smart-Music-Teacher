// handle CRUD for events
const express = require("express");
const emotionController = require("../controllers/emotionController");
const router = express.Router();


// upload recording video
router.post("/upload", emotionController.upload, emotionController.uploadFile);

module.exports.emotionRouter = router;