// handle CRUD for events
const express = require("express");
const styleController = require("../controllers/styleController");
const router = express.Router();


// upload recording video
router.post("/upload", styleController.upload, styleController.uploadFile);

router.post("/transfer", styleController.transferFile);

module.exports.styleRouter = router;