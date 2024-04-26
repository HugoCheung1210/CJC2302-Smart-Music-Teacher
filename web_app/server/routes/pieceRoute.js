// handle CRUD for events
const express = require("express");

const pieceController = require("../controllers/pieceController");
const router = express.Router();

router.get("/", pieceController.getAllPieces);

router.post("/", pieceController.addNewPiece);

router.get("/:pieceId", pieceController.getPieceById);

router.get("/:pieceId/recordings", pieceController.getRecordingsByPieceId);

module.exports.pieceRouter = router;