const Piece = require("../models/pieceModel");

require("dotenv").config();

// get all pieces
const getAllPieces = (req, res) => {
  Piece.find()
    .then((pieces) => {
      res.status(200).json(pieces);
    })
    .catch((err) => {
      res.status(400).json({ error: err });
    });
};

// get piece by pieceId
const getPieceById = (req, res) => {
  Piece.findOne({ pieceId: req.params.pieceId })
    .then((piece) => {
      res.status(200).json(piece);
    })
    .catch((err) => {
      res.status(400).json({ error: err });
    });
};

// TODO: get recordings by pieceId
const getRecordingsByPieceId = (req, res) => {
  res.status(501).json("Not implemented");
};

// add new piece
const addNewPiece = (req, res) => {
  const newPiece = new Piece({
    name: req.body.name,
    composer: req.body.composer,
    description: req.body.description,
    pdf_name: req.body.pdf_name,
    midi_name: req.body.midi_name,
  });

  newPiece
    .save()
    .then(() => {
      res.status(201).json("Piece added!");
    })
    .catch((err) => {
      res.status(400).json({ error: err });
    });
};

module.exports = {
  getAllPieces,
  getPieceById,
  getRecordingsByPieceId,
  addNewPiece,
};
