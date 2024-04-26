const mongoose = require("mongoose");
const AutoIncrement = require('mongoose-sequence')(mongoose);

const Schema = mongoose.Schema;

const pieceSchema = new Schema({
    name: {
        type: String,
        required: true,
    },
    composer: {
        type: String,
    },
    description: {
        type: String,
    },
    pdf_name: {
        type: String,
    },
    midi_name: {
        type: String,
    },
    /*
    link to piece pdf score 
    */
});

// Apply the auto-increment plugin and specify the options
pieceSchema.plugin(AutoIncrement, { inc_field: 'pieceId' });

const Piece = mongoose.model("Piece", pieceSchema);

module.exports = Piece;