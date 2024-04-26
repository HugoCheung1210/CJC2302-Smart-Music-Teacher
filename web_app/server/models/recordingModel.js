const mongoose = require("mongoose");
const AutoIncrement = require("mongoose-sequence")(mongoose);

const Schema = mongoose.Schema;

const scoreSchema = new Schema({
  overall: { type: Number, required: true },
  pitch_acc: { type: Number, required: true },
  tempo_acc: { type: Number, required: true },
  dyn_cons: { type: Number, required: true },
  dyn_range: { type: Number, required: true },
  tempo_stab: { type: Number, required: true },
  finger: { type: Number, required: true },
});

const commentSchema = new Schema({
    overall: { type: String, required: true},
    pitch: { type: String, required: true },
    tempo: { type: String, required: true },
    dynamics: { type: String, required: true },
    finger: { type: String, required: true },
});

const chartSchema = new Schema({
  xAxis: [Number],
  yAxis: [Number],
  verticalLines: [Number],
});

const chartsSchema = new Schema({
  pitchAccuracy: chartSchema,
  tempo: chartSchema,
  PLP: chartSchema,
  onset: chartSchema,
  dynamics: chartSchema,
  generalDynamics: chartSchema,
  fingeringAccuracy: chartSchema,
});

const recordingSchema = new Schema({
  pieceId: {
    type: Number,
    required: [true, "Recording must be associated with a piece"],
  },
  datetime: {
    type: Date,
    required: [true, "Recording must have a date and time"],
  },
  score: {
    type: scoreSchema,
  },
  comment: {
    type: commentSchema,
  },
  charts: {
    type: chartsSchema,
  },
  createdAt: {
    type: Date,
    default: Date.now,
  },
  visible: {
    type: Boolean,
    default: false,
  },
});

// Apply the auto-increment plugin and specify the options
recordingSchema.plugin(AutoIncrement, { inc_field: "recordingId" });

const Recording = mongoose.model("Recording", recordingSchema);

module.exports = Recording;
