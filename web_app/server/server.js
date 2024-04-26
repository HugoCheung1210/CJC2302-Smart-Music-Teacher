const express = require("express");
const cors = require("cors");
const path = require('path');
const mongoose = require("mongoose");
const bodyParser = require('body-parser');

const http = require('http');
const { initialize } = require('./io');

const { pieceRouter } = require("./routes/pieceRoute");
const { recordingRouter } = require("./routes/recordingRoute");
const { emotionRouter } = require("./routes/emotionRoute");
const { styleRouter } = require("./routes/styleRoute");

require("dotenv").config();

const app = express();
app.use(express.json());
app.use(cors());
app.use(bodyParser.json({ limit: '50mb' }));

// establish socket connection
const server = http.createServer(app);
const io = initialize(server);

const port = 3001;

// Database Connection
const db_url = process.env.DB_URL;
mongoose
  .connect(db_url)
  .then(() => {
    console.log("Connected to MongoDB");
  })
  .catch((err) => {
    console.log("Error connecting to MongoDB", err);
  });

app.use('/scores', express.static(path.join(__dirname, 'assets/scores')));

app.use('/recordings', express.static(path.join(__dirname, 'assets/recordings')));

app.use('/emotion', express.static(path.join(__dirname, 'assets/emotion')));

app.use('/style', express.static(path.join(__dirname, 'assets/style')));


// routes
app.get("/", (req, res) => {
  // to add landing page
  res.send("Hello World!");
});

app.use("/pieces", pieceRouter);
app.use("/recordings", recordingRouter);
app.use("/emotion", emotionRouter);
app.use("/style", styleRouter);

app.all("*", (req, res) => {
  res.status(404).json({ message: "Route not found" });
});

// start server
// app.listen(port, () => {
//   console.log(`Server listening at http://localhost:${port}`);
// });

server.listen(port, () => {
  console.log(`Server listening at http://localhost:${port}`);
});
