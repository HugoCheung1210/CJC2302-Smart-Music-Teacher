const multer = require("multer");
const path = require("path");
const fs = require("fs");
const { exec } = require("child_process");

require("dotenv").config();

// --------- Upload recording ---------
// Configure storage for Multer
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    const storagePath = path.join("assets", "style");

    // Check if the directory exists
    fs.access(storagePath, (error) => {
      if (error) {
        // Directory does not exist, so create it
        fs.mkdir(storagePath, { recursive: true }, (error) => {
          if (error) {
            return cb(error);
          }
          cb(null, storagePath);
        });
      } else {
        // Directory exists, clean up the directory
        cb(null, storagePath)
        // fs.readdir(storagePath, (err, files) => {
        //   if (err) {
        //     return cb(err);
        //   }
        //   // Use Promise.all to wait for all files to be deleted
        //   Promise.all(
        //     files.map((file) => {
        //       return new Promise((resolve, reject) => {
        //         fs.unlink(path.join(storagePath, file), (err) => {
        //           if (err) {
        //             return reject(err);
        //           }
        //           resolve();
        //         });
        //       });
        //     })
        //   )
        //     .then(() => cb(null, storagePath))
        //     .catch((err) => cb(err));
        // });
      }
    });
  },
  filename: function (req, file, cb) {
    cb(null, "input.mid");
  },
});

const upload = multer({ storage: storage }).single("file");

const uploadFile = (req, res) => {
  // run style recognition script
  const condaPath = process.env.CONDA_PATH;
  const scriptPath = path.normalize(
    path.join("analysis_scripts", "piece_analyser.py")
  );

  const projPath = path.normalize(
    path.join("assets", "style")
  );

  const musescorePath = process.env.MUSESCORE_PATH;

  const command = `${condaPath} ${scriptPath} --dir ${projPath} --musescorePath "${musescorePath}" --mode analyze`;
  console.log("start execute: ", command);

  exec(command, (error, stdout, stderr) => {
    if (error) {
      console.log("stdout: ", stdout);
      console.error(`exec error: ${error}`);
      res.status(500).json({ message: "style analysis failed" });
      return;
    }

    const output_lines = stdout.split("\n");
    const output = output_lines[output_lines.length - 2];
    console.log("output: ", output);

    res.status(200).json({ 
      message: "style analyzed successfully",
      style: output
    });
    console.log(`stdout: ${stdout}`);
    console.error(`stderr: ${stderr}`);
  });
};

const transferFile = (req, res) => {
  // run style recognition script
  const condaPath = process.env.CONDA_PATH;
  const scriptPath = path.normalize(
    path.join("analysis_scripts", "piece_analyser.py")
  );

  const projPath = path.normalize(
    path.join("assets", "style")
  );

  const musescorePath = process.env.MUSESCORE_PATH;

  const command = `${condaPath} ${scriptPath} --dir ${projPath} --musescorePath "${musescorePath}" --mode transfer`;
  console.log("start execute: ", command);

  exec(command, (error, stdout, stderr) => {
    if (error) {
      console.log("stdout: ", stdout);
      console.error(`exec error: ${error}`);
      res.status(500).json({ message: "style transfer failed" });
      return;
    }

    res.status(200).json({ 
      message: "style transferred successfully"
    });
    console.log(`stdout: ${stdout}`);
    console.error(`stderr: ${stderr}`);
  });
}

module.exports = {
  upload,
  uploadFile,
  transferFile
};
