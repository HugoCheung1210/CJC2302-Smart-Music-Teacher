const multer = require("multer");
const path = require("path");
const fs = require("fs");
const { exec } = require("child_process");

require("dotenv").config();

// --------- Upload recording ---------
// Configure storage for Multer
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    const storagePath = path.join("assets", "emotion");

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
        cb(null, storagePath);
      }
    });
  },
  filename: function (req, file, cb) {
    cb(null, "source_file" + path.extname(file.originalname));
  },
});

const upload = multer({ storage: storage }).single("file");

const uploadFile = (req, res) => {
  // run emotion recognition script
  const condaPath = process.env.CONDA_PATH;
  const scriptPath = path.normalize(
    path.join("analysis_scripts", "emotion_analysis.py")
  );

  const projPath = path.normalize(
    path.join("assets", "emotion")
  );

  // const command = `${condaPath} ${scriptPath} --dir ${projPath}`;
  const command = `echo "success"`;
  console.log("start execute: ", command);


  exec(command, (error, stdout, stderr) => {
    if (error) {
      console.error(`exec error: ${error}`);
      return;
    }
    // console.log(req.params.recordingId);
    res.status(200).json({ message: "emotion analyzed successfully" });
    console.log(`stdout: ${stdout}`);
    console.error(`stderr: ${stderr}`);
  });
};

module.exports = {
  upload,
  uploadFile,
};
