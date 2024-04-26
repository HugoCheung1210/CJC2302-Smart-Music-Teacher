import React from "react";
import { Link } from "react-router-dom";
import { useParams } from "react-router-dom";
import { useEffect, useState, useRef } from "react";
import axiosInstance from "../utils/axiosInstance";
import { Button } from "@material-tailwind/react";
import ReactPlayer from "react-player";


function AddRecording({ pieceId, closeModal }) {
  const [dateTime, setDateTime] = useState("");
  const [videoFile, setVideoFile] = useState(null);
  const [videoUrl, setVideoUrl] = useState(null);
  const [rotationAngle, setRotationAngle] = useState("0");
  const [loading, setLoading] = useState(false);

  const playerRef = useRef(null);

  const handleRotationChange = (event) => {
    setRotationAngle(event.target.value);
  };

  const handleVideoChange = (event) => {
    const file = event.target.files[0];
    setVideoFile(file);
    if (file) {
      const videoURL = URL.createObjectURL(file);
      setVideoUrl(videoURL);
    }
  };

  const handleDateTimeChange = (event) => {
    setDateTime(event.target.value);
  }

  // upload video after get recording id
  const handleVideoUpload = (recordingId) => {
    console.log("recordingId", recordingId);
    console.log("upload video file", videoFile);
    if (!videoFile) {
      alert("Please select a video to upload.");
      return;
    }
    
    const backgroundTime = playerRef.current.getCurrentTime();

    const formData = new FormData();
    formData.append("video", videoFile);
    formData.append("backgroundTime", backgroundTime);
    formData.append("rotationAngle", rotationAngle);
    formData.append("pieceId", pieceId);
    
    setLoading(true);
    fetch("http://localhost:3001/recordings/" + recordingId + "/upload", {
      method: "POST",
      body: formData,
    })
    .then(response => {
      setLoading(false);
      if (!response.ok) {
        throw new Error('Network response was not ok.');
      }
      return response.json();
    })
    .then(result => {
      alert(result.message);

      // wait for 0.5s then close
      setTimeout(() => {
        closeModal();
      }, 500);
    })
    .catch(error => {
      console.error("Error uploading video:", error);
      alert("Error uploading video.");
    });
  };

  const addNewRecording = () => {
    if (!dateTime) {
      alert("Please select a date and time.");
      return;
    }
    if (!videoFile) {
      alert("Please select a video to upload.");
      return;
    }
    
    // send post request 
    axiosInstance.post(`/recordings`, {
      pieceId: pieceId,
      dateTime: dateTime,
    }).then((response) => {
      console.log(response.data);
      const recordingId = response.data.recordingId;

      // alert(`New recording added with ID: ${recordingId}`);

      // upload the video only 
      handleVideoUpload(recordingId);

    }).catch((error) => {
      console.log("Error adding new recording:", error.response.data.error._message);
      alert("Error adding new recording: " + error.response.data.error._message);
    });
  };

  return (
    <div className="mt-5">
      {loading && <div className="absolute top-0 left-0 w-full h-full bg-gray-200 bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-white p-5 rounded-lg">
          <div className="flex justify-center">
            <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-gray-900"></div>
          </div>
          <div className="text-center font-semibold mt-4">Uploading...</div>
        </div>
      </div>}

      <div className="items-center">
        <div className="font-bold my-1 ms-1">Record Time</div>
        <div>
          <input
            type="datetime-local"
            className="block w-full px-4 py-2 mb-5 text-gray-800 bg-white border rounded-md shadow-sm focus:border-blue-500 focus:ring focus:ring-blue-200 focus:ring-opacity-50"
            // step="60"
            onChange={handleDateTimeChange}
            
          ></input>
        </div>
      </div>
      <div>
        <input
          type="file"
          accept="video/*"
          onChange={handleVideoChange}
          className="block w-full text-sm text-slate-500
                     file:mr-4 file:py-2 file:px-4
                     file:rounded-full file:border-0
                     file:text-sm file:font-semibold
                     file:bg-violet-50 file:text-violet-700
                     text-gray-800
                     hover:file:bg-violet-100"
        />
      </div>
      {videoFile && <div className="my-4 ">
        <div className="font-bold my-1 ms-1">Preview</div>
        
        <div className="flex justify-center px-1">
          <ReactPlayer url={videoUrl} ref={playerRef} width="100%" height="100%" controls={true} />
        </div>
        <div className="my-1 ms-1">
          Please drag the slider to a frame showing the keyboard without any hands.
        </div>
        <div className="my-3">
          <label htmlFor="rotation-dropdown" className="px-1">Rotation Angle</label>
          <select
            id="rotation-dropdown"
            value={rotationAngle}
            onChange={handleRotationChange}
            className="ml-2"
          >
            <option value="0">0°</option>
            <option value="90">90°</option>
            <option value="180">180°</option>
            <option value="270">270°</option>
          </select>
        </div>
      </div>}

      <div className="mt-5">
        <Button className="w-full mt-2 " onClick={addNewRecording}>
          Upload
        </Button>
      </div>
    </div>
  );
}

// modal box to be overlayed on recording overview
const AddRecordingModal = ({ onModalClose }) => {
  let { pieceId } = useParams();
  return (
    <div className="fixed inset-0 z-50 overflow-y-auto">
      <div className="flex items-center justify-center min-h-screen">
        <div className="fixed inset-0 transition-opacity">
          <div className="absolute inset-0 bg-black opacity-75"></div>
        </div>

        <div className="relative bg-white rounded-lg sm:w-4/5 w-3/4 py-7 px-8">
          <div className="flex justify-between">
            <h2 className="text-xl px-1 font-bold">New recording</h2>
            <Button className="" onClick={onModalClose}>
              CLOSE
            </Button>
          </div>
          <AddRecording pieceId={pieceId} closeModal={onModalClose} />
        </div>
      </div>
    </div>
  );
};

export default AddRecordingModal;
