import React from "react";
import { useParams, Link } from "react-router-dom";
import { useEffect, useState, useRef } from "react";
import { Button, Typography } from "@material-tailwind/react";
import axiosInstance from "../utils/axiosInstance";
import NavbarWithMegaMenu from "./NavbarWithMegaMenu";
import { ChevronDownIcon } from "@heroicons/react/24/solid";

function EmotionAnalysis() {
  const [file, setFile] = useState(null);
  const [audioPreview, setAudioPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(false);
  const [explainIsOpen, setExplainIsOpen] = useState(false);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setFile(file);
    console.log(audioPreview);
    // Revoke the previous URL if it exists
    if (audioPreview) {
      URL.revokeObjectURL(audioPreview.src);
    }

    setAudioPreview(null);
    // wait 0.2s
    setTimeout(() => {
      // If the file is an audio or video file, create an object URL for preview
      if (
        file &&
        (file.type.includes("audio") || file.type.includes("video"))
      ) {
        const audioURL = URL.createObjectURL(file);
        setAudioPreview({ src: audioURL });
      } else {
        // If it's not a compatible file type, set the preview to null
        setAudioPreview(null);
      }
    }, 200);
  };

  // upload video after get recording id
  const handleFlieUpload = () => {
    console.log("upload file", file);
    if (!file) {
      alert("Please select a file to upload.");
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);

    fetch("http://localhost:3001/emotion/upload", {
      method: "POST",
      body: formData,
    })
      .then((response) => {
        setLoading(false);
        if (!response.ok) {
          throw new Error("Network response was not ok.");
        }
        return response.json();
      })
      .then((result) => {
        // alert(result.message);
        // fetch and display result from server storage
        setResult(true);
      })
      .catch((error) => {
        setLoading(false);
        console.error("Error uploading video:", error);
        alert("Error uploading video.");
      });
  };

  const toggleExplanationDropdown = () => {
    setExplainIsOpen(!explainIsOpen);
  };

  return (
    <div>
      <NavbarWithMegaMenu />

      <div className="lg:w-4/5 lg:mx-auto">
        <div className="mx-5 my-5 ">
          <div className="font-bold text-xl">Emotion Analysis</div>
          <div className="my-2">
            Upload a video or audio file to analyze the emotions.
          </div>
          <div className="flex items-center justify-center">
            <input
              type="file"
              accept="video/*,audio/*"
              onChange={handleFileChange}
              className="block w-full text-sm text-slate-500 mt-5
                   file:mr-4 file:py-2 file:px-4
                   file:rounded-full file:border-0
                   file:text-sm file:font-semibold
                   file:bg-violet-50 file:text-violet-700
                   text-gray-800
                   hover:file:bg-violet-100"
            />
          </div>

          {audioPreview && (
            <div className="flex sm:justify-between ">
              <div className="my-4 ">
                {/* <div className="font-bold my-1 ms-1">Preview</div> */}
                <div>
                  <audio controls>
                    <source src={audioPreview.src} type={file.type} />
                    Your browser does not support the audio element.
                  </audio>
                </div>
              </div>
              {/* freeze button if loading */}
              <div className="my-6 me-5">
                <Button
                  size="md"
                  className="w-full px-10"
                  onClick={handleFlieUpload}
                  disabled={loading}
                >
                  {loading ? "Uploading..." : "Upload"}
                </Button>
              </div>
            </div>
          )}
        </div>

        <div className="flex flex-col w-full justify-center px-5 mt-2">
          <div className="flex flex-col justify-center ">
            <div
              className="flex cursor-pointer select-none items-center hover:bg-gray-100 py-2 px-1 rounded-md"
              onClick={toggleExplanationDropdown}
            >
              <Typography className="text-black font-bold">
                Valence Arousal Graph Explanation
              </Typography>
              <ChevronDownIcon
                strokeWidth={2.5}
                className={`ml-2 h-4 w-4 transition-transform ${
                  explainIsOpen ? "rotate-180" : ""
                }`}
              />
            </div>
            {explainIsOpen && (
              <div className="my-1 grid md:grid-cols-2 w-full">
                <div className="sm:w-4/5 lg:w-full mx-auto">
                  <img
                    src={
                      process.env.REACT_APP_SERVER_BASE_URL +
                      "emotion/arousal_valence_graph.png"
                    }
                    alt="Arousal Valence Graph"
                    className="justify-center mx-auto"
                  />
                </div>
                <div className="justify-center align-middle my-auto mx-2 p-6 bg-white rounded-lg shadow-md font-semibold text-gray-600 ">
                  The valence-arousal graph represents the emotional intensity (arousal) and positivity/negativity (valence) of the music.
                  <br />
                  <br />
                  Arousal is a measure of the intensity of the emotion, ranging from calm to excited. Valence is a measure of the positivity or negativity of the emotion, ranging from negative to positive.
                </div>
              </div>
            )}
          </div>
        </div>
        {result && (
          <div className="mx-5 my-5">
            {/* <div className="font-bold text-xl">Result</div> */}
            <div className="my-2">
              {/* Image from server/emotion/emotion_plot.png */}
              <img
                src={
                  process.env.REACT_APP_SERVER_BASE_URL +
                  "emotion/emotion_plot.png"
                }
                alt="Emotion Analysis"
                className="lg:w-1/2 md:w-4/5 justify-center mx-auto"
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default EmotionAnalysis;
