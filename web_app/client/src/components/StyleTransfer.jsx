import React from "react";
import { useParams, Link } from "react-router-dom";
import { useEffect, useState, useRef } from "react";
import { Button } from "@material-tailwind/react";
import { OpenSheetMusicDisplay } from "opensheetmusicdisplay";
import axiosInstance from "../utils/axiosInstance";
import NavbarWithMegaMenu from "./NavbarWithMegaMenu";

function StyleTransfer() {
  const [file, setFile] = useState(null);
  const [osmd, setOsmd] = useState(null);
  

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(false);
  
  const [currentStyle, setCurrentStyle] = useState("");
  const musicDisplayRef = useRef(null);

  const [transferLoading, setTransferLoading] = useState(false); 
  const [transferResult, setTransferResult] = useState(false);
  const [transferOsmd, setTransferOsmd] = useState(null);
  const transferMusicDisplayRef = useRef(null);

  const musicStyles = ["Baroque", "Classical", "Romantic", "Modern"];

  const styleDescription = [
    "Baroque music is characterized by its dramatic contrasts, expressive melodies, and use of basso continuo to create a rich, harmonically-driven style that aimed to communicate powerful emotions and narratives.",
    "Classical music is a formal musical tradition of the Western world, characterized by complexity in its musical form and harmonic organization, often incorporating elements from popular and folk music traditions.",
    "Romantic music is characterized by its emotional expressiveness, dramatic contrasts, and use of expanded musical forms and harmonies to evoke powerful feelings and narratives.",
    "Modern music is characterized by a diverse range of experimental styles that challenge traditional musical forms and conventions, often incorporating elements of atonality, polytonality, and new approaches to harmony, melody, and rhythm.",
  ];

  // update file state when user selects a file
  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setFile(file);
  };

  // upload video after get recording id
  const handleFileUpload = () => {
    console.log("upload file", file);
    if (!file) {
      alert("Please select a file to upload.");
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);

    fetch(process.env.REACT_APP_SERVER_BASE_URL + "style/upload", {
      method: "POST",
      body: formData,
    })
      .then((response) => {
        setLoading(false);
        if (!response.ok) {
          throw new Error("Network response was not ok.");
        }
        console.log(response);
        return response.json();
      })
      .then((result) => {
        setCurrentStyle(result.style);
        console.log("result", result);
        // fetch and display result from server storage
        setTimeout(() => {
          setResult(true);
          // set timeout
          setTimeout(() => {
            if (osmd) {
              osmd.clear();
            }
            renderMusicScore();
          }, 200);
        }, 200);
      })
      .catch((error) => {
        setLoading(false);
        console.error("Error uploading file:", error);
        alert("Error uploading file: " + error.message);
      });
  };

  // render music score using osmd
  const renderMusicScore = () => {
    if (!osmd) {
      const osmd = new OpenSheetMusicDisplay(musicDisplayRef.current, {
        autoResize: true,
        backend: "canvas",
        drawingParameters: "compacttight",
        drawTitle: false,
      });
      setOsmd(osmd);

      osmd
        .load(process.env.REACT_APP_SERVER_BASE_URL + "style/input_score_LR.xml")
        .then(() => {
          osmd.render();
        })
        .catch((error) => {
          console.error("Error rendering music score:", error);
        });
    } else {
      osmd
        .load(process.env.REACT_APP_SERVER_BASE_URL + "style/input_score_LR.xml")
        .then(() => {
          osmd.render();
          setOsmd(osmd);
        })
        .catch((error) => {
          console.error("Error rendering music score:", error);
        });
    }
  };

  const renderTransferMusicScore = () => {
    if (!transferOsmd) {
      const osmd = new OpenSheetMusicDisplay(transferMusicDisplayRef.current, {
        autoResize: true,
        backend: "canvas",
        drawingParameters: "compacttight",
        drawTitle: false,
      });
      setTransferOsmd(osmd);

      osmd
        .load(process.env.REACT_APP_SERVER_BASE_URL + "style/output_score_LR.xml")
        .then(() => {
          osmd.render();
        })
        .catch((error) => {
          console.error("Error rendering music score:", error);
        });
    } else {
      transferOsmd
        .load(process.env.REACT_APP_SERVER_BASE_URL + "style/output_score_LR.xml")
        .then(() => {
          osmd.render();
          setTransferOsmd(osmd);
        })
        .catch((error) => {
          console.error("Error rendering music score:", error);
        });
    }
  };

  // style radio button
  const StyleButton = ({ style, description }) => {
    
    const disabled = (style.trim() === currentStyle.trim());


    var text_style;
    if (disabled) {
      text_style = "text-gray-400 bg-gray-100 border-gray-200";
    } else {
      text_style = "text-gray-600 bg-white border-gray-200 hover:bg-black hover:text-white";
    }

    return (
      <li>
        <input
          type="radio"
          id={style}
          name="style"
          value={style}
          className="hidden peer"
          required
          disabled={disabled}
        />
        <label
          htmlFor={style}
          className={"inline-flex items-center justify-between w-full p-5 " + text_style + " border border-gray-200 rounded-lg cursor-pointer peer-checked:bg-black peer-checked:text-white   "}
        >
          <div className="block">
            <div className="w-full text-lg font-semibold">{style}</div>
            <div className="w-full">{description}</div>
          </div>
        </label>
      </li>
    );
  };

  const handleStyleTransfer = () => {
    console.log("style transfer");
    // get style button value
    const style = document.querySelector('input[name="style"]:checked').value;
    console.log("style selected", style);
    setTransferResult(false);
    setTransferLoading(true);

    // send request to server using axios
    axiosInstance.post("/style/transfer", { style: style }).then((response) => {
      console.log(response.data);

      setTransferLoading(false);
      setTransferResult(true);

      setTimeout(() => {
        if (transferOsmd) {
          transferOsmd.clear();
        }
        renderTransferMusicScore();
      }, 200);
    })
    
    // temp handle
    // setTransferResult(true);
    // renderTransferMusicScore();
  };

  return (
    <div>
      <NavbarWithMegaMenu />

      <div className="lg:w-4/5 lg:mx-auto">
        <div className="mx-5 my-5 ">
          <div className="font-bold text-xl">Piece Analysis</div>
          <div className="my-2">
            Upload a midi file and obtain its presentation in another music
            style.
          </div>
          <div className="flex items-center justify-between">
            <input
              type="file"
              accept=".mid"
              onChange={handleFileChange}
              className=" w-full text-sm text-slate-500
                   file:mr-4 file:py-2 file:px-4
                   file:rounded-full file:border-0
                   file:text-sm file:font-semibold
                   file:bg-violet-50 file:text-violet-700
                   text-gray-800
                   hover:file:bg-violet-100"
            />
            {file && (
              <div className="my-6 me-5">
                <Button
                  size="md"
                  className="w-full px-10"
                  onClick={handleFileUpload}
                  disabled={loading}
                >
                  {loading ? "Uploading..." : "Upload"}
                </Button>
              </div>
            )}
          </div>
        </div>
        {result && (
          <div className="mx-5 my-5">
            <div>
              {/* render music score using osmd */}
              <div className="place-items-center max-h-[50vh] overflow-y-scroll">
                <div ref={musicDisplayRef} />
              </div>
              <div className="my-4 flex">
                <div className="place-items-center text-center font-bold my-auto me-5">
                  Preview
                </div>
                <audio controls>
                  <source
                    src={
                      process.env.REACT_APP_SERVER_BASE_URL +
                      "style/input_score.wav"
                    }
                    type="audio/wav"
                  />
                  Your browser does not support the audio element.
                </audio>
              </div>
              <div>
                Music style:{" "}
                <span className="font-semibold">{currentStyle}</span>
              </div>
            </div>
          </div>
        )}
        <div className="mx-5 my-5">
          <div className="font-bold text-lg my-2 ms-1">Select Music Style</div>
          <div className="">
            <ul>
              {musicStyles && musicStyles.map((style, index) => {
                return (<StyleButton
                  style={style}
                  key={index}
                  description={styleDescription[index]}
                />);
              })}
            </ul>
          </div>
          <div>
            <Button
              size="lg"
              className="w-full px-10 my-5"
              onClick={() => {
                handleStyleTransfer();
              }}
              disabled={!result || transferLoading}
            >
              {transferLoading ? "Generating..." : "Style Transfer"}
            </Button>
          </div>
        </div>
        {transferResult && (
          <div className="mx-5 my-5">
            <div>
              {/* render music score using osmd */}
              <div className="place-items-center max-h-[50vh] overflow-y-scroll">
                <div ref={transferMusicDisplayRef} />
              </div>
              <div className="my-4 flex">
                <div className="place-items-center text-center font-bold my-auto me-5">
                  Preview
                </div>
                <audio controls>
                  <source
                    src={
                      process.env.REACT_APP_SERVER_BASE_URL +
                      "style/output_score.wav"
                    }
                    type="audio/wav"
                  />
                  Your browser does not support the audio element.
                </audio>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default StyleTransfer;
