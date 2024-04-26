import React, {useEffect, useState} from "react";
import { useParams } from "react-router-dom";
import axiosInstance from "../utils/axiosInstance";
import generateDateString from "../utils/utils";


function RecordingHeader() {
  let { recordingId } = useParams();
  const [recordingInfo, setRecordingInfo] = useState(null);
  const [pieceInfo, setPieceInfo] = useState(null);

  useEffect(() => {
    // Fetch recording data from the server
    axiosInstance.get("/recordings/" + recordingId)
      .then(response => {
        const recordingInfo = response.data;
        setRecordingInfo(recordingInfo);
  
        // Fetch piece data from the server
        return axiosInstance.get("/pieces/" + recordingInfo.pieceId);
      })
      .then(response => {
        const pieceInfo = response.data;
        setPieceInfo(pieceInfo);
        console.log(pieceInfo);
      })
      .catch(error => {
        console.error("error fetching information:", error);
      });
  }, [recordingId]);

  return (
    <div className="flex flex-row mx-6 mt-4 justify-between items-center">
      <div className="flex-grow">
        <div className="text-xl font-bold">{pieceInfo ? pieceInfo.name : ""}</div>

        <div className=" text-gray-500 font-semibold">{recordingInfo ? generateDateString(recordingInfo.datetime) : ""}</div>

        <div className="hidden">Extra description</div>
      </div>
    </div>
  );
}

export default RecordingHeader;
