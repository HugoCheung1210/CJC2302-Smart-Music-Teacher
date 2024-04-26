import React, { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import axiosInstance from "../utils/axiosInstance";
import { Button } from "@material-tailwind/react";
import { ArrowDownTrayIcon, PlusIcon } from "@heroicons/react/24/solid";

// TODO: recording modal
// header row showing information about the piece
function PieceHeader({ handleOpenModal }) {
  let { pieceId } = useParams();

  const [pieceInfo, setPieceInfo] = useState(null);

  // Fetch piece data from the server
  useEffect(() => {
    async function fetchPieceInformation() {
      try {
        const response = await axiosInstance.get("/pieces/" + pieceId);
        setPieceInfo(response.data);
        console.log(response.data);
      } catch (error) {
        console.error("error fetching piece information:", error);
      }
    }

    fetchPieceInformation();
  }, [pieceId]);

  // Check if pieceInfo is available before rendering
  if (!pieceInfo) {
    return <div>Loading...</div>;
  }

  return (
    <div>
      <div className="flex flex-row mx-6 mt-4 justify-between items-center">
        <div className="flex-grow">
          <div className="text-xl font-bold">{pieceInfo.name}</div>

          <div className=" text-gray-500 font-semibold">
            {pieceInfo.composer}
          </div>
        </div>

        <div className="flex flex-row">
          {pieceInfo.pdf_name && (
            <a
              href={"http://localhost:3001/scores/" + pieceInfo.pdf_name}
              target="_blank"
            >
              <Button className="flex items-center mx-2 my-4 px-3 gap-2">
                <ArrowDownTrayIcon className="h-4 w-4" /> PDF
              </Button>
            </a>
          )}
          <Button className="flex items-center mx-2 my-4 px-3 gap-2" onClick={handleOpenModal}>
            <PlusIcon className="h-4 w-4" /> Recording
          </Button>
        </div>
      </div>
      <div className="mx-6 text-sm">{pieceInfo.description}</div>
    </div>
  );
}

export default PieceHeader;
