import React from "react";
import { Link } from "react-router-dom";
import { useParams } from "react-router-dom";
import { useEffect, useState } from "react";

import axiosInstance from "../utils/axiosInstance";
import RecordingHeader from "./RecordingHeader";
import NavbarWithMegaMenu from "./NavbarWithMegaMenu";
import {
  Button,
  Video,
  Typography,
  Card,
  CardBody,
  CardFooter,
} from "@material-tailwind/react";

// for polar chart
import { PolarArea } from "react-chartjs-2";
import "chart.js/auto";
import ChartDataLabels from "chartjs-plugin-datalabels";
import CriteriaModal from "./CriteriaModal";
import ScoreCard from "./ScoreCard";


// polar chart
const PolarAreaChart = ({ scores, onLegendClick }) => {
  const [chartData, setChartData] = useState(null);
  useEffect(() => {
    setTimeout(() => {
      console.log("scores", scores);
      setChartData({
        labels: [
          "Pitch Accuracy🔍",
          // "Temporal Precision🔍",
          "Dynamic Consistency🔍",
          // "Dynamic Range🔍",
          "Tempo Stability🔍",
          "Piano Fingering🔍",
        ],
        datasets: [
          {
            data: [
              scores.pitch_acc,
              // scores.temp_prec,
              scores.dyn_cons,
              // scores.dyn_range,
              scores.tempo_stab,
              scores.finger
            ],
            borderWidth: 1,
          },
        ],
      });
    }, 0);
  }, [scores]);

  const options = {
    scales: {
      r: {
        ticks: {
          display: false, // hide axis label
        },
        startAngle: 270,
        max: 100,
      },
    },
    redraw: true,
    plugins: {
      datalabels: {
        display: true,
        color: "black",
        align: "end",
        anchor: "center",
        font: {
          size: 16,
          weight: "bold",
        },
        formatter: (value, context) => {
          // round value to integer
          return Math.round(value);
        },
      },
      legend: {
        onClick: (e, legendItem, legend) => {
          // override default action
          onLegendClick(legendItem.text);
        },
      },
    },
  };

  return (
    <div className="w-3/5 mx-auto">
      {chartData && (
        <PolarArea
          data={chartData}
          options={options}
          plugins={[ChartDataLabels]}
        />
      )}
    </div>
  );
};

// polar chart and related detail comments
const PerformanceChart = ({ onLegendClick, recordingInfo }) => {
  // const rendered = false;
  // setTimeout(() => {
  //   console.log("recordingInfo", recordingInfo);
  // }, 0);
  return ( 
    recordingInfo &&
    <div className="text-center w-full">
      <div className="text-xl font-bold">Detailed Scoring</div>
      <div className="flex justify-center my-4">
        <PolarAreaChart 
          scores={recordingInfo ? recordingInfo.score : ""}
          onLegendClick={onLegendClick} 
        />
      </div>
      <div className="my-3 mx-5">
        {recordingInfo ? recordingInfo.comment.pitch : ""}
        <br />
        {recordingInfo ? recordingInfo.comment.tempo : ""}
        <br />
        {recordingInfo ? recordingInfo.comment.dynamics : ""}
        <br />
        {recordingInfo ? recordingInfo.comment.finger : ""}
        <br />
      </div>
    </div>
  );
};

function RecordingOverview() {
  let { recordingId } = useParams();

  const [recordingInfo, setRecordingInfo] = useState(null);

  // fetch recording info
  useEffect(() => {
    axiosInstance
        .get(`/recordings/${recordingId}`)
        .then((response) => {
          setRecordingInfo(response.data);
        })
        .catch((error) => {
          console.log(error);
        });
    
  }, []);

  const [modalVisible, setModalVisible] = useState(false);
  const [selectedLegendItem, setSelectedLegendItem] = useState(null);

  const handleLegendClick = (legendItem) => {
    setSelectedLegendItem(legendItem);
    setModalVisible(true);
  };

  const handleCloseModal = () => {
    setModalVisible(false);
    setSelectedLegendItem(null); // Reset the selected legend item
  };

  return (
    <div>
      <NavbarWithMegaMenu />
      <RecordingHeader />

      <div className="m-4">
        {modalVisible && (
          <CriteriaModal
            criteriaName={selectedLegendItem}
            onModalClose={handleCloseModal}
            recordingId={recordingId}
          />
        )}
        <div className="grid lg:grid-cols-2">
          <div className="place-items-center align-middle p-5 mx-auto">
            {recordingInfo && <ScoreCard
              name={""}
              score={recordingInfo ? recordingInfo.score.overall : ""}
              overallComment={
                recordingInfo ? recordingInfo.comment.overall : ""
              }
              link={`/playback/${recordingId}`}
              linkText={"Playback"}
            />}
          </div>

          <PerformanceChart
            onLegendClick={handleLegendClick}
            recordingInfo={recordingInfo}
          />
        </div>
      </div>
    </div>
  );
}

export default RecordingOverview;
