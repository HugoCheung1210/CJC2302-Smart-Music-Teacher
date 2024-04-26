import React from "react";
import { useParams } from "react-router-dom";
import { useEffect, useRef, useState } from "react";
import ReactPlayer from "react-player";
import { OpenSheetMusicDisplay } from "opensheetmusicdisplay";
import { Button } from "@material-tailwind/react";

import { Line } from "react-chartjs-2";
import "chartjs-plugin-annotation";
import annotationPlugin from "chartjs-plugin-annotation";
import zoomPlugin from "chartjs-plugin-zoom";
import { Chart as ChartJS, registerables } from "chart.js";
import { ArrowDownTrayIcon } from "@heroicons/react/24/solid";

import axiosInstance from "../utils/axiosInstance";
import RecordingHeader from "./RecordingHeader";
import NavbarWithMegaMenu from "./NavbarWithMegaMenu";

ChartJS.register(...registerables, annotationPlugin, zoomPlugin);

// the analysis chart below the video player
const AnalysisChart = ({ currentVideoTime }) => {
  let { recordingId } = useParams();
  const [plpData, setPlpData] = useState({
    xAxis: [],
    yAxis: [],
    verticalLines: [],
  });
  const [genDynData, setGenDynData] = useState({
    xAxis: [],
    yAxis: [],
  });
  const [dynData, setDynData] = useState({
    xAxis: [],
    yAxis: [],
  });
  const [onsetData, setOnsetData] = useState({
    xAxis: [],
    yAxis: [],
    verticalLines: [],
  });
  const [tempoData, setTempoData] = useState({
    xAxis: [],
    yAxis: [],
  });

  // get data from server upon loading
  useEffect(() => {
    axiosInstance
      .get(`/recordings/${recordingId}`)
      .then((response) => {
        const PlpChart = response.data.charts.PLP;
        setPlpData({
          xAxis: PlpChart.xAxis,
          yAxis: PlpChart.yAxis,
          verticalLines: PlpChart.verticalLines,
        });

        const DynChart = response.data.charts.dynamics;
        setDynData({
          xAxis: DynChart.xAxis,
          yAxis: DynChart.yAxis,
        });

        const GenDynChart = response.data.charts.generalDynamics;
        setGenDynData({
          xAxis: GenDynChart.xAxis,
          yAxis: GenDynChart.yAxis,
        });

        const OnsetChart = response.data.charts.onset;
        setOnsetData({
          xAxis: OnsetChart.xAxis,
          yAxis: OnsetChart.yAxis,
          verticalLines: OnsetChart.verticalLines,
        });

        const TempoChart = response.data.charts.tempo;
        setTempoData({
          xAxis: TempoChart.xAxis,
          yAxis: TempoChart.yAxis,
        });
      })
      .catch((error) => {
        console.error(error);
      });
  }, [recordingId]);

  // setup vertical line data format for showing onset
  const onsetLines = Array(plpData.xAxis.length).fill(0);
  onsetData.verticalLines.forEach((line) => {
    // round line to nearest 0.1
    const index = Math.round(line * 10);
    if (index >= onsetLines.length) {
      onsetLines[onsetLines.length - 1] = 1;
    } else onsetLines[index] = 1;
  });

  const labels = plpData.xAxis.map((x) => x.toFixed(0));

  const data = {
    labels: labels,
    datasets: [
      {
        yAxisID: "yDyn",
        type: "line",
        label: "Dynamics",
        borderColor: "rgb(54, 162, 235)",
        borderWidth: 2,
        pointRadius: 0,
        fill: false,
        data: dynData.yAxis,
      },
      {
        yAxisID: "yTempo",
        type: "line",
        label: "Tempo",
        borderColor: "rgb(255, 99, 132)",
        borderWidth: 2,
        pointRadius: 0,
        fill: false,
        data: tempoData.yAxis,
      },
      {
        xAxisID: "xOnset",
        yAxisID: "yOnset",
        type: "bar",
        label: "Onset",
        borderColor: "rgba(75, 192, 192, 0.5)",
        backgroundColor: "rgba(75, 192, 192, 0.5)",
        borderWidth: 3,
        barThickness: 2,
        fill: true,
        data: onsetLines,
      },
    ],
  };

  const options = {
    responsive: true,

    scales: {
      x: {
        display: true,
        title: {
          display: true,
          text: "Time (s)",
        },
        grid: {
          display: true,
        },
        // ticks: {
        //   callback: function(val, index) {
        //     return Number(val) % 10 === 0 ? val / 10 : '';
        //   },
        //   fontStyle: "normal",
        // }
      },
      // disable onset axis
      xOnset: {
        display: false,
      },
      yOnset: {
        display: false,
      },
      yTempo: {
        display: false,
      },
      yDyn: {
        display: false,
      },
    },
    // plot vertical lines for playback
    plugins: {
      annotation: {
        annotations: {
          line1: {
            type: "line",
            scaleID: "x",
            value: currentVideoTime * 10 + 2,
            borderColor: "red",
            borderWidth: 2,
          },
        },
      },
      tooltip: {
        mode: "index",
        intersect: false,
        position: "nearest",
        callbacks: {
          title: function (tooltipItems) {
            const dataIndex = tooltipItems[0].dataIndex;
            // console.log("dataIndex", dataIndex);
            const xValue = dataIndex / 10.0;
            const formattedXValue = parseFloat(xValue).toFixed(1); // Format to 1 decimal place
            return `Time: ${formattedXValue}s`;
          },
          label: function (context) {
            let label = context.dataset.label || "";

            if (label) {
              label += ": ";
            }
            if (context.parsed.y !== null) {
              // round to 2 decimal places
              label += Math.round(context.parsed.y * 100) / 100;
            }
            return label;
          },
        },
      },
      zoom: {
        pan: {
          enabled: true,
          mode: "x",
        },
        zoom: {
          wheel: {
            enabled: true,
          },
          pinch: {
            enabled: true,
          },
          mode: "x",
        },
      },
    },
  };

  return (
    <div className="w-full">
      {/* <div>&nbsp;</div> */}
      <Line data={data} options={options} />
    </div>
  );
};

const MusicVideoPlayer = ({ videoUrl, musicXmlUrl }) => {
  const videoRef = useRef(null);
  const musicDisplayRef = useRef(null);
  const chartRef = useRef(null);
  const [osmd, setOsmd] = useState(null);
  const [currentVideoTime, setCurrentVideoTime] = useState(0);

  useEffect(() => {
    const osmd = new OpenSheetMusicDisplay(musicDisplayRef.current, {
      autoResize: true,
      backend: "canvas",
      drawPartNames: false,
      drawComposer: false,
      drawTitle: false,
    });

    osmd.load(musicXmlUrl).then(() => {
      osmd.render();
      osmd.cursor.show();
      setOsmd(osmd);
    });
  }, [musicXmlUrl]);

  // check progress bar and update playback
  useEffect(() => {
    const interval = setInterval(() => {
      if (videoRef.current) {
        const currentTime = videoRef.current.getCurrentTime();
        setCurrentVideoTime(currentTime);
      }

      if (chartRef.current) {
        chartRef.current.chartInstance.update();
      }
    }, 100);
    return () => clearInterval(interval);
  }, [videoRef, chartRef]);

  return (
    <div>
      <div className="m-4">
        {/* <div className="grid lg:grid-cols-2"> */}
        <div className="flex lg:flex-row flex-col">
          <div className="place-items-center overflow-y-auto flex-1 lg:max-w-[48vw] max-h-[50vh] lg:max-h-[80vh] ">
            <div ref={musicDisplayRef} />
          </div>

          <div className=" place-items-center mx-auto flex-1 lg:max-w-[43vw]">
            <div className="  ">
              <ReactPlayer
                className="mx-auto "
                width="100%"
                height="100%"
                ref={videoRef}
                url={videoUrl}
                controls
                // onProgress={onProgress}
              />
            </div>
            <div className="mx-auto">
              <AnalysisChart currentVideoTime={currentVideoTime} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

function Playback() {
  let { recordingId } = useParams();
  const [scoreUrl, setScoreUrl] = useState(null);

  useEffect(() => {
    // check if url exist on server
    const score_url = process.env.REACT_APP_SERVER_BASE_URL + "recordings/" + recordingId + "/output_score_LR.xml";

    axiosInstance
    .get(score_url)
    .then((response) => {
      console.log("Score URL exists");
      setScoreUrl(score_url);
    })
    .catch((error) => {
      console.log("Score URL does not exist");
      setScoreUrl(process.env.REACT_APP_SERVER_BASE_URL + "recordings/" + recordingId + "/output_score.xml");
    });
  });

  return ( scoreUrl && 
    <div>
      <NavbarWithMegaMenu />
      <div className="flex flex-row mx-8 my-2">
        <div className="flex-grow">
          <RecordingHeader />
        </div>

        <a href={process.env.REACT_APP_SERVER_BASE_URL + "recordings/" + recordingId + "/output.mid"}>
          <Button className="flex items-center mx-2 my-4 px-3 gap-2">
            <ArrowDownTrayIcon className="h-4 w-4" /> Playback MIDI
          </Button>
        </a>

        <div className="flex flex-row">
          <a href={"http://localhost:3000/recordings/" + recordingId}>
            <Button className="flex items-center mx-2 my-4 px-5 gap-2">
              Back
            </Button>
          </a>
        </div>
      </div>

      <div>
        <MusicVideoPlayer
          videoUrl={
            process.env.REACT_APP_SERVER_BASE_URL + "recordings/" + recordingId + "/raw_video.mp4"
          }
          musicXmlUrl={
            scoreUrl
          }
        />
      </div>
    </div>
  );
}

export default Playback;
