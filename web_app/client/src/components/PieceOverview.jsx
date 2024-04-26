import React from "react";
import { Link } from "react-router-dom";
import { useParams } from "react-router-dom";
import { useEffect, useState } from "react";

import socketIOClient from "socket.io-client";

import generateDateString from "../utils/utils";
import PieceHeader from "./PieceHeader";
import axiosInstance from "../utils/axiosInstance";
import NavbarWithMegaMenu from "./NavbarWithMegaMenu";
import AddRecordingModal from "./AddRecording";
import {
  Button,
  CardHeader,
  Typography,
  Card,
  CardBody,
  CardFooter,
} from "@material-tailwind/react";

import { ChevronDownIcon } from "@heroicons/react/24/solid";
// import { Doughnut } from "react-chartjs-2";
// import Chart from 'chart.js/auto';
import ScoreCard from "./ScoreCard";

// render the score of the recordings
const CircleScoreOld = ({ score }) => {
  // console.log("score:", score);

  // use state color
  const circle_color =
    score >= 75
      ? "text-green-700"
      : score >= 50
      ? "text-yellow-800"
      : "text-red-800";

  const text_color = score >= 75 ? "green" : score >= 50 ? "orange" : "red";

  const radius = 28;
  const stroke_width = 5.5;
  return (
    <div>
      <svg className="" viewBox="0 0 100 100">
        <circle
          class="text-gray-100 stroke-current"
          stroke-width={`${stroke_width}`}
          cx="50"
          cy="50"
          r={`${radius}`}
          fill="transparent"
        ></circle>

        {/* colored part */}
        <circle
          class={`${circle_color} progress-ring__circle stroke-current`}
          stroke-width={`${stroke_width}`}
          stroke-linecap="round" // or square
          cx="50"
          cy="50"
          r={`${radius}`}
          fill="transparent"
          // 300 (empty) -> full circle empty
          // 124 -> full circle filled
          stroke-dashoffset={`calc(300 - (176 * ${score} / 100))`}
        ></circle>

        <text
          x="50"
          y="52"
          className="font-bold text-xl"
          text-anchor="middle"
          alignment-baseline="middle"
          fill={`${text_color}`}
        >
          {score}
        </text>
      </svg>
    </div>
  );
};

// const CircleScore = ({ score }) => {
//   const data = {
//     datasets: [
//       {
//         data: [score, 100 - score],
//         backgroundColor: ["#10B981", "#F3F4F6"],
//         borderWidth: 0,
//       },
//     ],
//   };

//   const options = {
//     cutout: "82%",
//     maintainAspectRatio: false,
//     responsive: true,
//     events: ['click', 'touchstart'],
//     plugins: {
//       legend: {
//         display: false,
//       },
//       tooltip: {
//         enabled: false,
//       }
//     },
//     animation: {
//       duration: 0,
//     }
//   };

//   // Register the plugin if not already registered
//   const centerTextPlugin = {
//     id: 'centerTextPlugin',
//     beforeDraw: function(chart) {
//       const { ctx, chartArea: { top, right, bottom, left, width, height } } = chart;
//       ctx.save();

//       ctx.font = 'bold 34px Inter';
//       ctx.textBaseline = 'middle';
//       ctx.textAlign = 'center';
//       ctx.fillStyle = '#10B981';
//       const textX = Math.round((left + right) / 2);
//       const textY = Math.round((top + bottom) / 2);
//       ctx.fillText(score.toString(), textX, textY);
//       ctx.restore();
//     }
//   };

//   Chart.register(centerTextPlugin);

//   return (
//     <div className="m-8">
//       <Doughnut data={data} options={options} plugins={[centerTextPlugin]}/>
//     </div>

//   );
// };

// function ScoreCard({ name, score, overallComment, performanceId }) {
//   return (
//     <Card className="max-w-[20rem] m-2 flex flex-col">
//       <CardBody className="text-center flex flex-col flex-grow">
//         <div>
//           <CircleScore score={score} />
//         </div>
//         <div className="text-xl font-bold text-black py-2">{name}</div>

//         <div className="flex-grow">{overallComment}</div>
//       </CardBody>
//       <CardFooter className="">
//         <Link to={"/recordings/" + performanceId}>
//           <Button
//             fullWidth={true}
//             className="hover:scale-[1.02] focus:scale-[1.02] active:scale-100"
//             // color=""
//           >
//             Review
//           </Button>
//         </Link>
//       </CardFooter>
//     </Card>
//   );
// }

// card showing the score of the recordings
function ScoreCards({ cardItems }) {
  // console.log("scorecards loaded with items", cardItems);
  return (
    <div className="grid sm:grid-cols-2 lg:grid-cols-4 mx-6 my-4">
      {cardItems.map(({ name, score, overallComment, id }) => (
        <ScoreCard
          name={name}
          score={score}
          overallComment={overallComment}
          link={"/recordings/" + id}
          linkText="Review"
          key={id}
        />
      ))}
    </div>
  );
}

const HiddenRecording = ({ datetime }) => {
  const date = new Date(datetime);

  return (
    <div className="flex flex-col px-1">
      <Typography className="text-gray-500">
        {generateDateString(date)}
      </Typography>
    </div>
  );
};

function HiddenRecordings({ hiddenRecordingsDT = [] }) {
  const [isOpen, setIsOpen] = useState(false);

  const toggleDropdown = () => {
    setIsOpen(!isOpen);
  };

  return (
    <div className="flex flex-col w-full justify-center px-5 mt-2">
      <div className="flex flex-col justify-center ">
        <div
          className="flex cursor-pointer select-none items-center hover:bg-gray-100 py-2 px-1 rounded-md"
          onClick={toggleDropdown}
        >
          <Typography className="text-gray-700">
            Recordings in Analyzation Process
          </Typography>
          <ChevronDownIcon
            strokeWidth={2.5}
            className={`ml-2 h-4 w-4 transition-transform ${
              isOpen ? "rotate-180" : ""
            }`}
          />
        </div>
        {isOpen && (
          <div className="my-1">
            {hiddenRecordingsDT.map((recording) => (
              <HiddenRecording
                datetime={recording.datetime}
                key={recording.id}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function PieceOverview() {
  const { pieceId } = useParams();
  const [modalVisible, setModalVisible] = useState(false);
  const [hiddenRecordingsDT, setHiddenRecordingsDT] = useState();
  const [visibleRecordings, setVisibleRecordings] = useState(null);

  const fetchHiddenRecordings = async () => {
    // fetch hidden pieces
    axiosInstance
      .get(`/recordings/${pieceId}/hidden`)
      .then((response) => {
        console.log(response.data);
        // set list as a list of response.data.datetime
        setHiddenRecordingsDT(
          response.data.map((recording) => ({
            datetime: recording.datetime,
            id: recording.recordingId,
          }))
        );
      })
      .catch((error) => {
        console.error("error fetching hidden pieces:", error);
      });
  };

  const fetchVisibleRecordings = async () => {
    // fetch visible pieces
    axiosInstance
      .get(`/recordings/${pieceId}/visible`)
      .then((response) => {
        console.log(response.data);

        const resultList = response.data.map((recording) => ({
          name: generateDateString(new Date(recording.datetime)),
          score: recording.score.overall,
          overallComment: recording.comment.overall,
          id: recording.recordingId,
        }));

        setVisibleRecordings(resultList);
      })
      .catch((error) => {
        console.error("error fetching visible pieces:", error);
      });
  };

  // dummy recordings for testing
  const dummyFetchVisibleRecordings = () => {
    const tmpCardItems = [
      {
        name: "240214",
        score: 90,
        overallComment: "Nice speed, but need to work on dynamics.",
        performanceId: 1,
      },
      {
        name: "240210",
        score: 75,
        overallComment:
          "Good job, but need to work on tempo and pitch accuracy. Sometimes the notes are not clear.",
        performanceId: 2,
      },
      {
        name: "240208",
        score: 50,
        overallComment: "Nice speed, but need to work on dynamics.",
        performanceId: 3,
      },
      {
        name: "240201",
        score: 25,
        overallComment:
          "Spend more time on practicing and you will get better!",
        performanceId: 4,
      },
    ];
    setVisibleRecordings(tmpCardItems);
  };

  // run fetch hidden recording on load
  useEffect(() => {
    setTimeout(() => {
      fetchHiddenRecordings();
      fetchVisibleRecordings();
    }, 500);
  }, []);

  // use socket to receive reloadPage event
  useEffect(() => {
    const socket = socketIOClient(process.env.REACT_APP_SERVER_BASE_URL);

    socket.on("reloadPage", () => {
      // alert("reloadPage event received");
      fetchHiddenRecordings();
      fetchVisibleRecordings();
    });

    return () => socket.disconnect();
  }, []);

  const handleOpenModal = () => {
    setModalVisible(true);
  };

  const handleCloseModal = () => {
    setModalVisible(false);
    fetchHiddenRecordings();
  };

  return (
    <div>
      <NavbarWithMegaMenu />
      <PieceHeader handleOpenModal={handleOpenModal} />
      <HiddenRecordings hiddenRecordingsDT={hiddenRecordingsDT} />
      {visibleRecordings && <ScoreCards cardItems={visibleRecordings} />}
      {modalVisible && <AddRecordingModal onModalClose={handleCloseModal} />}
    </div>
  );
}
export default PieceOverview;
