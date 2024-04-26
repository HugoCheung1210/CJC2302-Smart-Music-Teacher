import Chart from "chart.js/auto";
import { Doughnut } from "react-chartjs-2";

import { Link } from "react-router-dom";

import {
  Card,
  CardBody,
  CardFooter,
  Button,
} from "@material-tailwind/react";

const CircleScore = ({ score }) => {
  // 70-100 is green, 30-69 is yellow, 0-29 is red
  let color;
  if (score >= 70) {
    color = "#10B981";
  } else if (score >= 40) {
    color = "#F59E0B";
  } else {
    color = "#EF4444";
  }

  const data = {
    datasets: [
      {
        data: [score, 100 - score],
        backgroundColor: [color, "#F3F4F6"],
        borderWidth: 0,
      },
    ],
  };

  const options = {
    cutout: "82%",
    maintainAspectRatio: false,
    responsive: true,
    events: ["click", "touchstart"],
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        enabled: false,
      },
    },
    animation: {
      duration: 0,
    },
  };

  // Register the plugin if not already registered
  const centerTextPlugin = {
    id: "centerTextPlugin",
    beforeDraw: function (chart) {
      const {
        ctx,
        chartArea: { top, right, bottom, left, width, height },
      } = chart;
      ctx.save();

      ctx.font = "bold 34px Inter";
      ctx.textBaseline = "middle";
      ctx.textAlign = "center";
      ctx.fillStyle = color;
      const textX = Math.round((left + right) / 2);
      const textY = Math.round((top + bottom) / 2);
      ctx.fillText(score.toString(), textX, textY);
      ctx.restore();
    },
  };

//   Chart.register(centerTextPlugin);

  return (
    <div className="m-8">
      <Doughnut data={data} options={options} plugins={[centerTextPlugin]} />
    </div>
  );
};

function ScoreCard({ name, score, overallComment, link, linkText }) {
    // console.log("scorecard loaded", name, score, overallComment, link, linkText);
  // alert("scorecard loaded");
  // round score to nearest integer
  const rounded_score = Math.round(score);
  
  return (
    <Card className="max-w-[20rem] m-2 flex flex-col">
      <CardBody className="text-center flex flex-col flex-grow">
        <div>
          <CircleScore score={rounded_score} />
        </div>
        <div className="text-xl font-bold text-black py-2">{name}</div>

        <div className="flex-grow">{overallComment}</div>
      </CardBody>
      <CardFooter className="">
        <Link to={link}>
          <Button
            fullWidth={true}
            className="hover:scale-[1.02] focus:scale-[1.02] active:scale-100"
          >
            {linkText}
          </Button>
        </Link>
      </CardFooter>
    </Card>
  );
}

export default ScoreCard;
