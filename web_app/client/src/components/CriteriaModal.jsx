import React from "react";
import { useEffect, useState, useRef } from "react";
import axiosInstance from "../utils/axiosInstance";
import { Button } from "@material-tailwind/react";
import { Line } from "react-chartjs-2";
import { OpenSheetMusicDisplay } from "opensheetmusicdisplay";

const AnalysisChart = ({ dataset, name }) => {
  const data = {
    labels: dataset.xAxis.map((x) => x.toFixed(0)),
    datasets: [
      {
        type: "line",
        borderColor: "RGBA(54, 162, 235, 1)",
        borderWidth: 2,
        pointRadius: 0,
        fill: false,
        data: dataset.yAxis,
      },
    ],
  };

  const options = {
    scales: {
      x: {
        display: true,
        title: {
          display: true,
          text: "Time (s)",
        },
      },
      y: {
        display: true,
      },
    },
    plugins: {
      legend: {
        display: false,
      },
      title: {
        display: true,
        text: name,
      },
      tooltip: {
        enabled: true,
        mode: "index",
        intersect: false,
      },
    },
  };

  return <Line data={data} options={options} />;
};

// individual modal contents
const PitchAccuarcyContent = ({ recordingId }) => {
  const musicDisplayRef = useRef(null);
  const [osmd, setOsmd] = useState(null);

  useEffect(() => {
    const osmd = new OpenSheetMusicDisplay(musicDisplayRef.current, {
      autoResize: true,
      backend: "canvas",
      drawPartNames: false,
      drawComposer: false,
      drawTitle: false,
    });

    setOsmd(osmd);
    osmd
      .load(
        "http://localhost:3001/recordings/" +
          recordingId +
          "/output_score_LR.xml"
      )
      .then(() => {
        osmd.render();
      });
  }, [recordingId]);

  return (
    <div>
      <p>
        This metric evaluates how accurately you strike each note compared to
        its intended pitch. Your score will reflect how often your notes align
        with the exact pitch that the piece demands, ensuring you're not too
        sharp or flat. This helps in fine-tuning your intonation and contributes
        profoundly to the harmonic integrity of your performance.
      </p>
      <div className="max-h-[75vh] overflow-y-scroll">
        <div ref={musicDisplayRef}></div>
      </div>
    </div>
  );
};

const TemporayPrecisionContent = ({ recordingId }) => {
  const [plpData, setPlpData] = useState({
    xAxis: [],
    yAxis: [],
    verticalLines: [],
  });

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
      })
      .catch((error) => {
        console.error(error);
      });
  }, [recordingId]);

  if (plpData === null) {
    return <div>Loading...</div>;
  }

  const imagePath = `http://localhost:3001/recordings/${recordingId}/tempo_analysis.png`;
  return (
    <div>
      <p>
        Temporal precision measures the timing of your notes against the exact
        moments they're supposed to be played. A high score in this area means
        you're keeping in step with the rhythm and pacing that the composer
        intended, a skill crucial for maintaining the structural and rhythmic
        integrity of the piece.
      </p>
      <div className="w-full">
        <AnalysisChart dataset={plpData} name={"PLP"} />
      </div>
    </div>
  );
};

const DynamicConsistencyContent = ({ recordingId }) => {
  const [dynData, setDynData] = useState({
    xAxis: [],
    yAxis: [],
  });

  const [genDynData, setGenDynData] = useState({
    xAxis: [],
    yAxis: [],
  });

  useEffect(() => {
    axiosInstance
      .get(`/recordings/${recordingId}`)
      .then((response) => {
        const dynChart = response.data.charts.dynamics;
        setDynData({
          xAxis: dynChart.xAxis,
          yAxis: dynChart.yAxis,
        });

        const genDynChart = response.data.charts.generalDynamics;
        setGenDynData({
          xAxis: genDynChart.xAxis,
          yAxis: genDynChart.yAxis,
        });
      })
      .catch((error) => {
        console.error(error);
      });
  }, [recordingId]);

  return (
    <div>
      <p>
        Dynamic consistency assesses your ability to maintain the same
        'pressure' or volume across notes that should be played at an equal
        dynamic level. A high score in this area indicates that you're able to
        keep the intensity of your performance consistent, ensuring that the
        piece's emotional impact remains steady and unwavering.
      </p>
      <div className="w-full">
        <AnalysisChart dataset={dynData} name={"Dynamics"} />
      </div>

      <div className="w-full">
        <AnalysisChart dataset={genDynData} name={"General Dynamics"} />
      </div>
    </div>
  );
};

const DynamicRangeContent = ({ recordingId }) => {
  const [dynData, setDynData] = useState({
    xAxis: [],
    yAxis: [],
  });

  const [genDynData, setGenDynData] = useState({
    xAxis: [],
    yAxis: [],
  });

  useEffect(() => {
    axiosInstance
      .get(`/recordings/${recordingId}`)
      .then((response) => {
        const dynChart = response.data.charts.dynamics;
        setDynData({
          xAxis: dynChart.xAxis,
          yAxis: dynChart.yAxis,
        });

        const genDynChart = response.data.charts.generalDynamics;
        setGenDynData({
          xAxis: genDynChart.xAxis,
          yAxis: genDynChart.yAxis,
        });
      })
      .catch((error) => {
        console.error(error);
      });
  }, [recordingId]);

  return (
    <div>
      <p>
        Dynamic range measures your ability to express the full spectrum of
        volume in your performance. A high score in this area means you're
        effectively conveying the emotional depth and intensity of the piece
        through the contrast between soft and loud dynamics, adding depth and
        dimension to your performance.
      </p>
      <div className="w-full">
        <AnalysisChart dataset={genDynData} name={"General Dynamics"} />
      </div>

      <div className="w-full">
        <AnalysisChart dataset={dynData} name={"Dynamics"} />
      </div>
    </div>
  );
};

const TempoStabilityContent = ({ recordingId }) => {
  const [plpData, setPlpData] = useState({
    xAxis: [],
    yAxis: [],
    verticalLines: [],
  });

  const [tempoData, setTempoData] = useState({
    xAxis: [],
    yAxis: [],
  });

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

  if (plpData === null || tempoData === null) {
    return <div>Loading...</div>;
  }

  return (
    <div>
      <div className="my-4">
        Tempo stability measures how consistently you maintain the speed set for
        the piece, whether it's a slow, contemplative adagio or a brisk, lively
        allegro. Your score in this area indicates your control over pacing, an
        important factor in ensuring the piece flows smoothly and cohesively
        from start to finish.
      </div>
      <div className="w-full">
        <AnalysisChart dataset={tempoData} name={"Tempo"} />
      </div>
      <div className="w-full">
        <AnalysisChart dataset={plpData} name={"PLP"} />
      </div>
    </div>
  );
};

const PianoFingeringContent = ({ recordingId }) => {
  const musicDisplayRef = useRef(null);
  const [osmd, setOsmd] = useState(null);

  useEffect(() => {
    const osmd = new OpenSheetMusicDisplay(musicDisplayRef.current, {
      autoResize: true,
      backend: "canvas",
      drawPartNames: false,
      drawComposer: false,
      drawTitle: false,
    });

    setOsmd(osmd);
    osmd
      .load(
        "http://localhost:3001/recordings/" +
          recordingId +
          "/output_gen_score_LR.xml"
      )
      .then(() => {
        osmd.render();
      });
  }, [recordingId]);

  return (
    <div>
      <p>
        Piano fingering evaluates how effectively you navigate the keyboard,
        ensuring that you use the most efficient and comfortable fingerings for
        each note. A high score in this area indicates that you're able to
        execute complex passages with ease and precision, enhancing your
        performance's fluidity and technical proficiency.
      </p>
      <div className="max-h-[80vh] overflow-y-scroll">
        <div ref={musicDisplayRef}></div>
      </div>
    </div>
  );
};

// modal box to be overlayed on recording overview
const CriteriaModal = ({ criteriaName, onModalClose, recordingId }) => {
  // map criteria name to its content
  const criteriaContentMap = {
    "Pitch Accuracy": <PitchAccuarcyContent recordingId={recordingId} />,
    "Temporal Precision": (
      <TemporayPrecisionContent recordingId={recordingId} />
    ),
    "Dynamic Consistency": (
      <DynamicConsistencyContent recordingId={recordingId} />
    ),
    "Dynamic Range": <DynamicRangeContent recordingId={recordingId} />,
    "Tempo Stability": <TempoStabilityContent recordingId={recordingId} />,
    "Piano Fingering": <PianoFingeringContent recordingId={recordingId} />,
  };
  // compare [0:-2] char of criteriaName to criteriaContentMap
  const criteriaContent = criteriaContentMap[criteriaName.slice(0, -2)];

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto">
      <div className="flex items-center justify-center min-h-screen">
        <div className="fixed inset-0 transition-opacity">
          <div className="absolute inset-0 bg-black opacity-75"></div>
        </div>

        <div className="relative bg-white rounded-lg w-3/4 py-10 px-8">
          <div className="flex justify-between">
            <h2 className="text-xl font-bold">{criteriaName.slice(0, -2)}</h2>
            <Button className="" onClick={onModalClose}>
              CLOSE
            </Button>
          </div>
          <div className="mt-4">{criteriaContent}</div>
        </div>
      </div>
    </div>
  );
};

export default CriteriaModal;
