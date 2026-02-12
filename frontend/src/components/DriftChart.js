import React, { useEffect, useState } from "react";
import {
  Chart as ChartJS,
  LineElement,
  PointElement,
  CategoryScale,
  LinearScale,
  Tooltip,
  Legend
} from "chart.js";
import { Line } from "react-chartjs-2";
import axios from "axios";

ChartJS.register(
  LineElement,
  PointElement,
  CategoryScale,
  LinearScale,
  Tooltip,
  Legend
);

function DriftChart() {
  const [chartData, setChartData] = useState(null);

  useEffect(() => {
    axios.get("http://127.0.0.1:5000/drift-data")
      .then(res => {
        setChartData({
          labels: res.data.labels,
          datasets: [
            {
              label: "Emotion Drift Score",
              data: res.data.scores,
              borderColor: "#6a5acd",
              backgroundColor: "rgba(106,90,205,0.2)",
              tension: 0.4,
              pointRadius: 4
            }
          ]
        });
      })
      .catch(() => setChartData(null));
  }, []);

  if (!chartData) {
    return <p> No drift data available yet</p>;
  }

  return (
    <div className="card">
      <h3> Live Emotion Drift</h3>
      <Line data={chartData} />
    </div>
  );
}

export default DriftChart;
