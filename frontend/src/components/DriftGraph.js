import React, { useEffect, useState } from "react";
import axios from "axios";
import {
  Chart as ChartJS,
  LineElement,
  PointElement,
  CategoryScale,
  LinearScale,
  Tooltip,
  Legend,
} from "chart.js";
import { Line } from "react-chartjs-2";

ChartJS.register(
  LineElement,
  PointElement,
  CategoryScale,
  LinearScale,
  Tooltip,
  Legend
);

function DriftGraph() {
  const [labels, setLabels] = useState([]);
  const [scores, setScores] = useState([]);

  const fetchDriftData = async () => {
    try {
      const res = await axios.get("http://127.0.0.1:5000/drift-data");
      setLabels(res.data.labels);
      setScores(res.data.scores);
    } catch (err) {
      console.error("Drift graph fetch failed", err);
    }
  };

  // 🔁 AUTO REFRESH (every 3 seconds)
  useEffect(() => {
    fetchDriftData();
    const interval = setInterval(fetchDriftData, 3000);
    return () => clearInterval(interval);
  }, []);

  const data = {
    labels,
    datasets: [
      {
        label: "Emotional Drift (Last 7 Days)",
        data: scores,
        borderColor: "#6C63FF",
        backgroundColor: "rgba(108,99,255,0.15)",
        tension: 0.4,
        pointRadius: 5,
        pointHoverRadius: 7,
        fill: true,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      y: {
        min: -1,
        max: 1,
        title: {
          display: true,
          text: "Emotion Polarity",
        },
      },
      x: {
        title: {
          display: true,
          text: "Date",
        },
      },
    },
    plugins: {
      legend: {
        display: true,
        position: "top",
      },
    },
  };

  return (
    <div className="card graph-card">
      <h3> Emotional Drift (Last 7 Days)</h3>
      <div style={{ height: "320px" }}>
        <Line data={data} options={options} />
      </div>
    </div>
  );
}

export default DriftGraph;
