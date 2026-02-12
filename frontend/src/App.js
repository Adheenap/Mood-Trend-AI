import React, { useState } from "react";
import "./App.css";

import { submitDailyText, getWeeklyReport } from "./services/api";

import DailyCard from "./components/DailyCard";
import WeeklyReportCard from "./components/WeeklyReportCard";
import DriftGraph from "./components/DriftGraph";

function App() {
  // ✅ STATE DEFINITIONS (THIS FIXES THE ERROR)
  const [text, setText] = useState("");
  const [dailyData, setDailyData] = useState(null);
  const [weeklyData, setWeeklyData] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async () => {
    if (!text.trim()) return;

    try {
      setLoading(true);

      const daily = await submitDailyText(text);
      setDailyData(daily);

      const weekly = await getWeeklyReport();
      setWeeklyData(weekly);

      setText("");
    } catch (err) {
      console.error("Submission failed", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <h1> <center> MINDTRACE-AI</center></h1>

      <div className="input-card">
        <textarea
          placeholder="How are you feeling today?"
          value={text}
          onChange={(e) => setText(e.target.value)}
        />
        <button onClick={handleSubmit} disabled={loading}>
          {loading ? "Analyzing..." : "SUBMIT"}
        </button>
      </div>

      <div className="dashboard-row">
        {dailyData && <DailyCard data={dailyData} />}
        <DriftGraph />
      </div>

      {weeklyData && <WeeklyReportCard data={weeklyData} />}
    </div>
  );
}

export default App;
