import { downloadWeeklyPDF } from "../services/api";

function WeeklyReportCard({ data }) {
  return (
    <div className="card">
      <h3>📅 Weekly Mental Health Report</h3>

      <p><strong>Average Score:</strong> {data.average_score}</p>
      <p><strong>Mood:</strong> {data.mood}</p>
      <p>{data.message}</p>

      <button
        onClick={downloadWeeklyPDF}
        style={{ marginTop: "10px" }}
      >
        📄 Download Weekly PDF
      </button>
    </div>
  );
}

export default WeeklyReportCard;
