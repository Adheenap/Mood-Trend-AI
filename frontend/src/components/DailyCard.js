function DailyCard({ data }) {
  return (
    <div className="card">
      <h3> Daily Mental Health</h3>
      <p>Date: {data.date}</p>
      <p>Mood: {data.mood}</p>
      <p>Score: {data.score}</p>
      <p>{data.message}</p>
    </div>
  );
}

export default DailyCard;
