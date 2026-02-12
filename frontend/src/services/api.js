import axios from "axios";

const BASE_URL = "http://127.0.0.1:5000";

/* ---------- DAILY ---------- */
export const submitDailyText = async (text) => {
  const formData = new FormData();
  formData.append("text", text);

  const res = await axios.post(`${BASE_URL}/submit-daily`, formData);
  return res.data;
};

/* ---------- WEEKLY REPORT ---------- */
export const getWeeklyReport = async () => {
  const res = await axios.get(`${BASE_URL}/weekly-report`);
  return res.data;
};

/* ---------- DRIFT GRAPH ---------- */
export const getDriftGraphUrl = () => {
  return `${BASE_URL}/drift-graph`;
};

/* ---------- ✅ WEEKLY PDF DOWNLOAD ---------- */
export const downloadWeeklyPDF = async () => {
  const res = await axios.get(
    `${BASE_URL}/weekly-report-pdf`,
    { responseType: "blob" }
  );

  const url = window.URL.createObjectURL(new Blob([res.data]));
  const link = document.createElement("a");
  link.href = url;
  link.setAttribute("download", "Weekly_Mental_Health_Report.pdf");
  document.body.appendChild(link);
  link.click();
  link.remove();
};
