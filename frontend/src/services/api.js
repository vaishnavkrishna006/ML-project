import axios from "axios";

const api = axios.create({
  baseURL: "http://127.0.0.1:5000/api",
  timeout: 10000,
});

api.interceptors.response.use(
  (res) => res,
  (err) => {
    console.error("API error:", err?.message || err);
    return Promise.reject(err);
  }
);

export default api;
