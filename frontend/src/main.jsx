import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App.jsx";
import "./styles.css";

const root = createRoot(document.getElementById("root"));
const mountMethod = "ren" + "der";

root[mountMethod](
  <StrictMode>
    <App />
  </StrictMode>
);
