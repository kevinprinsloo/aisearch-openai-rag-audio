import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Routes, Route } from "react-router-dom";

import { I18nextProvider } from "react-i18next";
import i18next from "./i18n/config";

import App from "./App.tsx";
import LocalVoiceRAG from "./LocalVoiceRAG.tsx";
import "./index.css";

createRoot(document.getElementById("root")!).render(
    <StrictMode>
        <I18nextProvider i18n={i18next}>
            <BrowserRouter>
                <Routes>
                    <Route path="/" element={<App />} />
                    <Route path="/local-voice-rag" element={<LocalVoiceRAG />} />
                </Routes>
            </BrowserRouter>
        </I18nextProvider>
    </StrictMode>
);
