App.js
// src/App.js
import React, { useState, useEffect } from "react";
import axios from "axios";

import FileUpload from "./components/FileUpload";
import MiniSidebar from "./components/MiniSidebar";
import RedlinePreview from "./components/RedlinePreview";

import "./index.css";
import "./App.css";

const HISTORY_KEY = "contract_review_history_v1";

function App() {
  const [file, setFile] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [appliedSuggestions, setAppliedSuggestions] = useState(new Set());
  const [history, setHistory] = useState([]);
  const [generating, setGenerating] = useState(false);

  useEffect(() => {
    const rawHistory = localStorage.getItem(HISTORY_KEY);
    if (rawHistory) {
      try {
        setHistory(JSON.parse(rawHistory));
      } catch {
        setHistory([]);
      }
    }
  }, []);

  const handleFileChange = (selectedFile, localPath = null) => {
    setFile(selectedFile);
    setAnalysis(null);
    setError("");
    setAppliedSuggestions(new Set());
    if (localPath && selectedFile) selectedFile.localPath = localPath;
  };

  const handleAnalyze = async () => {
    if (!file) {
      setError("Please upload a contract PDF or DOCX first.");
      return;
    }

    setLoading(true);
    setError("");
    setAppliedSuggestions(new Set());

    try {
      let res;

      if (file.localPath) {
        res = await axios.post(
          "http://127.0.0.1:8000/redline/convert/",
          { path: file.localPath },
          {
            headers: { "Content-Type": "application/json" },
            timeout: 300000,
          }
        );
      } else {
        const formData = new FormData();
        formData.append("file", file);
        res = await axios.post(
          "http://127.0.0.1:8000/redline/convert/",
          formData,
          {
            headers: { "Content-Type": "multipart/form-data" },
            timeout: 300000,
          }
        );
      }

      processConvertResponse(res.data, file.name);
    } catch (err) {
      console.error("Convert error:", err);
      setError(
        err.response?.data?.detail ||
          err.message ||
          "Could not convert contract."
      );
    } finally {
      setLoading(false);
    }
  };

  const processConvertResponse = (data, filename) => {
    const normalized = (data.clauses || []).map((c, i) => ({
      clause_id: c.clause_id ?? `c${i + 1}`,
      original_clause: c.original_clause ?? "",
      suggested_clause: c.suggested_clause ?? "",
      risk_level: c.risk_level ?? "Low",
      highlight_color: c.highlight_color ?? "green",
      explanation: c.explanation ?? "",
      applied: false,
    }));

    const newAnalysis = {
      html: data.html || "",
      clauses: normalized,
      summary: data.summary || {},
      download_link: data.download_link || null,
      // server will return original_docx when possible (absolute path on server)
      original_docx: data.original_docx || null,
    };

    setAnalysis(newAnalysis);

    const entry = {
      id: Date.now().toString(),
      name: filename,
      ts: new Date().toISOString(),
      summary: newAnalysis.summary,
      clauses: normalized,
      download_link: data.download_link || null,
      localPath: file?.localPath || null,
      html: data.html,
      original_docx: data.original_docx || null,
    };

    const newHistory = [entry, ...history].slice(0, 50);
    setHistory(newHistory);
    localStorage.setItem(HISTORY_KEY, JSON.stringify(newHistory));
  };

  const applySuggestion = (clause_id, newText) => {
    if (!analysis?.clauses) return;

    const updated = analysis.clauses.map((c) =>
      c.clause_id === clause_id ? { ...c, original_clause: newText, applied: true } : c
    );

    setAnalysis({ ...analysis, clauses: updated });
    setAppliedSuggestions((prev) => new Set([...prev, clause_id]));
  };

  const applyAllSuggestions = () => {
    if (!analysis?.clauses) return;

    const risky = analysis.clauses.filter(
      (c) =>
        (c.risk_level === "High" || c.risk_level === "Medium") &&
        !appliedSuggestions.has(c.clause_id)
    );

    const updated = analysis.clauses.map((c) =>
      risky.some((r) => r.clause_id === c.clause_id)
        ? { ...c, original_clause: c.suggested_clause, applied: true }
        : c
    );

    setAnalysis({ ...analysis, clauses: updated });
    setAppliedSuggestions(
      (prev) => new Set([...prev, ...risky.map((c) => c.clause_id)])
    );

    // notify DOM preview to swap text + animate
    window.dispatchEvent(new CustomEvent("applyAllRedlines"));
  };

  const appliedCount = appliedSuggestions.size;
  const totalRiskyClauses =
    analysis?.clauses?.filter(
      (c) => c.risk_level === "High" || c.risk_level === "Medium"
    ).length || 0;

  const generateEditedDoc = async () => {
    if (!analysis) return;
    setGenerating(true);
    setError("");

    try {
      const payload = {
        filename: (file?.name || `edited_${Date.now()}`).replace(/\s+/g, "_"),
        clauses: analysis.clauses.map((c) => ({
          clause_id: c.clause_id,
          original_clause: c.original_clause,
          suggested_clause: c.suggested_clause,
          risk_level: c.risk_level,
        })),
        // pass original_docx (server path) when available to preserve formatting
        original_docx: analysis.original_docx || null,
      };

      const res = await axios.post(
        "http://127.0.0.1:8000/redline/generate/",
        payload,
        {
          headers: { "Content-Type": "application/json" },
          timeout: 300000,
        }
      );

      const download_link = res.data?.download_link;
      if (!download_link) {
        throw new Error("No download link returned.");
      }

      const finalURL = download_link.startsWith("http")
        ? download_link
        : `http://127.0.0.1:8000${download_link}`;

      window.open(finalURL, "_blank");
    } catch (err) {
      console.error("Generate edited doc error:", err);
      setError(
        err.response?.data?.detail || err.message || "Could not generate edited DOCX."
      );
    } finally {
      setGenerating(false);
    }
  };

  const loadHistoryEntry = (entryId) => {
    const entry = history.find((h) => h.id === entryId);
    if (!entry) return;

    setAnalysis({
      html: entry.html,
      clauses: entry.clauses,
      summary: entry.summary,
      download_link: entry.download_link,
      original_docx: entry.original_docx || null,
    });

    setFile(null);
  };

  const deleteHistoryItem = (entryId) => {
    const newHistory = history.filter((h) => h.id !== entryId);
    setHistory(newHistory);
    localStorage.setItem(HISTORY_KEY, JSON.stringify(newHistory));
    setAnalysis(null);
  };

  return (
    <div className="min-h-screen">
      <div className="max-w-7xl mx-auto app-root">
        <div className="app-header">
          <div className="bg-gradient-to-r from-neon to-purple-700 py-4 px-6 rounded-xl app-glass fade-in-up" style={{ width: "100%" }}>
            <h1 className="text-3xl font-bold" style={{ color: "white", letterSpacing: "-0.4px" }}>⚖️ AI Contract Review & Redlining</h1>
            <p className="app-subtitle">Upload contracts, review redlines and apply playbook suggestions — now with a neon interface ✨</p>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 space-y-6">
            <div className="app-glass p-6 fade-in-up">
              <FileUpload
                onFileChange={handleFileChange}
                onReset={() => { setFile(null); setAnalysis(null); }}
                file={file}
              />

              {error && <div className="mt-4 p-3 bg-red-900/30 rounded text-sm"><strong>Error:</strong> {error}</div>}

              <div className="mt-4 flex gap-3 items-center">
                <button onClick={handleAnalyze} disabled={loading} className="btn-neon pulse-on-hover">
                  {loading ? "Converting..." : "🔍 Analyze & Convert"}
                </button>

                <button onClick={applyAllSuggestions} disabled={appliedCount >= totalRiskyClauses} className="btn-ghost">
                  Apply All ({appliedCount}/{totalRiskyClauses})
                </button>

                <button onClick={generateEditedDoc} disabled={!analysis || generating} className="btn-neon" style={{ background: "linear-gradient(90deg,#8b4bff,#c08bff)" }}>
                  {generating ? "Generating..." : "⬇ Download Edited DOCX"}
                </button>
              </div>
            </div>

            {analysis && (
              <div className="app-glass p-4 fade-in-up preview-shell">
                <RedlinePreview analysis={analysis} onApplySuggestion={applySuggestion} />
              </div>
            )}
          </div>

          <aside className="space-y-6">
            <div className="app-glass p-4 fade-in-up">
              <MiniSidebar summary={analysis?.summary || {}} />
            </div>

            <div className="app-glass p-4">
              <h3 className="text-sm text-muted">Analysis History</h3>
              <div className="mt-3 history-list">
                {history.length === 0 && <div className="text-muted">No past analyses yet.</div>}
                {history.map((h) => (
                  <div key={h.id} className="p-3 history-item app-glass">
                    <div className="flex items-center justify-between">
                      <div>
                        <div className="file-name">{h.name}</div>
                        <div className="text-muted">{new Date(h.ts).toLocaleString()}</div>
                      </div>

                      <div className="flex items-center gap-2">
                        {h.download_link && (
                          <a className="btn-neon" href={h.download_link.startsWith("http") ? h.download_link : `http://127.0.0.1:8000${h.download_link}`} target="_blank" rel="noreferrer">Download</a>
                        )}
                        <button onClick={() => loadHistoryEntry(h.id)} className="btn-ghost">Open</button>
                        <button onClick={() => deleteHistoryItem(h.id)} className="btn-ghost">Delete</button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

          </aside>
        </div>

      </div>
    </div>
  );
}

export default App;
