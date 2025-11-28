// src/components/RedlinePreview.jsx
import React, { useEffect, useRef, useState } from "react";

export default function RedlinePreview({ analysis, onApplySuggestion }) {
  const containerRef = useRef(null);
  const [localHtml, setLocalHtml] = useState("");
  const [modal, setModal] = useState(null);

  useEffect(() => {
    if (!analysis) return setLocalHtml("");
    setLocalHtml(analysis.html || buildHtmlFromClauses(analysis));
  }, [analysis]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const attach = () => {
      const riskySpans = container.querySelectorAll(".risky[data-clause-id]");
      riskySpans.forEach((sp) => {
        if (sp.__attached) return;
        sp.__attached = true;
        sp.style.cursor = "pointer";
        sp.addEventListener("click", () => {
          const id = sp.getAttribute("data-clause-id");
          openModalForClause(id);
        });
      });
    };

    attach();
    const mo = new MutationObserver(() => attach());
    mo.observe(container, { childList: true, subtree: true });

    const applyAllHandler = () => {
      const spans = container.querySelectorAll(".risky[data-clause-id]");
      spans.forEach((sp) => {
        const id = sp.getAttribute("data-clause-id");
        const clause = (analysis?.clauses || []).find((c) => c.clause_id === id);
        if (!clause) return;

        sp.innerText = clause.suggested_clause || clause.original_clause;
        sp.classList.remove("risky");
        sp.removeAttribute("data-clause-id");
        sp.removeAttribute("data-risk");
        sp.style.textDecoration = "none";

        onApplySuggestion(id, clause.suggested_clause || clause.original_clause);
      });
    };

    window.addEventListener("applyAllRedlines", applyAllHandler);

    return () => {
      mo.disconnect();
      window.removeEventListener("applyAllRedlines", applyAllHandler);
    };
  }, [localHtml, analysis, onApplySuggestion]);

  const openModalForClause = (clauseId) => {
    const c = analysis?.clauses?.find((cl) => cl.clause_id === clauseId);
    if (!c) return;

    setModal({
      clause_id: clauseId,
      original: c.original_clause,
      suggested: c.suggested_clause,
      risk_level: c.risk_level,
    });
  };

  const applySuggestionFromModal = () => {
    if (!modal) return;
    const { clause_id, suggested } = modal;

    const container = containerRef.current;
    if (container) {
      const span = container.querySelector(`.risky[data-clause-id="${clause_id}"]`);
      if (span) {
        span.innerText = suggested;
        span.classList.remove("risky");
        span.removeAttribute("data-clause-id");
        span.removeAttribute("data-risk");
        span.style.textDecoration = "none";
      }
    }

    onApplySuggestion(clause_id, suggested);
    setModal(null);
  };

  function buildHtmlFromClauses(analysisLocal) {
    if (!analysisLocal?.clauses) return "<div>No preview available</div>";
    const rows = [];
    for (const c of analysisLocal.clauses) {
      const risky = ["high", "medium"].includes(c.risk_level?.toLowerCase());
      const escaped = escapeHtml(c.original_clause);
      rows.push(
        risky
          ? `<p class="pblock"><span class="risky" data-clause-id="${c.clause_id}" data-risk="${c.risk_level}">${escaped}</span></p>`
          : `<p class="pblock">${escaped}</p>`
      );
    }
    return `<div class="document">${rows.join("\n")}</div>`;
  }

  function escapeHtml(str) {
    if (!str) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  return (
    <>
      <div className="p-4 bg-gray-900 text-gray-100 rounded h-[80vh] overflow-auto">
        <div ref={containerRef} dangerouslySetInnerHTML={{ __html: localHtml }} />
      </div>

      {modal && (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-end z-50 p-0">
          <div className="bg-gray-800 shadow-2xl h-full w-[380px] p-6 border-l border-gray-700 overflow-auto">

            {/* Header */}
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-bold">AI Suggestion</h3>
              <button className="text-sm text-gray-300" onClick={() => setModal(null)}>Close</button>
            </div>

            {/* Risk + ID */}
            <div className="flex items-center gap-2 mb-3">
              <span className={`px-2 py-1 rounded text-xs text-black font-semibold ${
                modal.risk_level === "High" ? "bg-red-400" :
                modal.risk_level === "Medium" ? "bg-yellow-400" :
                "bg-green-400"
              }`}>
                {modal.risk_level}
              </span>
              <span className="text-xs text-gray-400">Clause: {modal.clause_id}</span>
            </div>

            {/* ORIGINAL */}
            <div className="mb-4">
              <h4 className="text-sm text-gray-400 mb-1">Original</h4>
              <div className="p-3 bg-gray-700 rounded text-gray-200 whitespace-pre-wrap">
                {modal.original}
              </div>
            </div>

            {/* SUGGESTED */}
            <div className="mb-6">
              <h4 className="text-sm text-green-400 mb-1">Suggested</h4>
              <div className="p-3 bg-green-500/10 rounded text-green-300 whitespace-pre-wrap">
                {modal.suggested}
              </div>
            </div>

            {/* BUTTONS — Copy Removed */}
            <div className="flex flex-col gap-3">
              <button
                onClick={applySuggestionFromModal}
                className="w-full py-2 bg-purple-600 hover:bg-purple-700 rounded text-white font-semibold"
              >
                Apply Suggestion
              </button>

              <button
                onClick={() => setModal(null)}
                className="w-full py-2 bg-gray-600 hover:bg-gray-700 rounded text-white"
              >
                Close
              </button>
            </div>

            <p className="text-xs text-gray-500 mt-6 text-center">
              Powered by your playbook • Click outside to close
            </p>
          </div>
        </div>
      )}
    </>
  );
}
