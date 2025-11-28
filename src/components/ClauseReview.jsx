// src/components/ClauseReview.jsx
import React from "react";

function ClauseReview({ analysis, onApplySuggestion }) {
  if (!analysis?.clauses) return null;

  return (
    <div className="clause-review">
      <h2 className="text-lg font-bold mb-3">Clause Review Panel</h2>

      <div className="space-y-3">
        {analysis.clauses.map((clause) => (
          <div
            key={clause.clause_id}
            className="p-3 rounded bg-gray-900/30 border border-gray-700"
          >
            <div className="mb-2">
              <strong className="text-purple-400">{clause.clause_id}</strong>
              <span className="ml-2 text-sm text-gray-400">
                ({clause.risk_level})
              </span>
            </div>

            <div className="text-sm text-gray-300 mb-2">
              {clause.original_clause}
            </div>

            {(clause.risk_level === "High" ||
              clause.risk_level === "Medium") && (
              <div className="mt-2">
                <div className="text-xs text-gray-400 mb-1">
                  Suggested Replacement:
                </div>
                <div className="text-sm text-green-300 mb-2">
                  {clause.suggested_clause}
                </div>

                <button
                  className="btn-ghost text-xs"
                  onClick={() =>
                    onApplySuggestion(
                      clause.clause_id,
                      clause.suggested_clause
                    )
                  }
                >
                  Apply Suggestion
                </button>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

export default ClauseReview;
