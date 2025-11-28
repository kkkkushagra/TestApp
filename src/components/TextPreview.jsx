// src/components/TextPreview.jsx
import React from "react";

function TextPreview({ html }) {
  return (
    <div
      className="html-preview"
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}

export default TextPreview;
