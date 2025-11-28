// src/components/PdfViewer.jsx
import React from "react";

function PdfViewer({ pdfUrl }) {
  return (
    <iframe
      src={pdfUrl}
      title="PDF Viewer"
      className="w-full h-[80vh] border border-gray-700 rounded"
    />
  );
}

export default PdfViewer;
