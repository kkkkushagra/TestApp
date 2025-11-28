// src/components/FileUpload.jsx
import React, { useRef } from "react";

function FileUpload({ onFileChange, onReset, file }) {
  const inputRef = useRef();

  const handleSelect = (e) => {
    const f = e.target.files[0];
    if (f) {
      onFileChange(f);
    }
  };

  return (
    <div className="p-4 rounded bg-gray-900/40 border border-gray-700">
      <div className="mb-3 font-bold">Upload Contract</div>

      <input
        type="file"
        ref={inputRef}
        className="hidden"
        accept=".pdf,.docx"
        onChange={handleSelect}
      />

      <button
        className="btn-ghost mr-3"
        onClick={() => inputRef.current.click()}
      >
        Select File
      </button>

      {file && (
        <>
          <span className="text-sm text-gray-300">{file.name}</span>
          <button className="btn-ghost ml-3 text-xs" onClick={onReset}>
            Reset
          </button>
        </>
      )}
    </div>
  );
}

export default FileUpload;
