'use client';
import { useState, useEffect, useRef } from "react";
import AudioRecorder from "./AudioRecorder";
import FileSelector from './FileSelector';
import ErrorAlert from './ErrorAlert';
import WarningAlert from "./WarningAlert";
import UploadProgress from "./UploadProgress";
import config from "./config.json"

export default function index() {
  const [title, setTitle] = useState(`${config.appName} - ${config.title}`);
  const [disabled, setDisabled] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");
  const [isError, setIsError] = useState(false);
  const [warnMsg, setWarnMsg] = useState("");
  const [isWarning, setIsWarning] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [showProgress, setShowProgress] = useState(false);
  const [uploadState, setUploadState] = useState("Uploading...");
  const currentYear = new Date().getFullYear();
  const mainDivRef = useRef<HTMLDivElement>(null);

  function ProgressBar() {
    return (
      <div
        className="progress"
        style=
          {{ height: '1.5px',
          backgroundColor: 'transparent',
          border: 'none',
          overflow: 'hidden',
          position: 'fixed',
          top: 0,
          left: 0,
          width: '100%',
          zIndex: 99999
        }}
        >
        <div
          className="progress-bar"
          role="progressbar"
          style=
            {{ width: `${progress}%`,
            backgroundColor: '#f8f9fa',
            transition: 'width 0.4s ease'
            }}
          aria-valuenow={progress}
          aria-valuemin="0"
          aria-valuemax="100"
        >
        </div>
      </div>
    )
  }
    
  function uploadtoAPI(url, formData) {
    return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', url, true);
    setIsUploading(true);
    setTitle(`${config.appName} - Uploading...`);

    xhr.onload = function () {
      if (xhr.status === 200) {
        resolve(xhr.responseText); // Resolve with the response text
        setUploadState('Uploading...');
        setShowProgress(true);
        setProgress(20);
        setIsUploading(false);
        setDisabled(false);
        return xhr.responseText;
      } else {
        const jsonData = JSON.parse(xhr.responseText);
        const errorMessage =
          jsonData.error ??   // highest priority
          jsonData.status ??  // fallback
          null;           // nothing found
        reject(new Error(`${xhr.status} ${errorMessage}`));
        setUploadState('Uploading...');
        setIsUploading(false);
        setDisabled(false);
        setShowProgress(false);
        setProgress(0);
        setErrorMsg(`${xhr.status} ${errorMessage}`);
        setIsError(true);
        return null;
      }
    };
    xhr.upload.onload = function () {
      setTitle(`${config.appName} - Recognizing...`);
      setUploadState('Recognizing...')
    };
    xhr.upload.onprogress = function (event) {
      if (event.lengthComputable) {
        const percent = (event.loaded / event.total) * 100;
        setTitle(`${config.appName} - Uploading... ${Math.round(percent)}%`);
        const text = document.createElement('span');
        text.id = 'uploadProgressText';
        setUploadState(`Uploading... ${Math.round(percent)}%`);
      };
    };
    xhr.onerror = function () {
      reject(new Error('Backend not reachable'));
      setIsUploading(false);
      setDisabled(false);
      setShowProgress(false);
      setProgress(0);
      setTitle(`${config.appName} - ${config.title}`);
      setIsError(true)
      setErrorMsg('Error: Backend not reachable');
    };
    xhr.send(formData);
  })
};

useEffect(() => {
  if (title) {
    document.title = title;
  }
}, [title]);

useEffect(() => {
  if(isWarning) {
    setShowProgress(false);
    setProgress(0);
  }
}, [isWarning]);

return (
  <>
    <div id="main" ref={mainDivRef}>
      {showProgress &&
        <ProgressBar/>
      }
      <div className="masthead">
        <div className="container px-4 px-lg-5 d-flex h-100 align-items-center justify-content-center">
          <div className="d-flex justify-content-center" id='mainDiv'>
            {isError && (
              <ErrorAlert
                message={errorMsg}
                onClose={() => setIsError(false)}
              />
            )}
            {isWarning && (
              <WarningAlert
                message={warnMsg}
                onClose={() => setIsWarning(false)}
              />
            )}
            {isUploading && (
              <UploadProgress
                state = {uploadState}
              />
            )}
            <div className="text-center" id="panel">
              <h1 className="mx-auto my-0 mt-2 mb-3 text-uppercase">{config.appName}</h1>
              <div
                id="logo"
                className="mx-auto"
                style={{
                  backgroundImage: "url('/assets/img/logo.png')",
                  backgroundSize: 'cover',
                }}
              />
              <h2 className="mx-auto mt-2 mb-4">Record an audio</h2>
              <AudioRecorder
                disabled={disabled}
                uploadtoAPI={uploadtoAPI}
                setDisabled={setDisabled}
                setErrorMsg={setErrorMsg}
                setIsError={setIsError}
                setWarnMsg={setWarnMsg}
                setIsWarning={setIsWarning}
                setTitle={setTitle}
                mainDivRef={mainDivRef}
              />
              <h2 className="mx-auto mt-2 mb-4">or upload a file</h2>
              <FileSelector
                disabled={disabled}
                uploadtoAPI={uploadtoAPI}
                setDisabled={setDisabled}
                setErrorMsg={setErrorMsg}
                setIsError={setIsError}
                setWarnMsg={setWarnMsg}
                setIsWarning={setIsWarning}
                setTitle={setTitle}
                mainDivRef={mainDivRef}
                />
            </div>
          </div>
        </div>
      </div>
      <footer className="footer bg-black small text-center text-white-50"><div className="container px-4 px-lg-5">Copyright &copy; {config.appName} {currentYear}</div></footer>
    </div>
  </>
  );
}
