'use client';
import { useState, useEffect } from "react";
import AudioRecorder from "./AudioRecorder";
import FileSelector from './FileSelector';
import ErrorAlert from './ErrorAlert';
import WarningAlert from "./WarningAlert";
import UploadProgress from "./UploadProgress";

export default function index() {
  const [title, setTitle] = useState('TuneScout - Find the tracks that sticks');
  const [disabled, setDisabled] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");
  const [isError, setIsError] = useState(false);
  const [warnMsg, setWarnMsg] = useState("");
  const [isWarning, setIsWarning] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadState, setUploadState] = useState("Uploading...");
    
  function uploadtoAPI(url, formData) {
    return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', url, true);
    setIsUploading(true);
    setTitle('TuneScout - Uploading...');

    xhr.onload = function () {
      if (xhr.status === 200) {
        resolve(xhr.responseText); // Resolve with the response text
        setUploadState('Uploading...');
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
        setErrorMsg(`${xhr.status} ${errorMessage}`);
        setIsError(true);
        return null;
      }
    };
    xhr.upload.onload = function () {
      setTitle('TuneScout - Recognizing...');
      setUploadState('Recognizing...')
    };
    xhr.upload.onprogress = function (event) {
      if (event.lengthComputable) {
        const percent = (event.loaded / event.total) * 100;
        setTitle(`TuneScout - Uploading... ${Math.round(percent)}%`);
        const text = document.createElement('span');
        text.id = 'uploadProgressText';
        setUploadState(`Uploading... ${Math.round(percent)}%`);
      };
    };
    xhr.onerror = function () {
      reject(new Error('Backend not reachable'));
      setIsUploading(false);
      setDisabled(false);
      setTitle('TuneScout - Find the tracks that sticks');
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


return (
  <>
    <div id="main">
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
              <h1 className="mx-auto my-0 mt-2 mb-5 text-uppercase">TuneScout</h1>
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
                />
            </div>
          </div>
        </div>
      </div>
      <footer className="footer bg-black small text-center text-white-50"><div className="container px-4 px-lg-5">Copyright &copy; TuneScout 2025</div></footer>
    </div>
  </>
  );
}
