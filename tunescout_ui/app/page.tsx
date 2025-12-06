'use client';
import React, { useRef, useState, ChangeEvent } from "react";
import 'bootstrap/dist/css/bootstrap.min.css';
import 'typeface-roboto';
import '@fortawesome/fontawesome-free/css/all.min.css'; 

function FileSelector () {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [fileName, setFileName] = useState<string | null>(null); // State to store filename

  const handleButtonClick = () => {
    fileInputRef.current.click(); // Trigger the file input click programmatically
  };

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile) {
      setFileName(selectedFile.name)
    }
  };

  const handleRecognizeFile = async () => {
    try {
      const formData = new FormData();
      formData.append('file', document.getElementById('fileInput').files[0]);
      const url = 'http://172.16.241.129:8080/api/recognize';
      const response = await uploadtoAPI(url, formData);
      console.log(response)
    }
    catch (error) {
      console.error('Error during file upload:', error);
    }
  };

  function deleteUploadProgress() {
    const uploadProgress = document.getElementById('uploadProgress');
    if (uploadProgress) {
      uploadProgress.remove();  // This removes the element from the DOM
    }
  }

  function uploadtoAPI(url, formData) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open('POST', url, true);

      const spinner = document.createElement('div');
      spinner.id = 'spinner';
      spinner.className = 'spinner-border';

      const text = document.createElement('div');
      text.id = 'uploadProgressText';
      text.textContent = "Uploading...";

      const uploadProgress = document.createElement('div');
      uploadProgress.id = 'uploadProgress';
      uploadProgress.style.position = 'fixed'; 
      uploadProgress.style.top = '50%';
      uploadProgress.style.fontFamily = '"OPTICopperplate-Light", sans-serif';
      uploadProgress.className = "mx-auto mt-2 mb-8 text-center";
      uploadProgress.style.transform = 'translateY(-50%)'; // Adjust for exact middle positioning
      uploadProgress.style.backgroundColor = 'rgba(0, 0, 0, 0.5)'; // 50% transparency (black background)
      uploadProgress.style.color = '#fff';      // White text
      uploadProgress.style.padding = '10px 20px'; // Some padding
      uploadProgress.style.borderRadius = '5px'; // Rounded corners
      uploadProgress.style.fontSize = '2rem';   // Text size
      uploadProgress.style.zIndex = '1000';     // Ensure it stays above other content
      uploadProgress.appendChild(spinner);
      uploadProgress.appendChild(text);
      document.getElementById("mainDiv").appendChild(uploadProgress);

      xhr.onload = function () {
        if (xhr.status === 200) {
          resolve(xhr.responseText); // Resolve with the response text
          deleteUploadProgress();
        } else {
          reject(new Error(`Failed to upload file: ${xhr.statusText}`));
          deleteUploadProgress();
        }
      };
      xhr.upload.onload = function () {
        document.getElementById('uploadProgressText').textContent = `Recognizing...`;
      };
      xhr.upload.onprogress = function (event) {
        if (event.lengthComputable) {
          const percent = (event.loaded / event.total) * 100;
          console.log(`Upload Progress: ${Math.round(percent)}%`);
          const text = document.createElement('span');
          text.id = 'uploadProgressText';
          document.getElementById('uploadProgressText').textContent = `Uploading... ${Math.round(percent)}%`;
        };
      };
      xhr.onerror = function () {
        reject(new Error('API not reachable'));
        deleteUploadProgress();
      };
      xhr.send(formData);
    })
  }


  return (
    <span>
      <input ref={fileInputRef}
        id="fileInput"
        type="file"
        style={{ display: 'none' }}
        accept="audio/*,video/*"
        onChange={handleFileChange}
      />

      <button
        id="selectFileBtn"
        className="btn btn-primary mt-2 mb-2"
        onClick={handleButtonClick}
        >
          Select <i className="fas fa-file-audio"></i> <i className="fas fa-file-video"></i>
      </button>

      <button
        id="uploadBtn"
        className="mx-auto btn btn-primary mt-2 mb-2"
        style={{"backgroundColor": "#cbcbcbff", "color":"#000000ff"}}
        onClick={handleRecognizeFile}
        >
          Recognize <i className="fas fa-record-vinyl"></i>
      </button>
      
      {fileName ? (
        <div style={{color: '#c8c8c8ff', fontFamily: '"OPTICopperplate-Light", sans-serif'}}>
          <strong>Selected file:</strong> {fileName}
        </div>
      ) : (
        <div style={{color: '#c8c8c8ff', fontFamily: '"OPTICopperplate-Light", sans-serif'}}>
          <strong><br/></strong>
        </div>
      )}
    </span>
  )
}

export default function Index() {
  return (
    <div id="main">
          <meta charSet="utf-8" />
          <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no" />
          <meta name="description" content="" />
          <meta name="author" content="" />
          <link rel="icon" type="image/x-icon" href="assets/favicon.ico" />
          <link href="css/styles.css" rel="stylesheet" />
          <header className="masthead">
              <div className="container px-4 px-lg-5 d-flex h-100 align-items-center justify-content-center">
                  <div className="d-flex justify-content-center" id="mainDiv">
                      <div className="text-center" id="panel">
                          <h1 className="mx-auto my-0 mt-2 mb-5 text-uppercase">Audio<br/>Analyze</h1>
                          <h2 className="mx-auto mt-2 mb-4">Record an audio</h2>
                          <button id="recordBtn" className="mx-auto btn btn-primary mt-2 mb-5" style={{"backgroundColor": "red"}}>Record <i className="fas fa-microphone"></i></button>
                          <h2 className="mx-auto mt-2 mb-4">or upload a file</h2>
                          <FileSelector />
                      </div>
                  </div>
              </div>
          </header>
          <footer className="footer bg-black small text-center text-white-50"><div className="container px-4 px-lg-5">Copyright &copy; Your Website 2023</div></footer>
    </div>
  );
}
