import { useRef, useState } from "react";
import { useRouter } from 'next/navigation';
import config from './config.json';

export default function FileSelector ({ disabled, uploadtoAPI, setDisabled, setErrorMsg, setIsError, setWarnMsg, setIsWarning, setTitle }) {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [fileName, setFileName] = useState<string | null>(null); // State to store filename
  const [selectedFile, setSelectedFile] = useState(null);
  const router = useRouter();

  const handleButtonClick = () => {
    fileInputRef.current.click();
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];

    if (file) {
      setSelectedFile(file);
      setFileName(file.name);
    } else {
      setSelectedFile(null);
      setFileName(null);
    }
  };

  const handleRecognizeFile = async () => {
    if (!selectedFile)
    {
      setIsError(true);
      setIsWarning(false);
      setErrorMsg('Error: No file selected');
    }
    else {
      try {
        setDisabled(true);
        const formData = new FormData();
        formData.append('file', selectedFile);
        const url = `${config.apiBaseUrl.replace(/\/$/, '')}/api/recognize`;
        const response = await uploadtoAPI(url, formData);
        if (response) {
          const resultToken = JSON.parse(response).token;
          if (resultToken !== undefined && resultToken !== null) {
            router.push(`/results/${resultToken}`);
          }
          else {
            setIsError(false);
            setIsWarning(true);
            setTitle(`${config.appName} - ${config.title}`);
            setWarnMsg('Warning: No results were found');
          }
        }
      }
      catch (error) {
        setTitle(`${config.appName} - ${config.title}`);
        setIsError(true);
        setIsWarning(false);
        setErrorMsg(error.toString());
      }
      finally {
        setDisabled(false);
      }
    };
  };

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
        style={{"backgroundColor": "#212529"}}
        disabled={disabled}
        onClick={handleButtonClick}
        >
          Select <i className="fas fa-file-audio"></i> <i className="fas fa-file-video"></i>
      </button>

      <button
        id="uploadBtn"
        className="mx-auto btn btn-primary mt-2 mb-2"
        style={{"backgroundColor": "#f8f9fa", "color":"#000000ff"}}
        disabled={disabled}
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