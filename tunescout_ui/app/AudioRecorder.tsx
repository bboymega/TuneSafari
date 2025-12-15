import { useState, useRef, useEffect } from "react";
import { useRouter } from 'next/navigation';
import config from './config.json'

export default function AudioRecorder({ disabled, uploadtoAPI, setDisabled, setErrorMsg, setIsError, setWarnMsg, setIsWarning, setTitle }) {
  const [isRecording, setIsRecording] = useState(false);
  const [seconds, setSeconds] = useState(0);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<BlobPart[]>([]);
  const activeStreamRef = useRef<MediaStream | null>(null);
  const router = useRouter();

  useEffect(() => { // Count recording seconds, max 10 seconds.
    let interval;
    if (isRecording) {
      interval = setInterval(() => {
        setSeconds(prev => {
          if (prev + 1 >= config.maxDuration) {
            handleSubmitRecord();
            return config.maxDuration;
          }
          return prev + 1;
        });
      }, 1000);
    } else {
      // Stop and reset counting if recording stops
      clearInterval(interval);
      setSeconds(0);
    }
    return () => clearInterval(interval);
  }, [isRecording]);

  function formatTime(seconds) { // Format seconds to mm:ss
    const minutes = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  }

  const startRecording = async () => {
    // Request microphone permission first
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          channelCount: 1, 
          sampleRate: 44100,
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false
        } 
      });
      activeStreamRef.current = stream;
      const recorder = new MediaRecorder(stream);
      mediaRecorderRef.current = recorder;
      audioChunksRef.current = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunksRef.current.push(e.data);
      };

      recorder.onstop = () => {
        if (activeStreamRef.current) {
          activeStreamRef.current.getTracks().forEach(track => track.stop());
          activeStreamRef.current = null; 
        }
        const blob = new Blob(audioChunksRef.current, { type: "audio/webm" });
        audioChunksRef.current = [];
        if (stopResolveRef.current) {
          stopResolveRef.current(blob);
        }
      };
    // Start recording after permission granted
    recorder.start();
    setIsRecording(true);
    setDisabled(true);
    setTitle(`${config.appName} - Recording...`);
    // Permission denied
    } catch (err) {
      setErrorMsg(`Error accessing microphone: ${err}`);
      setTitle(`${config.appName} - Find the tracks that sticks`);
      setIsWarning(false);
      setIsError(true);
      setIsRecording(false);
      setDisabled(false);
    }
  };

  const stopResolveRef = useRef<(blob: Blob) => void>();

  const stopRecording = (): Promise<Blob | null> => {
    return new Promise((resolve) => {
      stopResolveRef.current = resolve;
      mediaRecorderRef.current.stop();
      setTimeout(() => {
        setIsRecording(false);
      }, 0);
    });
  };
  
  const handleSubmitRecord = async () => {
    const blob = await stopRecording();
    try {
      setDisabled(true);
      console.log(blob);
      const formData = new FormData();
      formData.append('file', blob);
      const url = `${config.apiBaseUrl.replace(/\/$/, '')}/api/recognize`;
      const response = await uploadtoAPI(url, formData);
      if (response) {
        const resultToken = JSON.parse(response).token;
        if (resultToken !== undefined && resultToken !== null) {
          router.push(`/results/${resultToken}`);
        }
        else {
          setTitle(`${config.appName} - ${config.title}`);
          setIsError(false);
          setIsWarning(true);
          setWarnMsg('Warning: No results were found');
        }
      }
    } catch (error) {
      setTitle(`${config.appName} - ${config.title}`);
      setIsError(true);
      setIsWarning(false);
      setErrorMsg(error.toString());
    }
    finally {
      setDisabled(false);
    }
  };

  return (
    <div>
      {isRecording && (
        <div
          id="recordProgress"
          onClick={handleSubmitRecord}
          style={{
            fontFamily: '"OPTICopperplate-Light", sans-serif',
            position: "fixed",
            top: "50%",
            left: "50%",
            transform: "translate(-50%, -50%)",
            backgroundColor: "rgba(0,0,0,0.9)",
            color: "#fff",
            padding: "10px 20px",
            borderRadius: "5px",
            fontSize: "2rem",
            textAlign: "center",
            zIndex: 9999,
            cursor: "pointer"
          }}
        >
          <div
            style={{
              width: "300px",
              height: "180px",
              backgroundImage: "url('assets/img/recording.gif')",
              backgroundSize: "cover",
              margin: "0 auto"
            }}
          />

          <div><i className="fas fa-circle recording-dot"></i> Recording...<br/>{formatTime(seconds)}</div>

          <div style={{ fontSize: "1.2rem" }} className="mt-5">
            <i className="fa-solid fa-hand-pointer"></i> Tap this box<br />to stop<br />and recognize
          </div>
        </div>
      )}

      <button
        id="recordBtn"
        className="mx-auto btn btn-primary mt-2 mb-5"
        style={{ backgroundColor: "red" }}
        disabled={disabled}
        onClick={startRecording}
      >
        Record <i className="fas fa-microphone" />
      </button>
    </div>
  );
}
