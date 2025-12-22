import { useState, useRef, useEffect } from "react";
import { useRouter } from 'next/navigation';
import config from './config.json'
import CloseButton from 'react-bootstrap/CloseButton';
import { RecognitionResponse } from './page';

interface AudioRecorderProps {
  disabled: boolean;
  uploadtoAPI: (url: string, formData: FormData) => Promise<RecognitionResponse>;
  setDisabled: (disabled: boolean) => void;
  setErrorMsg: (msg: string) => void;
  setIsError: (isError: boolean) => void;
  setWarnMsg: (msg: string) => void;
  setIsWarning: (isWarning: boolean) => void;
  setTitle: (title: string) => void;
  mainDivRef: React.RefObject<HTMLDivElement | null>;
}

export default function AudioRecorder({ disabled, uploadtoAPI, setDisabled, setErrorMsg, setIsError, setWarnMsg, setIsWarning, setTitle, mainDivRef }: AudioRecorderProps) {
  const [isRecording, setIsRecording] = useState(false);
  const [seconds, setSeconds] = useState(0);
  const [cancelled, setCancelled] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<BlobPart[]>([]);
  const activeStreamRef = useRef<MediaStream | null>(null);
  const router = useRouter();

  const handleCloseButtonClick = () => {
    setSeconds(0);
    setCancelled(true);
    setTitle(`${config.appName} - ${config.title}`);
    mainDivRef.current!.style.userSelect = "auto";
    if (activeStreamRef.current) {
        // Stop every track (audio and video, if present)
        activeStreamRef.current.getTracks().forEach(track => {
            track.stop();
        });
    }
    activeStreamRef.current = null;
    audioChunksRef.current = [];
    mediaRecorderRef.current!.stop();
    setDisabled(false);
    setIsRecording(false);
  }

  useEffect(() => { // Count recording seconds, max 10 seconds.
    let interval: ReturnType<typeof setInterval> | undefined = undefined;
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
      mainDivRef.current!.style.userSelect = "auto";
    }
    return () => {if (interval) clearInterval(interval)};
  }, [isRecording]);

  function formatTime(seconds: number) { // Format seconds to mm:ss
    const minutes = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  }

  const startRecording = async () => {
    // Request microphone permission first
    try {
      mainDivRef.current!.style.userSelect = "none";
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
      mainDivRef.current!.style.userSelect = "auto";
    }
  };

  const stopResolveRef = useRef<((blob: Blob) => void) | null>(null);

  const stopRecording = (): Promise<Blob | null> => {
    return new Promise((resolve) => {
      stopResolveRef.current = resolve;
      mediaRecorderRef.current!.stop();
      setTimeout(() => {
        setIsRecording(false);
      }, 0);
    });
  };
  
  const handleSubmitRecord = async () => {
    if(cancelled) {
      setCancelled(false);
      return;
    }
    if (!activeStreamRef.current && audioChunksRef.current.length === 0) {
        setCancelled(false);
        return;
    }
    const blob = await stopRecording();
    try {
      setDisabled(true);
      const formData = new FormData();

      if (!blob) {
        setDisabled(false);
        return;
      }

      formData.append('file', blob);
      const url = `${config.apiBaseUrl.replace(/\/$/, '')}/api/recognize`;
      const response = await uploadtoAPI(url, formData);
      if (response) {
        const resultToken = response.token;
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
    } catch (error: unknown) {
      setTitle(`${config.appName} - ${config.title}`);
      setIsError(true);
      setIsWarning(false);
      if (error instanceof Error) {
        setErrorMsg(error.message); 
      } else {
        setErrorMsg(String(error));
      }
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
          <CloseButton 
            aria-label='Hide'
            variant= 'white'
            style = {{
              zIndex: 99999,
              position: 'absolute',
              top: '8px',
              right: '8px',
              fontSize: '1.5rem'
            }}
            onClick = {handleCloseButtonClick}
          />

          <div
            style={{
              width: "300px",
              height: "180px",
              backgroundImage: "url('assets/img/recording.gif')",
              backgroundSize: "cover",
              margin: "0 auto",
              marginTop: "20px"
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
        className="mx-auto btn btn-primary mt-2 mb-4"
        style={{ backgroundColor: "red" }}
        disabled={disabled}
        onClick={startRecording}
      >
        Record <i className="fas fa-microphone" />
      </button>
    </div>
  );
}
