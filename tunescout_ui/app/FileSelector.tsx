import { useRef, useState, useCallback, useEffect, useLayoutEffect } from "react";
import { useRouter } from 'next/navigation';
import config from './config.json';
import CloseButton from 'react-bootstrap/CloseButton';
import { RecognitionResponse } from './page';

export interface FileSelectorProps {
  disabled: boolean;
  uploadtoAPI: (url: string, data: FormData) => Promise<RecognitionResponse>;
  setDisabled: (disabled: boolean) => void;
  setErrorMsg: (msg: string) => void;
  setIsError: (isError: boolean) => void;
  setWarnMsg: (msg: string) => void;
  setIsWarning: (isWarning: boolean) => void;
  setTitle: (title: string) => void;
}

export default function FileSelector ({ disabled, uploadtoAPI, setDisabled, setErrorMsg, setIsError, setWarnMsg, setIsWarning, setTitle }: FileSelectorProps) {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [fileName, setFileName] = useState<string | null>(null); // State to store filename
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [showMediaTrimmer, setShowMediaTrimmer ] = useState(false);
  const [isAudioOnly, setIsAudioOnly] = useState(true);
  const videoRef = useRef<HTMLVideoElement>(null);
  const router = useRouter();

  function MediaTrimmer() {
    const [duration, setDuration] = useState(0);

    const handleLoadedMetadata = (e: React.SyntheticEvent<HTMLVideoElement>) => {
      const duration = (e.target as HTMLVideoElement).duration; // Duration in seconds
      setDuration(duration);
      if ((e.target as HTMLVideoElement).videoHeight === 0) {
        setIsAudioOnly(true);
      } else {
        setIsAudioOnly(false);
      }
    };

    const handleCloseButtonClick = () => {
      setShowMediaTrimmer(false);
      setSelectedFile(null);
      setFileName(null);
      setDisabled(false);
    }

    interface TimelineZoomProps {
      setShowTimelineZoom: (show: boolean) => void;
      clipStart: number;
      setClipStart: React.Dispatch<React.SetStateAction<number>>;
      clipEnd: number;
      setClipEnd: React.Dispatch<React.SetStateAction<number>>;
    }

    const TimelineZoom = ({ setShowTimelineZoom, clipStart, setClipStart, clipEnd, setClipEnd }: TimelineZoomProps) => {
      const [zoomLevel, setZoomLevel] = useState(1);
      const MIN_CLIP_DURATION = 1;
      const MAX_CLIP_DURATION = config.maxDuration;
      const timelineRef = useRef<HTMLDivElement>(null);
      const [activeHandle, setActiveHandle] = useState(1);
      const [position, setPosition] = useState({ x: 0, y: 0 });
      const [isDragging, setIsDragging] = useState(false);
      const viewportRef = useRef<HTMLDivElement>(null);
      const zoomSliderRef = useRef<HTMLInputElement>(null);
      const offset = useRef({ x: 0, y: 0 });

      const draggingHandleRef = useRef<string>(null);
      const startDragTimeRef = useRef({ clipStart: 0, clipEnd: 0 });
      const startMouseXRef = useRef(0);

      const getHandleZIndex = (handleType: string) => {
        const startPercent = (clipStart / duration) * 100;
        const ACTIVE_Z = 100;
        const BACK_Z = 20;
        if (startPercent > 95.0 && handleType === 'start') return ACTIVE_Z;
        if (startPercent > 95.0 && handleType === 'end') return BACK_Z;
        if (activeHandle === 0 && handleType === 'start') return ACTIVE_Z;
        if (activeHandle === 1 && handleType === 'end') return ACTIVE_Z;
        return BACK_Z;
      };

      const formatTime = (seconds: number) => {
        const min = Math.floor(seconds / 60);
        const sec = Math.floor(seconds % 60);
        return `${min}:${sec < 10 ? '0' : ''}${sec}`;
      };

      const getSelectedRangeStyle = () => {
        const startPercent = (clipStart / duration) * 100;
        const endPercent = (clipEnd / duration) * 100;
        return { left: `${startPercent}%`, width: `${endPercent - startPercent}%` };
      };

      const handlePointerMove = useCallback((e: PointerEvent) => {
        if (!draggingHandleRef.current || !timelineRef.current) return;
        const trackBounds = timelineRef.current.getBoundingClientRect();
        const trackWidth = trackBounds.width;
        const handleType = draggingHandleRef.current;
        const currentX = e.clientX;

        if (handleType === 'middle') {
          const mouseDeltaX = currentX - startMouseXRef.current;
          const timeDelta = (mouseDeltaX / trackWidth) * duration;
          let newStart = startDragTimeRef.current.clipStart + timeDelta;
          let newEnd = startDragTimeRef.current.clipEnd + timeDelta;

          if (newStart < 0) { const off = 0 - newStart; newStart += off; newEnd += off; }
          if (newEnd > duration) { const off = newEnd - duration; newStart -= off; newEnd -= off; }
          setClipStart(newStart);
          setClipEnd(newEnd);
          return;
        }

        const newPixelPosition = currentX - trackBounds.left;
        let newTime = (newPixelPosition / trackWidth) * duration;
        newTime = Math.max(0, Math.min(duration, newTime));

        if (handleType === 'start') {
          setActiveHandle(0);
          const maxStart = clipEnd - MIN_CLIP_DURATION;
          newTime = Math.min(newTime, maxStart);
          if (clipEnd - newTime > MAX_CLIP_DURATION) {
            setClipStart(newTime);
            setClipEnd(Math.min(duration, newTime + MAX_CLIP_DURATION));
          } else {
            setClipStart(newTime);
            if (clipEnd - newTime <= MIN_CLIP_DURATION) setClipEnd(newTime + MIN_CLIP_DURATION);
          }
        } else if (handleType === 'end') {
          setActiveHandle(1);
          const minEnd = clipStart + MIN_CLIP_DURATION;
          newTime = Math.max(newTime, minEnd);
          if (newTime - clipStart > MAX_CLIP_DURATION) {
            setClipEnd(newTime);
            setClipStart(Math.max(0, newTime - MAX_CLIP_DURATION));
          } else {
            setClipEnd(newTime);
            if (newTime - clipStart <= MIN_CLIP_DURATION) setClipStart(newTime - MIN_CLIP_DURATION);
          }
        }
      }, [clipStart, clipEnd, duration, MAX_CLIP_DURATION]);

      const handlePointerUp = useCallback(() => {
        draggingHandleRef.current = null;
        window.removeEventListener('pointermove', handlePointerMove);
        window.removeEventListener('pointerup', handlePointerUp);
      }, [handlePointerMove]);

      const handlePointerDown = (handleType: string, e: React.PointerEvent<HTMLDivElement>) => {
        e.stopPropagation();
        (e.target as Element).setPointerCapture?.(e.pointerId);
        draggingHandleRef.current = handleType;
        startMouseXRef.current = e.clientX;
        startDragTimeRef.current = { clipStart, clipEnd };
        window.addEventListener('pointermove', handlePointerMove);
        window.addEventListener('pointerup', handlePointerUp);
      };

      const handleWheel = useCallback((e: WheelEvent) => {
        e.preventDefault();
        if (!timelineRef.current) return;
        const trackWidth = timelineRef.current.getBoundingClientRect().width;
        const pixelDelta = Math.abs(e.deltaX) > Math.abs(e.deltaY) ? e.deltaX : e.deltaY;
        const timeDelta = (pixelDelta / trackWidth) * duration * 0.3;

        setClipStart(() => {
          let currentDuration = clipEnd - clipStart;
          currentDuration = Math.max(MIN_CLIP_DURATION, Math.min(MAX_CLIP_DURATION, currentDuration));
            
          let newStart = clipStart + timeDelta;
          let newEnd = newStart + currentDuration;

          if (newStart < 0) {
            newStart = 0;
            newEnd = currentDuration;
          } else if (newEnd > duration) {
            newEnd = duration;
            newStart = duration - currentDuration;
          }

          setClipEnd(newEnd);
          return newStart;
        });
      }, [clipEnd, duration, MIN_CLIP_DURATION, MAX_CLIP_DURATION, setClipEnd, setClipStart]);


      useEffect(() => {
        const viewport = viewportRef.current;
        if (!viewport) return;
        viewport.addEventListener('wheel', handleWheel, { passive: false });
        return () => viewport.removeEventListener('wheel', handleWheel);
      }, [handleWheel]);

      useEffect(() => {
        const slider = zoomSliderRef.current;
        if (!slider) return;
        const handleZoomWheel = (e: WheelEvent) => {
          e.preventDefault(); e.stopPropagation();
          const direction = e.deltaY < 0 ? 1 : -1;
          setZoomLevel((prev) => Math.max(1, Math.min(Math.max(5, Math.ceil((duration / 30) / 5) * 5), prev + direction * 0.5)));
        };
        slider.addEventListener('wheel', handleZoomWheel, { passive: false });
        return () => slider.removeEventListener('wheel', handleZoomWheel);
      }, []);

      const handleDragPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
        setIsDragging(true);
        offset.current = { x: e.clientX - position.x, y: e.clientY - position.y };
        (e.target as HTMLElement).setPointerCapture(e.pointerId);
      };

      useEffect(() => {
        const handleMove = (e: PointerEvent) => {
          if (!isDragging) return;
          setPosition({ x: e.clientX - offset.current.x, y: e.clientY - offset.current.y });
        };
        const handleUp = () => setIsDragging(false);
        if (isDragging) {
          window.addEventListener('pointermove', handleMove);
          window.addEventListener('pointerup', handleUp);
        }
        return () => {
          window.removeEventListener('pointermove', handleMove);
          window.removeEventListener('pointerup', handleUp);
        };
      }, [isDragging]);

      const adjustFocus = useCallback(() => {
        if (timelineRef.current && viewportRef.current) {
          const timeline = timelineRef.current;
          const viewport = viewportRef.current;
          const scrollPosition = (clipStart / duration) * (timeline.scrollWidth - viewport.clientWidth);
          viewport.scrollTo({ left: scrollPosition, behavior: 'smooth' });
        }
      }, [clipStart, duration]);

      useLayoutEffect(() => { adjustFocus(); }, [zoomLevel, clipStart, adjustFocus]);

      return (
        <div
          id="zoomControl"
          style={{
            position: "fixed", top: `calc(50% + ${position.y}px)`, left: `calc(50% + ${position.x}px)`,
            transform: "translate(-50%, -50%)", backgroundColor: "rgba(0,0,0,0.95)", color: "#fff",
            padding: "10px 20px", borderRadius: "5px", zIndex: 9999, touchAction: "none", userSelect: "none"
          }}
        >
          <div
            onPointerDown={handleDragPointerDown}
            style={{
              position: 'absolute', top: 0, left: 0, right: 0, height: '40px',
              backgroundColor: 'rgba(255, 255, 255, 0.1)', cursor: 'grab',
              display: 'flex', alignItems: 'center', justifyContent: 'center', borderRadius: '5px 5px 0 0'
            }}
          >
            <div style={{ width: '30px', height: '2px', backgroundColor: '#6d6d6dff' }}></div>
          </div>

          <CloseButton 
            variant="white" 
            style={{ position: 'absolute', top: '8px', right: '8px', zIndex: 200 }} 
            onClick={() => setShowTimelineZoom(false)} 
          />

          <div className="video-trimmer-container mt-5" style={{ width: '350px' }}>
            <div className="d-flex flex-column align-items-center mb-4">
              <div className="d-flex align-items-center gap-3" style={{ width: '300px' }}>
                <i className="fa-solid fa-magnifying-glass-minus text-white"></i>
                <input
                  type="range"
                  ref={zoomSliderRef}
                  className="form-range custom-zoom-slider"
                  min="1"
                  max= {Math.max(5, Math.ceil((duration / 30) / 5) * 5)}
                  step="0.1"
                  value={zoomLevel}
                  onChange={(e) => setZoomLevel(parseFloat(e.target.value))}
                  onPointerDown={(e) =>  e.stopPropagation()}
                  style={{ cursor: 'pointer', touchAction: 'none' }}
                />
                <i className="fa-solid fa-magnifying-glass-plus text-white"></i>
              </div>
              <small className="text-white-50 mt-1">Zoom: {zoomLevel.toFixed(1)}x</small>
            </div>

            <div className="mx-auto mb-3">
              <span className="time-label-zoom-start">{formatTime(clipStart)}</span>
              <span className="time-label-zoom-end">{formatTime(clipEnd)}</span>
            </div>

            <div className="timeline-viewport" ref={viewportRef} style={{ overflowX: 'auto', width: '100%', padding: '20px 0' }}>
              <div
                className="timeline-track"
                ref={timelineRef}
                style={{
                  width: `${100 * zoomLevel}%`, position: 'relative',
                  height: '40px', backgroundColor: '#333', touchAction: 'none'
                }}
              >
                <div
                  className="trimmer-handle left-handle"
                  style={{ left: `${(clipStart / duration) * 100}%`, zIndex: getHandleZIndex('start') }}
                  onPointerDown={(e) => handlePointerDown('start', e)}
                >
                  <div className="handle-grip"></div>
                </div>
                <div
                  className="selected-range"
                  style={getSelectedRangeStyle()}
                  onPointerDown={(e) => handlePointerDown('middle', e)}
                ></div>

                <div
                  className="trimmer-handle right-handle"
                  style={{ left: `${(clipEnd / duration) * 100}%`, zIndex: getHandleZIndex('end') }}
                  onPointerDown={(e) => handlePointerDown('end', e)}
                >
                  <div className="handle-grip"></div>
                </div>
              </div>
            </div>
          </div>
        </div>
      );
    };

    interface TrimmerHandlerProps {
      handleRecognizeFile: (startTimeStamp: number, trimmedDuration: number) => Promise<void>;
    }

    function TrimmerHandler({ handleRecognizeFile }: TrimmerHandlerProps) {
      const MIN_CLIP_DURATION = 1;
      const MAX_CLIP_DURATION = config.maxDuration;
      const initialClipEnd = Math.min(MAX_CLIP_DURATION, duration);

      const [clipStart, setClipStart] = useState(0);
      const [clipEnd, setClipEnd] = useState(initialClipEnd);
      const timelineRef = useRef<HTMLDivElement>(null);
      const [activeHandle, setActiveHandle] = useState(1);
      const [isEditingStart, setIsEditingStart] = useState(false);
      const [isEditingEnd, setIsEditingEnd] = useState(false);
      const [rawClipStartInput, setRawClipStartInput] = useState("");
      const [rawClipEndInput, setRawClipEndInput] = useState("");
      const [passStartTimeRegexCheck, setPassStartTimeRegexCheck] = useState(true);
      const [passEndTimeRegexCheck, setPassEndTimeRegexCheck] = useState(true);
      const [showTimelineZoom, setShowTimelineZoom] = useState(false);

      const draggingHandleRef = useRef<string>(null);
      const startDragTimeRef = useRef({ clipStart: 0, clipEnd: 0 });
      const startMouseXRef = useRef<number>(0);

      const timeCheckRegex = /^([0-9]{0,5}|[0-9]{0,4}:[0-9]{0,2})$/;

      useEffect(() => {
        if (!showMediaTrimmer) setShowTimelineZoom(false);
      }, [showMediaTrimmer]);

      const confirmMediaTrim = () => {
        const clipDuration = clipEnd - clipStart;
        setShowMediaTrimmer(false);
        handleRecognizeFile(clipStart, clipDuration);
      };

      const getHandleZIndex = (handleType: string) => {
        const startPercent = (clipStart / duration) * 100;
        const ACTIVE_Z = 100;
        const BACK_Z = 20;
        if (startPercent > 95.0 && handleType === 'start') return ACTIVE_Z;
        if (startPercent > 95.0 && handleType === 'end') return BACK_Z;
        if (activeHandle === 0 && handleType === 'start') return ACTIVE_Z;
        if (activeHandle === 1 && handleType === 'end') return ACTIVE_Z;
        return BACK_Z;
      };

      const formatTime = (seconds: number) => {
        const min = Math.floor(seconds / 60);
        const sec = Math.floor(seconds % 60);
        return `${min}:${sec < 10 ? '0' : ''}${sec}`;
      };

      const parseTime = (timeString: string) => {
        const cleanedString = timeString.trim();
        if (cleanedString.includes(':')) {
          const parts = cleanedString.split(':');
          if (parts.length !== 2) return null;
          let [minStr, secStr] = parts;
          let minutes = minStr === "" ? 0 : parseInt(minStr, 10);
          let seconds = secStr === "" ? 0 : parseInt(secStr, 10);
          if (isNaN(minutes) || isNaN(seconds)) return null;
          if (seconds > 59) {
            minutes += Math.floor(seconds / 60);
            seconds = seconds % 60;
          }
          return (minutes * 60) + seconds;
        } else {
          if (!/^\d+$/.test(cleanedString)) return null;
          const totalSeconds = parseInt(cleanedString, 10);
          return (isNaN(totalSeconds) || totalSeconds < 0) ? null : totalSeconds;
        }
      };

      const handleTimeInputChange = (e: React.ChangeEvent<HTMLInputElement>, method: number) => {
        if (method === 0) {
          if (!timeCheckRegex.test(e.target.value)) {
            setPassStartTimeRegexCheck(false);
            return;
          }
          setPassStartTimeRegexCheck(true);
          setRawClipStartInput(e.target.value);
        } else if (method === 1) {
          if (!timeCheckRegex.test(e.target.value)) {
            setPassEndTimeRegexCheck(false);
            return;
          }
          setPassEndTimeRegexCheck(true);
          setRawClipEndInput(e.target.value);
        }
      };

      const handleTimeInputConfirm = (e:React.ChangeEvent<HTMLInputElement>, method:number) => {
        setPassEndTimeRegexCheck(true);
        let time = parseTime(e.target.value);
        if (time != null) {
          if (method === 0) {
            setActiveHandle(0);
            setIsEditingStart(false);
            if (time <= duration - MIN_CLIP_DURATION) {
              if (time <= clipEnd - MIN_CLIP_DURATION) {
                if (clipEnd - time <= MAX_CLIP_DURATION) {
                  setClipStart(time);
                } else {
                  setClipEnd(time + MAX_CLIP_DURATION);
                  setClipStart(time);
                }
              } else {
                setClipEnd(time + MIN_CLIP_DURATION);
                setClipStart(time);
              }
            } else {
              setClipEnd(duration);
              setClipStart(duration - MIN_CLIP_DURATION);
            }
          } else if (method === 1) {
            setActiveHandle(1);
            setIsEditingEnd(false);
            if (time > duration) {
              setClipEnd(duration);
              setClipStart(duration - MAX_CLIP_DURATION);
            } else if (time >= MIN_CLIP_DURATION) {
              if (time >= clipStart + MIN_CLIP_DURATION) {
                if (time - clipStart > MAX_CLIP_DURATION) {
                  setClipStart(time - MAX_CLIP_DURATION);
                  setClipEnd(time);
                } else {
                  setClipEnd(time);
                }
              } else {
                setClipStart(time - MIN_CLIP_DURATION);
                setClipEnd(time);
              }
            } else {
              setClipStart(0);
              setClipEnd(MIN_CLIP_DURATION);
            }
          }
        } else {
          setRawClipStartInput(formatTime(clipStart));
        }
        setRawClipStartInput(formatTime(clipStart));
        setRawClipEndInput(formatTime(clipEnd));
      };

      const handlePointerMove = useCallback((e:PointerEvent) => {
        if (!draggingHandleRef.current || !timelineRef.current) return;
        const trackBounds = timelineRef.current.getBoundingClientRect();
        const trackWidth = trackBounds.width;
        const handleType = draggingHandleRef.current;
      
        const currentX = e.clientX;

        if (handleType === 'middle') {
          const mouseDeltaX = currentX - startMouseXRef.current;
          const timeDelta = (mouseDeltaX / trackWidth) * duration;
          let newStart = startDragTimeRef.current.clipStart + timeDelta;
          let newEnd = startDragTimeRef.current.clipEnd + timeDelta;

          if (newStart < 0) {
            const offset = 0 - newStart; 
            newStart += offset; newEnd += offset;
          }
          if (newEnd > duration) {
            const offset = newEnd - duration;
            newStart -= offset; newEnd -= offset;
          }
          setClipStart(newStart);
          setClipEnd(newEnd);
          return;
        }

        const newPixelPosition = currentX - trackBounds.left;
        let newTime = (newPixelPosition / trackWidth) * duration;
        newTime = Math.max(0, Math.min(duration, newTime));

        if (handleType === 'start') {
          setActiveHandle(0);
          const maxStart = clipEnd - MIN_CLIP_DURATION;
          newTime = Math.min(newTime, maxStart);
          const currentDuration = clipEnd - newTime;
          if (currentDuration > MAX_CLIP_DURATION) {
            const exceeded = currentDuration - MAX_CLIP_DURATION;
            setClipStart(newTime);
            setClipEnd(Math.min(duration, clipEnd - exceeded));
          } else {
            setClipStart(newTime);
            if (clipEnd - newTime <= MIN_CLIP_DURATION) setClipEnd(newTime + MIN_CLIP_DURATION);
          }
        } else if (handleType === 'end') {
          setActiveHandle(1);
          const minEnd = clipStart + MIN_CLIP_DURATION;
          newTime = Math.max(newTime, minEnd);
          const currentDuration = newTime - clipStart;
          if (currentDuration > MAX_CLIP_DURATION) {
            const exceeded = currentDuration - MAX_CLIP_DURATION;
            setClipStart(Math.max(0, clipStart + exceeded));
            setClipEnd(newTime);
          } else {
            setClipEnd(newTime);
            if (newTime - clipStart <= MIN_CLIP_DURATION) setClipStart(newTime - MIN_CLIP_DURATION);
          }
        }
      }, [clipStart, clipEnd, duration, MAX_CLIP_DURATION]);

      const handlePointerUp = useCallback(() => {
        draggingHandleRef.current = null;
        window.removeEventListener('pointermove', handlePointerMove);
        window.removeEventListener('pointerup', handlePointerUp);
      }, [handlePointerMove]);

      const handlePointerDown = (handleType:string, e:React.PointerEvent<HTMLDivElement>) => {
        (e.target as HTMLElement).setPointerCapture?.(e.pointerId);
        
        draggingHandleRef.current = handleType;
        startMouseXRef.current = e.clientX;
        startDragTimeRef.current = { clipStart, clipEnd };

        window.addEventListener('pointermove', handlePointerMove);
        window.addEventListener('pointerup', handlePointerUp);
      };

      useEffect(() => {
        setPassEndTimeRegexCheck(true);
        setRawClipStartInput(formatTime(clipStart));
      }, [clipStart]);

      useEffect(() => {
        setPassEndTimeRegexCheck(true);
        setRawClipEndInput(formatTime(clipEnd));
      }, [clipEnd]);

      useEffect(() => {
        const clampedClipStart = Math.max(0, clipStart);
        const clampedClipEnd = Math.min(duration, clipEnd);
        const validClipStart = Math.min(clampedClipStart, clampedClipEnd);
        const validClipEnd = Math.max(clampedClipStart, clampedClipEnd);
        if (validClipStart !== clipStart) setClipStart(validClipStart);
        if (validClipEnd !== clipEnd) setClipEnd(validClipEnd);
      }, [clipStart, clipEnd, duration]);

      useEffect(() => {
        const video = videoRef.current;
        if (video && video.readyState >= 1 && activeHandle !== 1) video.currentTime = clipStart;
      }, [clipStart, activeHandle, videoRef]);

      useEffect(() => {
        const video = videoRef.current;
        if (video && video.readyState >= 1 && activeHandle === 1) video.currentTime = clipEnd;
      }, [clipEnd, activeHandle, videoRef]);

      useEffect(() => {
        const video = videoRef.current;
        if (!video) return;
        const handleTimeUpdate = () => {
          if (video.currentTime <= clipStart - 0.001) {
            video.pause(); video.currentTime = clipStart;
          }
          if (video.currentTime >= clipEnd + 0.001) {
            video.currentTime = clipStart;
            if (!video.paused) video.play(); else video.pause();
          }
        };
        video.addEventListener('timeupdate', handleTimeUpdate);
        return () => video.removeEventListener('timeupdate', handleTimeUpdate);
      }, [clipStart, clipEnd, videoRef]);

      const getSelectedRangeStyle = () => {
        const startP = (clipStart / duration) * 100;
        const endP = (clipEnd / duration) * 100;
        return { left: `${startP}%`, width: `${endP - startP}%` };
      };

      const handleTimelineZoom = () => {
        setShowTimelineZoom(true);
      };

    return (
      <>
      {showTimelineZoom && (
        <TimelineZoom
          setShowTimelineZoom = {setShowTimelineZoom}
          clipStart = {clipStart}
          setClipStart = {setClipStart}
          clipEnd = {clipEnd}
          setClipEnd = {setClipEnd}
        />
      )}
      <div className="video-trimmer-container mt-5" style={{ userSelect: 'none' }}>
        <button
          id="zoomBtn"
          className="mx-auto btn btn-primary mb-3"
          style={{"backgroundColor": "#212529", "color":"#f8f9fa", padding: "0.5rem 1rem"}}
          onClick={handleTimelineZoom}
        >
          Zoom <i className="fa-solid fa-magnifying-glass"></i>
        </button>
        <div className="timeline-track" ref={timelineRef} style={{ touchAction: 'none', position: 'relative' }}>
          <div 
            className="trimmer-handle left-handle"
            style={{ left: `${(clipStart / duration) * 100}%`, zIndex: getHandleZIndex('start') }}
            onPointerDown={(e) => { e.stopPropagation(); handlePointerDown('start', e); }}
          >
            <div className="handle-grip"></div>
            {isEditingStart ? (
              <input
                type="text"
                inputMode="decimal"
                className={passStartTimeRegexCheck ? "time-label-edit-start" : "time-label-edit-start-error"}
                value={rawClipStartInput}
                onChange={(e) => handleTimeInputChange(e, 0)}
                onBlur={(e) => handleTimeInputConfirm(e, 0)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    (e.target as HTMLElement).blur();
                  }
                }}
                autoFocus
              />
            ) : (
              <span className="time-label" onClick={() => setIsEditingStart(true)}>{formatTime(clipStart)}</span>
            )}
          </div>

          <div className="selected-range" style={getSelectedRangeStyle()} onPointerDown={(e) => handlePointerDown('middle', e)}></div>

          <div 
            className="trimmer-handle right-handle"
            style={{ left: `${(clipEnd / duration) * 100}%`, zIndex: getHandleZIndex('end') }}
            onPointerDown={(e) => { e.stopPropagation(); handlePointerDown('end', e); }}
          >
            <div className="handle-grip"></div>
            {isEditingEnd ? (
              <input
                type="text"
                inputMode="decimal"
                className={passEndTimeRegexCheck ? "time-label-edit-end" : "time-label-edit-end-error"}
                value={rawClipEndInput}
                onChange={(e) => handleTimeInputChange(e, 1)}
                onBlur={(e) => handleTimeInputConfirm(e, 1)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    (e.target as HTMLElement).blur();
                  }
                }}
                autoFocus
              />
            ) : (
              <span className="time-label" onClick={() => setIsEditingEnd(true)}>{formatTime(clipEnd)}</span>
            )}
          </div>
        </div>

        <div className="duration-info">
          Selected: {formatTime(clipEnd - clipStart)} / Total: {formatTime(duration)}
        </div>
      </div>
      <button
        id="uploadBtn"
        className="mx-auto btn btn-primary mt-2 mb-2"
        style={{"backgroundColor": "#f8f9fa", "color":"#000000ff"}}
        onClick={confirmMediaTrim}
        >
          Recognize <i className="fas fa-record-vinyl"></i>
      </button>
    </>
    );
  }

    return (
      <div
        id="mediaTrimmer"
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
          fontSize: "1rem",
          textAlign: "center",
          zIndex: 9999
        }}
      >
        <CloseButton 
          aria-label='Hide'
          variant= 'white'
          style = {{
            zIndex: 200,
            position: 'absolute',
            top: '8px',
            right: '8px',
          }}
          onClick = {handleCloseButtonClick}
        />
        <div style={{ display: 'flex', justifyContent: 'center', width: '100%' }}>
          <h2 style={{  fontSize: '1.2rem', userSelect: 'none' }} className="mt-1" >Select Time Range</h2>
        </div>
        <div style={{ display: 'flex', justifyContent: 'center', width: '100%' }}>
          <video
            ref={videoRef}
            src={URL.createObjectURL(selectedFile!)}
            controls
            className="mt-2"
            style= {{
              maxWidth: '100%',
              maxHeight: isAudioOnly ? '60px' : '480px',
            }} // Max size, responsive
            onLoadedMetadata={handleLoadedMetadata}
          />
        </div>
        <TrimmerHandler
          handleRecognizeFile = {handleRecognizeFile}
        />
      </div>
    )}

  const handleButtonClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];

    if (file) {
      setSelectedFile(file);
      setFileName(file.name);
      setShowMediaTrimmer(true);
      setDisabled(true);
    } else {
      setSelectedFile(null);
      setFileName(null);
    }
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleRecognizeFile = async (startTimeStamp: number, trimmedDuration: number) => {
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
        formData.append('start', startTimeStamp.toString());
        formData.append('duration', trimmedDuration.toString());
        const url = `${config.apiBaseUrl.replace(/\/$/, '')}/api/recognize`;
        const response = await uploadtoAPI(url, formData);
        if (response) {
          const resultToken = response.token;
          if (resultToken !== undefined && resultToken !== null) {
            router.push(`/results/${resultToken}`);
          }
          else {
            setIsError(false);
            setIsWarning(true);
            setSelectedFile(null);
            setFileName(null);
            setTitle(`${config.appName} - ${config.title}`);
            setWarnMsg('Warning: No results were found');
          }
        }
      }
      catch (error: unknown) {
        setTitle(`${config.appName} - ${config.title}`);
        setIsError(true);
        setIsWarning(false);
        if (error instanceof Error) {
          setErrorMsg(error.message);
        } else {
          setErrorMsg(String(error));
        }
        setSelectedFile(null);
        setFileName(null);
      }
      finally {
        setDisabled(false);
      }
    };
  };

  return (
    <>
      {showMediaTrimmer && (
        <MediaTrimmer />
      )}
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
          className="mx-auto btn btn-primary mt-2 mb-2"
          style={{"backgroundColor": "#212529"}}
          disabled={disabled}
          onClick={handleButtonClick}
          >
            Select <i className="fas fa-file-audio"></i> <i className="fas fa-file-video"></i>
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
    </>
  )
}