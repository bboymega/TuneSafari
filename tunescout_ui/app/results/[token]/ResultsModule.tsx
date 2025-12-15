'use client';
import { useRouter } from 'next/navigation';
import { useEffect, useState } from 'react';
import ErrorAlert from '@/app/ErrorAlert';
import SuccessAlert from '@/app/SuccessAlert';
import config from '@/app/config.json';

type SongResult = {
  song_name: string;
};

type ResultsModuleProps = {
  resultsJson: { results: SongResult[] };
  setProgress;
  setShowProgress;
};

export default function ResultsModule({ resultsJson, setProgress, setShowProgress }: ResultsModuleProps) {
  const router = useRouter();
  const [errorMsg, setErrorMsg] = useState("");
  const [isError, setIsError] = useState(false);
  const [successMsg, setSuccessMsg] = useState("");
  const [isSuccess, setIsSuccess] = useState(false);

  //Title of page: Results • {time}{title}
  const formatOffset = (sec: number) => {
    const minutes = Math.floor(sec / 60);
    const seconds = sec % 60;
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  const resultTitle = resultsJson.results
  .map((r) => `(${formatOffset(Math.round(r.offset_seconds))}) ${r.song_name}`)
  .join(' • ');

  useEffect(() => {
    document.title = `Results • ${resultTitle}`;
  }, [resultTitle]);

  const handleCopyText = (text: string) => {
    if (navigator?.clipboard) {
      navigator.clipboard.writeText(text)
        .then(() => {
          setIsSuccess(true);
          setIsError(false);
          setSuccessMsg('Copied to clipboard');
          setErrorMsg('');
        })
        .catch(() => {
          setIsSuccess(false);
          setIsError(true);
          setSuccessMsg('');
          setErrorMsg('Error: Failed to copy');
        })
    } else {
      setIsSuccess(false);
      setIsError(true);
      setSuccessMsg('');
      setErrorMsg('Error: Clipboard API not available');
    }
  };

  const platforms = [
    { name: 'Google', url: (q: string) => `https://www.google.com/search?q=${q}` },
    { name: 'Spotify', url: (q: string) => `https://open.spotify.com/search/${q}` },
    { name: 'Apple Music', url: (q: string) => `https://music.apple.com/us/search?term=${q}` },
    { name: 'YouTube', url: (q: string) => `https://www.youtube.com/results?search_query=${q}` },
    { name: 'SoundCloud', url: (q: string) => `https://soundcloud.com/search?q=${q}` },
    { name: 'YT Music', url: (q: string) => `https://music.youtube.com/search?q=${q}` },
    { name: 'Bandcamp', url: (q: string) => `https://bandcamp.com/search?q=${q}` },
  ];

  return (
    <>
      {isError && (
        <ErrorAlert
          message={errorMsg}
          onClose={() => setIsError(false)}
        />
      )}
      {isSuccess && (
        <SuccessAlert
          message={successMsg}
          onClose={() => setIsSuccess(false)}
        />
      )}
      <div className="text-center" id="panel">
        <h1 className="mx-auto my-0 mt-2 mb-5 text-uppercase">{config.appName}</h1>
        <h2 className="mx-auto mt-2 mb-4">Possible Results</h2>
        <div id="resultsList" className="list-group">
          {resultsJson.results.map((result, index) => {
            const sanitizedSearch = encodeURIComponent(result.song_name.replace(/[_-]/g, ' '));
            const offsetSec = Math.round(result.offset_seconds);
            return (
              <div
                key={index}
                className="list-group-item list-group-item-action flex-column align-items-start active d-flex w-100 justify-content-between"
                style={{
                  fontFamily: '"OPTICopperplate-Light", sans-serif',
                  color: '#fff',
                  backgroundColor: '#000000ff',
                }}
              >
                <small>#{index + 1} ⬤ Found at {Math.floor(offsetSec/60)}:{(offsetSec%60).toString().padStart(2,"0")}</small>
                <small><br/></small>
                <h5
                  className="mb-1"
                  style={{
                    textAlign: 'center',
                    whiteSpace: 'normal',
                    overflowWrap: 'break-word',
                  }}
                >
                  {result.song_name}
                </h5>

                <span
                  className="mt-4 mb-1"
                  style={{
                    display: 'flex',
                    flexWrap: 'wrap',
                    gap: '6px',
                    justifyContent: 'center',
                  }}
                >
                  <small>Search on: </small>
                  {platforms.map((p) => (
                    <a
                      key={p.name}
                      className="btn btn-light"
                      style={{ padding: '3px 6px' }}
                      target="_blank"
                      rel="noopener noreferrer"
                      href={p.url(sanitizedSearch)}
                    >
                      {p.name}
                    </a>
                  ))}
                  <button
                    className="btn btn-dark"
                    style={{ padding: '3px 6px' }}
                    onClick={() => handleCopyText(result.song_name.replace(/[_-]/g, ' '))}
                  >
                    Copy Title <i className="fa-solid fa-copy"></i>
                  </button>
                </span>
              </div>
            );
          })}
        </div>

        {/* Bottom buttons */}
        <div
          className="mx-auto mt-4"
          style={{ display: 'inline-flex', gap: '15px' }}
        >
          <button
            className="btn btn-dark mx-auto mt-4 mb-2"
            style={{ padding: '10px 20px' }}
            onClick={() => 
              {
                router.push('/');
                setShowProgress(true);
                setProgress(66);
              }}
          >
            Return <i className="fa-solid fa-circle-chevron-left"></i>
          </button>
          <button
            className="btn btn-light mx-auto mt-4 mb-2"
            style={{ padding: '10px 20px' }}
            onClick={() => handleCopyText(window.location.href)}
          >
            Copy Result Link <i className="fa-solid fa-copy"></i>
          </button>
        </div>
      </div>
    </>
  );
}
