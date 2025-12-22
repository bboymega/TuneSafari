'use client';
import { useParams, useRouter } from 'next/navigation';
import { useEffect, useState, useRef } from "react";
import ErrorAlert from '@/app/ErrorAlert';
import ResultsModule from './ResultsModule';
import config from '@/app/config.json';
import { RecognitionResponse } from '@/app/page';

export default function ResultPage() {
    const fetchCalled = useRef(false);
    const params = useParams();       // useParams returns { token: string }
    const token = params.token;
    const router = useRouter();
    const [title, setTitle] = useState(`${config.appName} - Loading Results...`);
    const [errorMsg, setErrorMsg] = useState("");
    const [isError, setIsError] = useState(false);
    const [resultsJson, setResultsJson] = useState<RecognitionResponse>({ results: [], status: "", token: "" });
    const [isResultsFetched, setIsResultsFetched] = useState(false);
    const currentYear = new Date().getFullYear();
    const [progress, setProgress] = useState(20);
    const [showProgress, setShowProgress] = useState(false);

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
                aria-valuemin={0}
                aria-valuemax={100}
            >
            </div>
        </div>
        )
    }

    const handleResultFetch = async () => {
        const url = `${config.apiBaseUrl.replace(/\/$/, '')}/api/fetch/${token}`;
        setShowProgress(true);
        setProgress(66);
        await fetch(url)
            .then(response => 
                response.json().then(jsonData => {
                    setProgress(85);
                    const errorMessage =
                        jsonData.error ??
                        jsonData.status ??
                        null;
                    if (!response.ok) {
                        setProgress(100);
                        setShowProgress(false);
                        return Promise.reject(new Error(`${response.status} ${errorMessage}`));
                    }
                    return jsonData;
                })
            )
            .then(jsonData => {
                setProgress(95);
                setResultsJson(jsonData);
                setIsResultsFetched(true);
                setProgress(100);
                setTimeout(() => setShowProgress(false), 500);
            })
            .catch(error => {
                if (error instanceof TypeError) {
                    setErrorMsg('Error: Backend not available');
                } else {
                    setErrorMsg(`${error}`);
                }
                setIsError(true);
                setProgress(100);
                setShowProgress(false);
                setTimeout(() => router.push('/'), 3500);
            });
    };
        
    useEffect(() => {
        if (!fetchCalled.current) {
            handleResultFetch();
            fetchCalled.current = true;
        }
    }, []); // calls backend only once

    useEffect(() => {
        if (title) {
        document.title = title;
        }
    }, [title]);
    
    return (
    <>
        <div id="main">
            {showProgress &&
            <ProgressBar/>
            }
            <div className="masthead">
                <div className="container px-4 px-lg-5 d-flex h-100 align-items-center justify-content-center">
                    <div className="d-flex justify-content-center" id="mainDiv">
                        {isError && (
                            <ErrorAlert
                                message={errorMsg}
                                onClose={() => setIsError(false)}
                            />
                        )}
                        {isResultsFetched && (
                            <ResultsModule
                                resultsJson={resultsJson}
                                setProgress={setProgress}
                                setShowProgress={setShowProgress}
                            />
                        )}
                    </div>
                </div>
            </div>
            <footer className="footer bg-black small text-center text-white-50"><div className="container px-4 px-lg-5">Copyright &copy; {config.appName} {currentYear}</div></footer>
        </div>
    </>
  );
}