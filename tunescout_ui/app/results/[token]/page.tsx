'use client';
import { useParams, useRouter } from 'next/navigation';
import { useEffect, useState, useRef } from "react";
import ErrorAlert from '@/app/ErrorAlert';
import ResultsModule from './ResultsModule';
import config from '@/app/config.json';

export default function ResultPage() {
    const fetchCalled = useRef(false);
    const params = useParams();       // useParams returns { token: string }
    const token = params.token;
    const router = useRouter();
    const [title, setTitle] = useState('TuneScout - Loading Results...');
    const [errorMsg, setErrorMsg] = useState("");
    const [isError, setIsError] = useState(false);
    const [resultsJson, setResultsJson] = useState("");
    const [isResultsFetched, setIsResultsFetched] = useState(false);

    const handleResultFetch = async () => {
        const url = `${config.api_base_url.replace(/\/$/, '')}/api/fetch/${token}`;
        await fetch(url)
            .then(response => 
                response.json().then(jsonData => {
                    const errorMessage =
                        jsonData.error ??
                        jsonData.status ??
                        null;
                    if (!response.ok) {
                        return Promise.reject(new Error(`${response.status} ${errorMessage}`));
                    }
                    return jsonData;
                })
            )
            .then(jsonData => {
                setResultsJson(jsonData);
                setIsResultsFetched(true);
            })
            .catch(error => {
                setErrorMsg('Error: Backend not reachable. Redirecting...');
                setIsError(true);
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
                            />
                        )}
                    </div>
                </div>
            </div>
            <footer className="footer bg-black small text-center text-white-50"><div className="container px-4 px-lg-5">Copyright &copy; TuneScout 2025</div></footer>
        </div>
    </>
  );
}