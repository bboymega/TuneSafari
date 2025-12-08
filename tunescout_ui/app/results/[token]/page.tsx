'use client';
import { useParams } from 'next/navigation';
import { useRouter } from 'next/navigation';
import { useEffect, useState, useRef } from "react";
import 'bootstrap/dist/css/bootstrap.min.css';
import 'typeface-roboto';
import '@fortawesome/fontawesome-free/css/all.min.css';

export default function ResultPage() {
    const [results, setResults] = useState(null);
    const fetchCalled = useRef(false);
    const params = useParams();       // useParams returns { token: string }
    const token = params.token;
    const router = useRouter();

    function createErrorAlert(message) {
        if (document.getElementById('errorAlert')) {
            document.getElementById('errorAlert').remove();
        }
        if (document.getElementById('successAlert')) {
            document.getElementById('successAlert').remove();
        }
        const errorAlert = document.createElement('div');
        errorAlert.className = 'alert alert-danger';
        errorAlert.id = 'errorAlert'
        errorAlert.role = 'alert';
        errorAlert.style = "position:absolute; z-index:9999; transition:opacity 0.5s ease; opacity:1;";
        errorAlert.innerHTML = message;
        setTimeout(() => {
        errorAlert.style.opacity = "0";
        setTimeout(() => errorAlert.remove(), 500);
        }, 3000);
        return errorAlert;
    };

    function createSuccessAlert(message) {
        if (document.getElementById('errorAlert')) {
            document.getElementById('errorAlert').remove();
        }
        if (document.getElementById('successAlert')) {
            document.getElementById('successAlert').remove();
        }
        const successAlert = document.createElement('div');
        successAlert.className = 'alert alert-success';
        successAlert.id = 'successAlert'
        successAlert.role = 'alert';
        successAlert.style = "position:absolute; z-index:9999; transition:opacity 0.5s ease; opacity:1;";
        successAlert.innerHTML = message;
        setTimeout(() => {
        successAlert.style.opacity = "0";
        setTimeout(() => successAlert.remove(), 500);
        }, 1000);
        return successAlert;
    }

    const handleResultFetch = () => {
    const url = `http://172.16.241.129:8080/api/fetch/${token}`;
    fetch(url)
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
            setResults(jsonData);
            for (let index = 0; index < jsonData.results.length; index++) {
                const resultBox = document.createElement('div')
                resultBox.className = 'list-group-item list-group-item-action flex-column align-items-start active d-flex w-100 justify-content-between';
                resultBox.style.fontFamily = '"OPTICopperplate-Light", sans-serif';
                resultBox.style.color = '#fff';
                resultBox.style.backgroundColor = '#000000ff';

                const indexCount = document.createElement('small');
                indexCount.textContent = `#${index + 1}`

                const songTitle = document.createElement('h5');
                songTitle.className = 'mb-1';
                songTitle.style.textAlign = 'center';
                songTitle.style.whiteSpace = 'normal';
                songTitle.style.overflowWrap = 'break-word';
                songTitle.textContent = jsonData.results[index].song_name;

                const searchPrompt = document.createElement('small');
                searchPrompt.textContent = 'Search on: ';

                const buttonBox = document.createElement('span')
                buttonBox.className = 'mt-3';
                buttonBox.style.display = 'flex';
                buttonBox.style.flexWrap = 'wrap';       // allows buttons to wrap to next line
                buttonBox.style.gap = '10px';            // spacing between buttons
                buttonBox.style.justifyContent = 'center'; // center buttons horizontally

                const sanitizedSearch = encodeURIComponent(jsonData.results[index].song_name.replace(/[_-]/g, ' '));
                // Google search button
                const searchBtn = document.createElement('a');
                searchBtn.type = 'button';
                searchBtn.className = 'btn btn-light';
                searchBtn.style.padding = '4px 8px';
                searchBtn.textContent = 'Google';
                searchBtn.target = '_blank';
                searchBtn.href = `https://www.google.com/search?q=${sanitizedSearch}`;

                // Spotify search button
                const spBtn = document.createElement('a');
                spBtn.type = 'button';
                spBtn.className = 'btn btn-light';
                spBtn.style.padding = '4px 8px';
                spBtn.textContent = 'Spotify';
                spBtn.target = '_blank';
                spBtn.href = `https://open.spotify.com/search/${sanitizedSearch}`;

                // Apple Music search button
                const apBtn = document.createElement('a');
                apBtn.type = 'button';
                apBtn.className = 'btn btn-light';
                apBtn.style.padding = '4px 8px';
                apBtn.textContent = 'Apple Music';
                apBtn.target = '_blank';
                apBtn.href = `https://music.apple.com/us/search?term=${sanitizedSearch}`;

                // YouTube search button
                const ytBtn = document.createElement('a');
                ytBtn.type = 'button';
                ytBtn.className = 'btn btn-light';
                ytBtn.style.padding = '4px 8px';
                ytBtn.textContent = 'YouTube';
                ytBtn.target = '_blank';
                ytBtn.href = `https://www.youtube.com/results?search_query=${sanitizedSearch}`;

                // SoundCloud search button
                const scBtn = document.createElement('a');
                scBtn.type = 'button';
                scBtn.className = 'btn btn-light';
                scBtn.style.padding = '4px 8px';
                scBtn.textContent = 'SoundCloud';
                scBtn.target = '_blank';
                scBtn.href = `https://soundcloud.com/search?q=${sanitizedSearch}`;

                // YT Music search button
                const ytMusicBtn = document.createElement('a');
                ytMusicBtn.type = 'button';
                ytMusicBtn.className = 'btn btn-light';
                ytMusicBtn.style.padding = '4px 8px';
                ytMusicBtn.textContent = 'YT Music';
                ytMusicBtn.target = '_blank';
                ytMusicBtn.href = `https://music.youtube.com/search?q=${sanitizedSearch}`;

                // Bandcamp search button
                const bcBtn = document.createElement('a');
                bcBtn.type = 'button';
                bcBtn.className = 'btn btn-light';
                bcBtn.style.padding = '4px 8px';
                bcBtn.textContent = 'Bandcamp';
                bcBtn.target = '_blank';
                bcBtn.href = `https://bandcamp.com/search?q=${sanitizedSearch}`;

                // Copy name button
                const cpBtn = document.createElement('button');
                cpBtn.type = 'button';
                cpBtn.className = 'btn btn-dark';
                cpBtn.style.padding = '4px 8px';
                cpBtn.innerHTML = 'Copy Name <i class="fa-solid fa-copy"></i>';

                cpBtn.addEventListener('click', () => {
                    if (typeof navigator !== 'undefined' && navigator.clipboard) {
                        navigator.clipboard.writeText(jsonData.results[index].song_name.replace(/[_-]/g, ' '))
                        .then(() => {
                            const successAlert = createSuccessAlert('Copied to clipboard');
                            document.getElementById("mainDiv").appendChild(successAlert);
                        })
                        .catch(err => {
                            const errorAlert = createErrorAlert('Error: Failed to copy');
                            document.getElementById("mainDiv").appendChild(errorAlert);
                        });
                    } else {
                        const errorAlert = createErrorAlert('Error: Clipboard API not available');
                        document.getElementById("mainDiv").appendChild(errorAlert);
                    }
                });

                buttonBox.appendChild(searchPrompt);
                buttonBox.appendChild(searchBtn);
                buttonBox.appendChild(spBtn);
                buttonBox.appendChild(apBtn);
                buttonBox.appendChild(ytBtn);
                buttonBox.appendChild(scBtn);
                buttonBox.appendChild(ytMusicBtn);
                buttonBox.appendChild(bcBtn);
                buttonBox.appendChild(cpBtn);
                resultBox.appendChild(indexCount);
                resultBox.appendChild(songTitle);
                resultBox.appendChild(buttonBox);
                document.getElementById('resultsList').appendChild(resultBox);
            }
            const returnBtnDiv = document.createElement('div');
            returnBtnDiv.className = 'mx-auto mt-4';
            const returnBtn = document.createElement('button');
            returnBtn.className = 'btn btn-dark';
            returnBtn.innerHTML = 'Return <i class="fa-solid fa-circle-chevron-left"></i>';
            returnBtn.style.padding = '20px 40px';
            returnBtnDiv.appendChild(returnBtn);
            document.getElementById('panel').appendChild(returnBtnDiv);
            returnBtn.onclick = () => {
                router.push('/');
            };

        })
        .catch(error => {
            const errorAlert = createErrorAlert('Error: Backend not reachable. <strong>Redirecting...</strong>');
            document.getElementById("mainDiv").appendChild(errorAlert);
            setTimeout(() => router.push('/'), 3500);
        });
    };

    useEffect(() => {
        if (!fetchCalled.current) {
            handleResultFetch();
            fetchCalled.current = true;
        }
    }, []); // calls backend only once
    
    return (
        <div id="main">
            <meta charSet="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no" />
            <meta name="description" content="" />
            <meta name="author" content="" />
            <link rel="icon" type="image/x-icon" href="../assets/favicon.ico" />
            <link href="../css/styles.css" rel="stylesheet" />
            <header className="masthead">
                <div className="container px-4 px-lg-5 d-flex h-100 align-items-center justify-content-center">
                    <div className="d-flex justify-content-center" id="mainDiv">
                        <div className="text-center" id="panel">
                            <h1 className="mx-auto my-0 mt-2 mb-5 text-uppercase">TuneScout</h1>
                            <h2 className="mx-auto mt-2 mb-4">Possible Results</h2>
                            <div className="list-group" id="resultsList">
                            </div>
                        </div>
                    </div>
                </div>
            </header>
            <footer className="footer bg-black small text-center text-white-50"><div className="container px-4 px-lg-5">Copyright &copy; TuneScout 2025</div></footer>
        </div>
  );
}