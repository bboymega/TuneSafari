const fileInputTmp = document.getElementById('fileInputTmp');
const fileList = document.getElementById('fileList');
let allFiles = new DataTransfer();

fetch('./config.json')
    .then(response => response.json()) 
    .then(config => {
        if (config.appName) {
            const currentYear = new Date().getFullYear();
            document.getElementById('appName').innerHTML = config.appName + '</br>Uploader';
            document.title = config.appName + ' Uploader';
            document.getElementById('footer').innerHTML = `Copyright &copy; ${config.appName} ${currentYear}`;
        }

        if(config.themeColor) document.querySelector('meta[name="theme-color"]').setAttribute('content', config.themeColor);
    })
    .catch(error => {
        console.error(error);
    });

function appendFiles() {
    document.getElementById("uploadFiles").innerHTML = `Fingerprint <i class="fa-solid fa-upload">`;
    resetFilelistStatus();
    const fileInputTmp = document.createElement('input');
    fileInputTmp.type = 'file';
    fileInputTmp.accept = 'audio/*,video/*';
    fileInputTmp.multiple = true;

    fileInputTmp.onchange = e => {
        const newFiles = e.target.files;
        if (newFiles.length === 0) return;

        fileListContainer.hidden = false;
        document.getElementById('appendFiles').disabled = true;

        Array.from(newFiles).forEach(file => {
            allFiles.items.add(file);

            const li = document.createElement('li');
            li.className = 'list-group-item d-flex justify-content-between align-items-center text-start';

            const sizeInKB = file.size / 1024;
            const sizeText = sizeInKB >= 1024 
                ? `${(sizeInKB / 1024).toFixed(1)} MB` 
                : `${sizeInKB.toFixed(1)} KB`;

            li.innerHTML = `
                <div class="flex-grow-1">
                    <div class="fw-bold text-truncate" style="max-width: min(250px, 50vw);">${file.name}</div>
                    <small class="text-muted"><span class="filesize-text">${sizeText}</span><span class="progress-text" hidden> • Uploading 0%</span></small>
                </div>
                <button type="button" class="btn-close ms-2" aria-label="Close"></button>
            `;

            // Remove a file from the list when the remove button is clicked
            li.querySelector('.btn-close').addEventListener('click', () => {
                const dt = new DataTransfer();
                Array.from(allFiles.files).forEach(f => {
                    if (f !== file) dt.items.add(f);
                });
                    
                allFiles = dt;
                li.remove();
                document.getElementById('fileInput').files = allFiles.files;

                if (allFiles.files.length === 0) {
                    document.getElementById("uploadFiles").innerHTML = `Fingerprint <i class="fa-solid fa-upload">`;
                    fileListContainer.hidden = true;
                    document.getElementById('appendFiles').disabled = false;
                }
                if (fileList.children.length === 0) {
                    document.getElementById("uploadFiles").innerHTML = `Fingerprint <i class="fa-solid fa-upload">`;
                    fileListContainer.hidden = true;
                    document.getElementById('appendFiles').disabled = false;
                }
            });

            fileList.appendChild(li);
        });

        document.getElementById('fileInput').files = allFiles.files;

        if (fileList.children.length === 0) {
            document.getElementById("uploadFiles").innerHTML = `Fingerprint <i class="fa-solid fa-upload">`;
            fileListContainer.hidden = true;
            document.getElementById('appendFiles').disabled = false;
        }

    };
    fileInputTmp.click();
}

function clearList() {
    const dt = new DataTransfer();
    allFiles = dt;
    document.getElementById('fileInput').files = allFiles.files;
    document.getElementById("uploadFiles").innerHTML = `Fingerprint <i class="fa-solid fa-upload">`;
    fileList.innerHTML = '';
    fileListContainer.hidden = true;
    document.getElementById('appendFiles').disabled = false;
}

function connectionSettings() {
    const connectionSettings = document.createElement('div');
    connectionSettings.id = 'connectionSettings';

    Object.assign(connectionSettings.style, {
        fontFamily: '"OPTICopperplate-Light", sans-serif',
        position: 'fixed',
        top: '50%',
        left: '50%',
        transform: 'translate(-50%, -50%)',
        backgroundColor: 'rgba(0,0,0,0.9)',
        color: '#fff',
        padding: '20px',
        borderRadius: '5px',
        textAlign: 'center',
        zIndex: '9999',
        maxWidth: 'min(90vw, 480px)'
    });

    fetch("connectionSettings.html")
    .then(response => response.text())
    .then(html => {
        connectionSettings.innerHTML = html;
        connectionSettings.appendChild(closeButton);
        setTimeout(() => {
            loadCredential();
        }, 10);
    });

    const closeButton = document.createElement('button');
    closeButton.type = 'button';
    closeButton.className = 'btn-close btn-close-white'; 
    closeButton.setAttribute('aria-label', 'Hide');

    Object.assign(closeButton.style, {
        zIndex: '200',
        position: 'absolute',
        top: '8px',
        right: '8px'
    });

    closeButton.onclick = function() {
        connectionSettings.remove();
    };

    document.getElementById('mainDiv').appendChild(connectionSettings);
        
}
    
function showAlert(message, type, alertId) {
    const alertPlaceholder = document.getElementById(alertId);
    if(alertPlaceholder.innerHTML!='') {
        alertPlaceholder.innerHTML='';
    }
    const wrapper = document.createElement('div');
        
    wrapper.innerHTML = [
        `<div class="alert alert-${type} alert-dismissible fade show" role="alert">`,
        `   <div>${message}</div>`,
        '</div>'
    ].join('');

    alertPlaceholder.append(wrapper);
        
    if(type === "success") {
        setTimeout(() => {
            const alertElement = wrapper.querySelector('.alert');
            if (alertElement) {
                alertElement.classList.remove('show');
                setTimeout(() => {
                    wrapper.remove();
                }, 500); 
            }
        }, 1500);
    } else {
        setTimeout(() => {
            const alertElement = wrapper.querySelector('.alert');
            if (alertElement) {
                alertElement.classList.remove('show');
                setTimeout(() => {
                    wrapper.remove();
                }, 500); 
            }
        }, 3000);
    }
}

function saveCredential() {
    try {
        const settings = {
            protocol: document.getElementById("protocol").value,
            server: document.getElementById("server").value,
            port: document.getElementById("port").value,
            token: document.getElementById("token").value
        };

        if (!settings.server) {
            showAlert("Please enter a server address!", "danger", "alertPlaceholderSettings");
            return;
        }

        localStorage.setItem("connectionSettings", JSON.stringify(settings));
        showAlert("Settings saved successfully!", "success", "alertPlaceholderSettings");
        document.getElementById("connectionSettings").remove();
            
    } catch (error) {
        console.error("Error loading settings: " + error.message);
    }
}

function loadCredential() {
    const protocolSelect = document.getElementById('protocol');
    const portInput = document.getElementById('port');

    protocolSelect.addEventListener('change', function() {
        if (this.value === 'https') {
            portInput.value = 443;
            portInput.placeholder = "443";
        } else if (this.value === 'http') {
            portInput.value = 80;
            portInput.placeholder = "80";
        }
    });
        
    try {
        const savedSettings = localStorage.getItem("connectionSettings");

        if (savedSettings) {
            const settings = JSON.parse(savedSettings);
            if (settings.protocol) document.getElementById("protocol").value = settings.protocol;
            if (settings.server) document.getElementById("server").value = settings.server;
            if (settings.port) document.getElementById("port").value = settings.port;
            if (settings.token) document.getElementById("token").value = settings.token;
        }
    } catch (error) {
        console.error("Error loading settings: " + error.message);
    }
}

function uploadFile(url, file, fileListItem, token) {
    return new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        const fileList = document.getElementById("fileList");
        const fileListContainer = document.getElementById("fileListContainer");

        xhr.upload.addEventListener("progress", (event) => {
            if (event.lengthComputable) {
                const percentComplete = (event.loaded / event.total) * 100;
                fileListItem.style.background = `linear-gradient(to right, rgba(0,0,0,0.15) ${Math.round(percentComplete)}%, transparent ${Math.round(percentComplete)}%)`;
                const percentLabel = fileListItem.querySelector('.progress-text');
                if (percentLabel) {
                    percentLabel.hidden = false;
                    percentLabel.textContent = ` • Uploading ${Math.round(percentComplete)}%`;
                }
            }
        });

        xhr.upload.addEventListener("load", (event) => {
            fileListItem.style.background = "rgba(0,0,0,0.15)";
            const percentLabel = fileListItem.querySelector('.progress-text');
            if (percentLabel) {
                percentLabel.hidden = false;
                percentLabel.textContent = " • Fingerprinting";
            }
        })

        xhr.addEventListener("load", () => {
            if (xhr.status >= 200 && xhr.status < 300) {
                fileListItem.style.background = "rgba(25, 135, 84, 0.25)";
                const percentLabel = fileListItem.querySelector('.progress-text');
                const fileSizeLabel = fileListItem.querySelector('.filesize-text');
                if (percentLabel) {
                    percentLabel.hidden = false;
                    fileSizeLabel.style.color = "rgb(25, 135, 84)";
                    percentLabel.style.color = "rgb(25, 135, 84)";
                    percentLabel.textContent = " • Completed";
                }

                resolve(xhr.response);

            } else if (xhr.status == 409) {
                fileListItem.style.background = "rgba(255, 193, 7, 0.25)";
                const percentLabel = fileListItem.querySelector('.progress-text');
                const fileSizeLabel = fileListItem.querySelector('.filesize-text');
                if (percentLabel) {
                    percentLabel.hidden = false;
                    try {
                        const response = JSON.parse(xhr.responseText);
                        errorMessage = response.message || `${xhr.status}`;
                    } catch (e) {
                        errorMessage = xhr.statusText || `${xhr.status}`;
                    }
                    fileSizeLabel.style.color = "rgb(255, 145, 2)";
                    percentLabel.style.color = "rgb(255, 145, 2)";
                    percentLabel.textContent = ` • ${errorMessage}`;
                }

                resolve(xhr.response);
            }
            else {
                fileListItem.style.background = "rgba(248, 215, 218, 1)";
                const percentLabel = fileListItem.querySelector('.progress-text');
                const fileSizeLabel = fileListItem.querySelector('.filesize-text');
                if (percentLabel) {
                    percentLabel.hidden = false;
                    try {
                        const response = JSON.parse(xhr.responseText);
                        errorMessage = response.message || `${xhr.status}`;
                    } catch (e) {
                        errorMessage = xhr.statusText || `${xhr.status}`;
                    }
                    fileSizeLabel.style.color = "rgb(220, 53, 69)";
                    percentLabel.style.color = "rgb(220, 53, 69)";
                    percentLabel.textContent = ` • ${errorMessage}`;
                    reject(new Error(errorMessage));

                }

                reject(new Error(`Upload failed with status: ${xhr.status}`));
            }
        });

        xhr.addEventListener("error", () => {
            fileListItem.style.background = "rgba(248, 215, 218, 1)";
            const percentLabel = fileListItem.querySelector('.progress-text');
            const fileSizeLabel = fileListItem.querySelector('.filesize-text');
            if (percentLabel) {
                percentLabel.hidden = false;
                fileSizeLabel.style.color = "rgb(220, 53, 69)";
                percentLabel.style.color = "rgb(220, 53, 69)";
                percentLabel.textContent = ` • Network error`;
            }
            reject(new Error("Network Error"));
        });

        xhr.addEventListener("abort", () => {
            fileListItem.style.background = "rgba(248, 215, 218, 1)";
            const percentLabel = fileListItem.querySelector('.progress-text');
            const fileSizeLabel = fileListItem.querySelector('.filesize-text');
            if (percentLabel) {
                percentLabel.hidden = false;
                fileSizeLabel.style.color = "rgb(220, 53, 69)";
                percentLabel.style.color = "rgb(220, 53, 69)";
                percentLabel.textContent = ` • Upload aborted`;
            }
            reject(new Error("Upload Aborted"))
        });

        xhr.open("POST", url);
        const formData = new FormData();
        formData.append("file", file);
        xhr.setRequestHeader("Authorization", `Bearer ${token}`);
        xhr.send(formData);
    });
}
    
function resetFilelistStatus() {
    const fileListItems = document.getElementById("fileList").children;
    const fileInput = document.getElementById("fileInput");
    const dt = new DataTransfer();

    const itemsArray = Array.from(fileListItems);
        
    itemsArray.forEach((fileListItem, index) => {
        fileListItem.style.background = "";
        const percentLabel = fileListItem.querySelector('.progress-text');
        const fileSizeLabel = fileListItem.querySelector('.filesize-text');

        if (percentLabel) {
            if (percentLabel.textContent === " • Completed" || percentLabel.textContent === " • Already fingerprinted") {
                fileListItem.remove();
            } else {
                const file = fileInput.files[index];
                if (file) {
                    dt.items.add(file);
                    percentLabel.hidden = true;
                    fileSizeLabel.style.color = "";
                    percentLabel.style.color = "";
                    percentLabel.textContent = " • 0%";
                }
            }
        }
    });

    fileInput.files = dt.files;
    allFiles.items.clear();
    Array.from(dt.files).forEach(file => {
        allFiles.items.add(file);
    });
    if(fileInput.files.length == 0) {
        document.getElementById("uploadFiles").innerHTML = `Fingerprint <i class="fa-solid fa-upload">`;
        fileList.innerHTML = '';
        fileListContainer.hidden = true;
        document.getElementById('appendFiles').disabled = false;
    }
}

async function handleUploadFiles() {
    try {
        document.getElementById("uploadFiles").innerHTML = `Fingerprint <i class="fa-solid fa-upload">`;
        const closeButtons = document.querySelectorAll('#fileList .btn-close');
        closeButtons.forEach(btn => btn.disabled = true);
        document.getElementById("connectionSettingsBtn").disabled = true;
        document.getElementById("appendFiles").disabled = true;
        document.getElementById("appendFilesInContainer").disabled = true;
        document.getElementById("uploadFiles").disabled = true;
        document.getElementById("clearList").disabled = true;

        const savedSettings = localStorage.getItem("connectionSettings");

        if (savedSettings) {
            let protocol;
            let server;
            let port;
            let token;
            const settings = JSON.parse(savedSettings);
            if (settings.protocol) protocol = settings.protocol;
            if (settings.server) server = settings.server;
            if (settings.port) port = settings.port;
            if (settings.token) token = settings.token;
                
            const url = protocol + "://" + server + ":" + port + "/api/fingerprint";
            resetFilelistStatus();
            const fileInput = document.getElementById('fileInput');

            if(fileInput.files.length == 0) {
                document.getElementById("connectionSettingsBtn").disabled = false;
                document.getElementById("appendFiles").disabled = false;
                document.getElementById("appendFilesInContainer").disabled = false;
                document.getElementById("uploadFiles").disabled = false;
                document.getElementById("uploadFiles").innerHTML = `Fingerprint <i class="fa-solid fa-upload">`;
                document.getElementById("clearList").disabled = false;
                fileList.innerHTML = '';
                fileListContainer.hidden = true;
                document.getElementById('appendFiles').disabled = false;
                return;
            }

            const files = fileInput.files;
            const fileListItems = document.getElementById("fileList").children;
            let noError = true;

            for(let i = 0; i < files.length; i++) {
                const targetItem = fileListItems[i]; 
                try {
                    await uploadFile(url, files[i], targetItem, token);
                } catch (err) {
                    noError = false;
                    console.error("Upload failed for file " + i, err);
                }
            }
            closeButtons.forEach(btn => btn.disabled = false);
            document.getElementById("connectionSettingsBtn").disabled = false;
            document.getElementById("appendFiles").disabled = false;
            document.getElementById("appendFilesInContainer").disabled = false;
            document.getElementById("uploadFiles").disabled = false;
            if(noError) document.getElementById("uploadFiles").innerHTML = `Done <i class="fa-solid fa-check"></i>`;
            else document.getElementById("uploadFiles").innerHTML = `Retry <i class="fa-solid fa-arrow-rotate-right"></i>`;
            document.getElementById("clearList").disabled = false;
        }
    } catch (error) {
        const closeButtons = document.querySelectorAll('#fileList .btn-close');
        closeButtons.forEach(btn => btn.disabled = false);
        console.error("Error loading settings: " + error.message);
        document.getElementById("connectionSettingsBtn").disabled = false;
        document.getElementById("appendFiles").disabled = false;
        document.getElementById("appendFilesInContainer").disabled = false;
        document.getElementById("uploadFiles").disabled = false;
        document.getElementById("clearList").disabled = false;
    }
}
