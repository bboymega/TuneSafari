import { Hono } from 'hono';

export interface Env {}

const app = new Hono<{ Bindings: Env }>();

app.get('/install-api', (c) => {
  const service_name = c.req.query('service_name') || 'tunescout_api';
  const daemon_user = c.req.query('daemon_user') || 'www-data';
  const daemon_group = c.req.query('daemon_group') || 'www-data';
  const installation_path = c.req.query('installation_path') || '/var/www/tunescout_api';
  const release_url = 'https://github.com/bboymega/TuneScout/releases/download/v1.0.0/tunescout_api.tar.gz';
  const gunicorn_workers = c.req.query('workers') || '10';
  const gunicorn_timeout = c.req.query('timeout') || '600';
  const gunicorn_bind = c.req.query('bind') || '127.0.0.1:50080';
  const systemd_path = c.req.query('systemd_path') + `/${service_name}.service` || `/etc/systemd/system/${service_name}.service`;

  const script = `#!/usr/bin/env bash
set -e

# --- Generated Configuration ---
SERVICE_NAME="${service_name}"
DAEMON_USER="${daemon_user}"
DAEMON_GROUP="${daemon_group}"
INSTALLATION_PATH="${installation_path}"
RELEASE_URL="${release_url}"
GUNICORN_WORKERS=${gunicorn_workers}
GUNICORN_TIMEOUT=${gunicorn_timeout}
GUNICORN_BIND="${gunicorn_bind}"
SYSTEMD_PATH="${systemd_path}"

if [[ $EUID -ne 0 ]]; then
  echo "This script must be run with sudo."
  exit 1
fi


if [[ ! -d "$INSTALLATION_PATH" ]]; then
  echo "Creating and setting permissions for: $INSTALLATION_PATH"
  mkdir -p "$INSTALLATION_PATH"
  chown -R "$DAEMON_USER:$DAEMON_GROUP" "$INSTALLATION_PATH"
fi


echo "Downloading and extracting release..."
curl -sSL "$RELEASE_URL" | sudo -u "$DAEMON_USER" tar -xzf - --strip-components=1 -C "$INSTALLATION_PATH"


cd "$INSTALLATION_PATH"
if [[ ! -d "venv" ]]; then
  echo "Creating virtual environment..."
  sudo -u "$DAEMON_USER" python3 -m venv venv
fi


echo "Installing/Updating requirements..."
sudo -u "$DAEMON_USER" ./venv/bin/pip install --upgrade pip
if [[ -f "requirements.txt" ]]; then
  sudo -u "$DAEMON_USER" ./venv/bin/pip install -r requirements.txt
fi

sudo -u "$DAEMON_USER" ./venv/bin/pip install gunicorn

echo "Generating systemd service at $SYSTEMD_PATH"
cat > "$SYSTEMD_PATH" <<EOF
[Unit]
Description=Gunicorn instance for \${SERVICE_NAME}
After=network.target

[Service]
User=\${DAEMON_USER}
Group=\${DAEMON_GROUP}
WorkingDirectory=\${INSTALLATION_PATH}
ExecStart=\${INSTALLATION_PATH}/venv/bin/gunicorn \\
  --workers \${GUNICORN_WORKERS} \\
  --timeout \${GUNICORN_TIMEOUT} \\
  --bind \${GUNICORN_BIND} \\
  api:app

Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

echo "Reloading services..."
systemctl daemon-reload
systemctl enable "\$SERVICE_NAME"
systemctl restart "\$SERVICE_NAME"

echo "Deployment complete. Checking status..."
systemctl status "\$SERVICE_NAME" --no-pager
`;

  return c.text(script, 200, {
    "content-type": "text/x-shellscript; charset=utf-8",
    "cache-control": "no-store",
  });
});

app.get('/install-ui', (c) => {
  const service_name = c.req.query('service_name') || 'tunescout_ui';
  const user = c.req.query('daemon_user') || 'www-data';
  const group = c.req.query('daemon_group') || 'www-data';
  const install_path = c.req.query('installation_path') || '/var/www/tunescout_ui';
  const release_url = 'https://github.com/bboymega/TuneScout/releases/download/v1.0.0/tunescout_ui.tar.gz';
  const api_base_url = c.req.query('api_base_url') || 'http://127.0.0.1:50080';
  const bind = c.req.query('bind') || '127.0.0.1:60080';
  const [hostname, port] = bind.split(':');
  const finalHostname = hostname || '127.0.0.1';
  const finalPort = port || '60080';
  const systemd_path = c.req.query('systemd_path') + `/${service_name}.service` || `/etc/systemd/system/${service_name}.service`;
  const node_env = 'production';

  const script = `#!/usr/bin/env bash
set -e

# ===== CONFIG =====
SERVICE_NAME="${service_name}"
DAEMON_USER="${user}"
DAEMON_GROUP="${group}"
INSTALLATION_PATH="${install_path}"
RELEASE_URL="${release_url}"
API_BASE_URL="${api_base_url}"
SYSTEMD_PATH="${systemd_path}"
NODE_ENV="${node_env}"
PORT=${finalPort}
HOSTNAME="${finalHostname}"

# ===== CHECKS =====
if [[ $EUID -ne 0 ]]; then
  echo "Please run as root (sudo)"
  exit 1
fi

if ! command -v node &> /dev/null; then
  echo "Error: Node.js is not installed."
  exit 1
fi

# ===== PREPARE DIRECTORY =====
echo "Preparing installation directory at $INSTALLATION_PATH..."
if [ -d "$INSTALLATION_PATH" ]; then
    # Clean old files but keep node_modules to speed up npm install if possible
    find "$INSTALLATION_PATH" -maxdepth 1 ! -name 'node_modules' ! -name '.' -exec rm -rf {} +
else
    mkdir -p "$INSTALLATION_PATH"
fi
chown "$DAEMON_USER:$DAEMON_GROUP" "$INSTALLATION_PATH"

# ===== DOWNLOAD AND EXTRACT =====
echo "Downloading & extracting TuneScout UI release..."
curl -sSL "$RELEASE_URL" | sudo -u "$DAEMON_USER" tar -xzf - --strip-components=1 -C "$INSTALLATION_PATH"

# ===== CONFIG BACKEND API =====
CONFIG_FILE="$INSTALLATION_PATH/app/config.json"
if [ -f "$CONFIG_FILE" ]; then
    echo "Updating apiBaseUrl in config.json..."
    # Uses sed to find "apiBaseUrl": "..." and replace the value
    sudo -u "$DAEMON_USER" sed -i 's|"apiBaseUrl":\\s*".*"|"apiBaseUrl": "'"$API_BASE_URL"'"|g' "$CONFIG_FILE"
else
    echo "Warning: config.json not found at $CONFIG_FILE"
fi

# ===== INSTALL DEPENDENCIES =====
echo "Installing node dependencies..."
cd "$INSTALLATION_PATH"
sudo -u "$DAEMON_USER" npm install --production
sudo -u "$DAEMON_USER" npm run build

# ===== CREATE SYSTEMD UNIT =====
echo "Creating systemd unit at $SYSTEMD_PATH"
cat > "$SYSTEMD_PATH" <<EOF
[Unit]
Description=Node.js Application for TuneScout UI
After=network.target

[Service]
Type=simple
User=\${DAEMON_USER}
Group=\${DAEMON_GROUP}
WorkingDirectory=\${INSTALLATION_PATH}
Environment="NODE_ENV=\${NODE_ENV}"
ExecStart=/usr/bin/npm start -- --hostname $HOSTNAME --port $PORT
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# ===== APPLY CHANGES =====
echo "Reloading services..."
systemctl daemon-reload
systemctl enable "$SERVICE_NAME"
systemctl restart "$SERVICE_NAME"

echo "Deployment complete. Checking status..."
systemctl status "$SERVICE_NAME" --no-pager
`;

  return c.text(script, 200, {
    "content-type": "text/x-shellscript; charset=utf-8",
    "cache-control": "no-store",
  });
});

app.get('/', (c) => {
  return c.html(`
<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>TuneScout Deployment Worker</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body { background-color: #f8f9fa; font-family: system-ui, -apple-system, sans-serif; }
            .worker-card { max-width: 650px; margin: 50px auto; }
            .config-label { font-size: 0.8rem; font-weight: 700; color: #6c757d; text-transform: uppercase; letter-spacing: 0.5px; }
            .card-header { border-bottom: none; }
            .accordion-button:not(.collapsed) { background-color: #e7f1ff; color: #0c63e4; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="card worker-card shadow border-0">
                <div class="card-header bg-dark text-white py-3 text-center rounded-top">
                    <h1 class="h5 mb-0">TuneScout Deployment Worker</h1>
                </div>
                
                <div class="card-body p-4">
                    <div class="accordion" id="actionAccordion">
                        
                        <div class="accordion-item mb-3 border shadow-sm">
                            <h2 class="accordion-header">
                                <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#installMenuAPI">
                                    <span class="me-2">📥</span> <strong>Install TuneScout API Service</strong>
                                </button>
                            </h2>
                            <div id="installMenuAPI" class="accordion-collapse collapse" data-bs-parent="#actionAccordion">
                                <form id="install-api" class="accordion-body">
                                    <div class="row g-3">
                                        <div class="col-md-6">
                                            <label class="config-label">Service Name</label>
                                            <input type="text" name="service_name" class="form-control" placeholder="tunescout_api" value="tunescout_api">
                                        </div>
                                        <div class="col-md-6">
                                            <label class="config-label">Installation Path</label>
                                            <input type="text" name="installation_path" class="form-control" placeholder="/var/www/tunescout_api" value="/var/www/tunescout_api">
                                        </div>
                                        <div class="col-md-6">
                                            <label class="config-label">Daemon User</label>
                                            <input type="text" name="daemon_user" class="form-control" placeholder="www-data" value="www-data">
                                        </div>
                                        <div class="col-md-6">
                                            <label class="config-label">Daemon Group</label>
                                            <input type="text" name="daemon_group" class="form-control" placeholder="www-data" value="www-data">
                                        </div>
                                        <div class="col-md-4">
                                            <label class="config-label">Workers</label>
                                            <input type="number" name="gunicorn_workers" class="form-control" placeholder="10" value="10">
                                        </div>
                                        <div class="col-md-4">
                                            <label class="config-label">Timeout (s)</label>
                                            <input type="number" name="gunicorn_timeout" class="form-control" placeholder="600" value="600">
                                        </div>
                                        <div class="col-md-4">
                                            <label class="config-label">Bind Port</label>
                                            <input type="text" name="gunicorn_bind" class="form-control" placeholder="127.0.0.1:50080" value="127.0.0.1:50080">
                                        </div>
                                        <div class="col-12 text-muted small">
                                            <label class="config-label">Systemd Path</label>
                                            <input type="text" name="systemd_path" class="form-control" placeholder="/etc/systemd/system/" value="/etc/systemd/system/">
                                        </div>
                                    </div>
                                    <hr>
                                    <button type="button" onclick="copyCmd(this, 'install-api', '/install-api')" class="btn btn-success w-100 fw-bold">COPY COMMAND</button>
                                </form>
                            </div>
                        </div>
                         <div class="accordion-item mb-3 border shadow-sm">
                            <h2 class="accordion-header">
                                <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#installMenuUI">
                                    <span class="me-2">📥</span> <strong>Install TuneScout UI Service</strong>
                                </button>
                            </h2>
                            <div id="installMenuUI" class="accordion-collapse collapse" data-bs-parent="#actionAccordion">
                                <form id="install-ui" class="accordion-body">
                                    <div class="row g-3">
                                        <div class="col-md-6">
                                            <label class="config-label">Service Name</label>
                                            <input type="text" name="service_name" class="form-control" placeholder="tunescout_ui" value="tunescout_ui">
                                        </div>
                                        <div class="col-md-6">
                                            <label class="config-label">Installation Path</label>
                                            <input type="text" name="installation_path" class="form-control" placeholder="/var/www/tunescout_ui" value="/var/www/tunescout_ui">
                                        </div>
                                        <div class="col-md-6">
                                            <label class="config-label">API Base URL</label>
                                            <input type="text" name="api_base_url" class="form-control" placeholder="http://127.0.0.1:50080" value="http://127.0.0.1:50080">
                                        </div>
                                        <div class="col-md-6">
                                            <label class="config-label">Daemon User</label>
                                            <input type="text" name="user" class="form-control" placeholder="www-data" value="www-data">
                                        </div>
                                        <div class="col-md-6">
                                            <label class="config-label">Daemon Group</label>
                                            <input type="text" name="group" class="form-control" placeholder="www-data" value="www-data">
                                        </div>
                                        <div class="col-md-4">
                                            <label class="config-label">Bind Port</label>
                                            <input type="text" name="bind" class="form-control" placeholder="127.0.0.1:60080" value="127.0.0.1:60080">
                                        </div>
                                        <div class="col-12 text-muted small">
                                            <label class="config-label">Systemd Path</label>
                                            <input type="text" name="systemd_path" class="form-control" placeholder="/etc/systemd/system/" value="/etc/systemd/system/">
                                        </div>
                                    </div>
                                    <hr>
                                    <button type="button" onclick="copyCmd(this, 'install-ui', '/install-ui')" class="btn btn-success w-100 fw-bold">COPY COMMAND</button>
                                </form>
                            </div>
                        </div>

                       
                    </div>
                </div>

                <div class="card-footer bg-light text-center py-3">
                    <a href="https://github.com/bboymega/TuneScout" target="_blank" class="text-decoration-none text-secondary small">
                        View Repository on GitHub
                    </a>
                </div>
            </div>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
        <script>
            async function copyCmd(btn, formId, endpoint) {
                const form = document.getElementById(formId);
                const formData = new FormData(form);
                const params = new URLSearchParams(formData).toString();
                const scriptUrl = window.location.origin + endpoint + '?' + params;
                const command = \`curl -sSL "\${scriptUrl}" | sudo bash\`;

                try {
                    await navigator.clipboard.writeText(command);
                    
                    // UI Feedback
                    const originalText = btn.innerText;
                    const originalClass = btn.className;
                    
                    btn.innerText = "COPIED TO CLIPBOARD!";
                    btn.className = "btn btn-primary w-100 fw-bold"; // Change color to blue
                    
                    setTimeout(() => {
                        btn.innerText = originalText;
                        btn.className = originalClass;
                    }, 2000);
                } catch (err) {
                    btn.innerText = "FAILED TO COPY";
                    btn.className = "btn btn-warning w-100 fw-bold";
                }
            }
        </script>
    </body>
    </html>
  `);
});

export default app;
