## TuneSafari User Interface Deployment

You can use Docker to easily deploy the TuneSafari UI Service.

Simply pull the docker image:
```
docker pull bboymega/tunesafari_ui:1.0
```
Prepare your `config.json`. Create your configuration file based on the example provided at the end of this readme.

Spin up a container with the following command:
```
docker run -d -p 3000:3000 -v /path/to/your/config.json:/app/app/config.json bboymega/tunesafari_ui:1.0
```

**FOR PRODUCTION**: Set up a reverse proxy using Apache2 or Nginx to expose the service securely through HTTPS.

## Features

- **Audio Recording**: Record audio directly from the interface and recognize music from the recording.

- **File Upload**: Upload audio or video files for recognition.

- **Media Trimming**: Select and recognize specific portions of uploaded media files.

- **Custom Branding & External URLs**: Personalize the UI with custom branding and add external URLs for additional services or resources.

## Configuration File

### app/config.json
This file serves as the configuration file for the TuneSafari UI Service.

#### Key Sections

| Key            | Description                                                                                     | Default Value                            |
| -------------- | ----------------------------------------------------------------------------------------------- | ---------------------------------------- |
| `maxDuration`  | The maximum duration (in seconds) for audio input. Audio longer than this will be truncated.    | `None` (No limit)  |
| `apiBaseUrl`   | The base URL of the backend API. It should not include any path (e.g., `https://example.com/`). | `None` |
| `appName`      | The name of the application. This is used for branding and display purposes.                    | `TuneSafari`                              |
| `appLabel`     | A short label or tagline describing the app's purpose.                                          | `Find the Tracks That Stick`             |
| `title`        | The title for the web application (used for the browser tab and other UI elements).             | `Find the Tracks That Stick`             |
| `description`  | A short description of the app, typically used for SEO and meta tags.                           | `TuneSafari - Find the Tracks That Stick` |
| `keywords`     | An array of keywords for SEO purposes.                                                          | `["TuneSafari", "Music Recognition"]`     |
| `themeColor`   | The theme color for the app's interface (used for mobile browsers, etc.).                       | `#190f13`                                |
| `externalLink` | A list of external links with titles and URLs.      | See below for examples                   |
| `linkColor`    | The color for clickable links in the app.                                                       | `#d2d9df`                                |

#### External Links

The External Links section allows you to provide additional resources or links for branding purposes. You can customize the list of external links in JSON format, as shown below:

```
{
    "title": "TuneSafari Github Repo",
    "url": "https://github.com/bboymega/TuneSafari"
},
{
    "title": "Bug Report / Feature Request",
    "url": "https://github.com/bboymega/TuneSafari/issues"
}
```

#### config.json example
```
{
    "maxDuration": 10,
    "apiBaseUrl": "https://api.example.com/",
    "appName": "TuneSafari",
    "appLabel": "Find the Tracks That Sticks",
    "title": "Find the tracks that sticks",
    "description": "TuneSafari - Find the tracks that sticks",
    "keywords": ["TuneSafari", "Music Recognition"],
    "themeColor": "#190f13",
    "externalLink": [
        {
            "title": "TuneSafari Github Repo",
            "url": "https://github.com/bboymega/TuneSafari"
        },
        {
            "title": "Bug Report / Feature Request",
            "url": "https://github.com/bboymega/TuneSafari/issues"
        }
    ],
    "linkColor": "#d2d9df"
}
```

## Advanced Customization

#### app/favicon.ico
Replace this image to customize the app’s browser tab icon.

#### public/img/background.png
Replace this image to customize the app’s background. Ensure it’s optimized for web use.

#### public/img/logo.png
Replace this image to customize the app’s logo.

#### public/img/recording.gif
This is the animation shown during audio recording. Replace it with your own GIF.

#### public/img/loading.gif
Customize the loading animation by replacing this GIF with your own.
