## About
A scalable music fingerprinting and recognition algorithm featuring multi-processing, multi-instance support and API integration.

This project is under development, core algorithm based on [worldveil/dejavu](https://github.com/worldveil/dejavu), which is licensed under the MIT License.

## Features

- **Multi-processing**: Enables faster processing times by distributing requests across multiple processing threads or cores and multiple database instances.

- **Result Storage Support**: Supports storing processed results in databases for easy retrieval, analysis, and re-use.

- **Redis Caching**: Utilizes Redis for caching frequently queried audio fingerprints to improve response time and reduce database load.

- **High Performance Database Support**: Supports **ClickHouse** for rapid queries and high-speed data handling for large-scale deployments.

- **Multi Database Instance Support**: A scalable design that enables seamless integration of multiple databases, allowing the recognition engine and result storage to independently scale and handle large volumes of data efficiently across various databases.

- **Resilient Database Instance Handling**: Utilizing the remaining available database instances in the event that some configured database instances are temporaily down or unreachable. **WARNING**: It is not recommended to keep permanently removed or offline database instances in the configuration file, as this may lead to downgraded performance due to continuous retry attempts.

- **Universal Multimedia Format Support**: Automatically converts submitted audio or video data to a standardized WAV format [44.1 kHz (configurable via `DEFAULT_FS`), 16-bit] before processing to handle a wide range of multimedia formats.

- **Media Trimming**: Supports recognizing selected portions of the submitted media file.

- **REST API**: Allows audio data recognition and fingerprinting through URL submission.

- **Rate Limiting**: Configurable settings to manage the rate of incoming requests for recognizing, fetching results and fingerprinting independently.

- **Token authentication and write protection for audio fingerprinting requests**: Ensures that only authorized users can add audio fingerprint data to the database, with built-in write protection to maintain data integrity.

## Documentation

This project is composed of three main components: **tunescout_api**, **tunescout_ui**, and **tunescout_uploader**. Below you will find a brief overview of each part, with links to their respective detailed documentation.

### [tunescout_api](./tunescout_api/README.md)

The backend API for handling music recognition, result storage and audio fingerprinting. Built with Python/Flask.

### [tunescout_ui](./tunescout_ui/README.md)

The web application serving as the user interface (UI) for the project. Built with Next.js/React.

Documentation Status: In progress.

### tunescout_uploader

The uploader module for handling the upload of audio data. Uploaded data is processed by the backend API with audio fingerprints stored in the database. This module can be run locally and is built with HTML/JavaScript.

Documentation Status: In progress.