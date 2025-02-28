# Antistentorian

## Overview

This application utilizes Faster-Whisper for local transcription, including live transcription and drag-and-drop support for audio files.

## Installation

1. Install PyTorch and the required dependencies from `requirements.txt` in an isolated environment.
2. Modify line 36 of transcription_app.py to change the default model used by the app. I recommend distil-large-v3.

## Usage

1. Edit batch file (e.g., "Antistentorian.bat") with the following content: your environment and its path.
2. Create a shortcut to the batch file and place it in your preferred location, such as "Start Menu" or "Desktop".

### Features

- **Dictation**: Use `Ctrl+Shift+;` to start dictation. Press `Ctrl+Shift+;` again to save the transcribed text to clipboard.
- **Drag-and-drop**: Drag audio files onto the application for processing and conversion into a saved text file in the same directory as the original file.
- **Time stamps**: Enable or disable time stamps via the provided buttons.
- **Save and open**: Use the available buttons to save the transcribed text and open the resulting `.txt` file after the Whisper model completes its transcription.
