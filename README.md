## Antistentorian

**Description**

Antistentorian is an all-in-one transcription application that leverages faster-whisper for efficient transcription, supporting default recording devices, audio or video files dropped onto the interface, and transcripts from YouTube URLs. Saved transcriptions are stored in the same directory as the source files, while youtube transcripts are directed to the Transcripts folder.

The generated text is displayed within the user interface, providing an option for users to edit it directly. Ollama can be connected and provide quick fixes to your text. There is also TTS functionality using Kokoro and its fast speech generation.

**How to Use**

1. Clone the Git repository.
2. Create a Conda environment and activate it.
3. Install the required dependencies.
4. Edit your batch file to activate the Conda environment and navigate to your desired working directory.
5. Run the script, which will automatically download the necessary model to the .cache folder.
6. Use the shortcut Ctrl+Shift+; to initiate and conclude recording.

In the application, you can drag any audio or video file onto it for immediate text transcription generation.

**Model Modification**

To modify the default model used by Antistentorian, change line 36 to specify your preferred model. We recommend "distil-large-v3" due to its moderate memory usage of over one gigabyte while maintaining performance, in contrast to the tiny model's significantly lower resource consumption of 0.1 GB.

**Credits**

- Kokoro
- Faster-Whisper
- Yt-dlp
- Ollama
