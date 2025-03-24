Antistentorian is a all in one transcription app utilizing faster-whisper to transcribe; your default recording device, a drag-and-dropped audio or video file, and now transcripts from youtube urls. Saved transcriptions are in the same directory as the files or they'll go into the Transcripts folder.

The generated text is now displayed inside the UI as well as being able to be saved to your clipboard.

Change line 36 to the default model you want to use I recommend "distil-large-v3" as it uses a little over a gigabyte of vram while tiny model will use 0.1 gb.

How to use
Git clone the repository, create conda environment, activate, and install requirements. 
Be sure to edit your batch file to activate your conda environment and change directories to your folder.
Run the file and after automatically downloading the model to your .cache folder you can use the ctrl+shift+; to begin and end recording.

In the app you can drag any file to see and generate the text transcript. 

