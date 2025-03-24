import os
import sys
import subprocess
import threading
import time
import tempfile
import keyboard
import sounddevice as sd
import soundfile as sf
import numpy as np
import torch
from tkinter import Tk, Label, Button, Checkbutton, IntVar, StringVar, Frame, OptionMenu, filedialog, messagebox, Text, Scrollbar
from tkinterdnd2 import DND_FILES, TkinterDnD
from faster_whisper import WhisperModel
import pyperclip
from pydub import AudioSegment
import mimetypes

class CombinedTranscriptionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Antistentorian")
        self.root.geometry("600x600")  # Reduced height
        self.root.configure(bg='#121212')
        
        # Initialize status_label before calling update_status
        self.status_label = Label(root, text="Ready", bg='#121212', fg='#4CAF50', anchor="w")
        self.status_label.pack(side="bottom", fill="x", padx=10, pady=5)
        
        # Variables
        self.include_timestamps = IntVar(value=0)
        self.open_file_after = IntVar(value=0)
        self.copy_to_clipboard = IntVar(value=1)
        self.hotkey = StringVar(value="ctrl+shift+;")
        self.compute_type = StringVar(value="float16")
        self.model_size = StringVar(value="distil-large-v3")
        
        # Recording variables
        self.recording = False
        self.audio_data = []
        self.sample_rate = 16000
        self.recording_thread = None
        
        # Transcription variables
        self.transcription_thread = None
        self.stop_transcription = False
        self.model = None
        self.model_loaded = False  # Add this line
        
        # Check GPU availability
        self.has_gpu = self.check_gpu()
        self.device = "cuda" if self.has_gpu else "cpu"
        
        # Main frame
        main_frame = Frame(root, bg='#121212')
        main_frame.pack(pady=5, padx=10, fill="both", expand=True)
        
        # Create top control panel
        top_panel = Frame(main_frame, bg='#121212')
        top_panel.pack(fill="x", pady=2)
        
        # Split into left and right sections
        left_panel = Frame(top_panel, bg='#121212')
        left_panel.pack(side="left", fill="x", expand=True)
        
        right_panel = Frame(top_panel, bg='#121212')
        right_panel.pack(side="right", fill="x", expand=True)
        
        # Create UI sections in new layout
        self.create_input_section(left_panel)  # Combined file and recording controls
        self.create_settings_section(right_panel)  # Compact settings
        self.create_text_output_section(main_frame)  # Keeps original size
        self.create_control_section(main_frame)
        
        # Set up drag and drop
        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind('<<Drop>>', self.drop_files)
        
        # Register hotkey
        self.register_hotkey()
        
        # Load model
        self.load_model_thread = threading.Thread(target=self.load_model)
        self.load_model_thread.daemon = True
        self.load_model_thread.start()

    def check_gpu(self):
        """Check if CUDA is available and print GPU info"""
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                self.update_status(f"Found GPU: {gpu_name}")
            return True
        else:
            self.update_status("No GPU found. Using CPU.")
            return False
    
    def create_input_section(self, parent):
        """Combined file, recording and download controls"""
        input_frame = Frame(parent, bg='#121212', bd=2, relief="groove")
        input_frame.pack(fill="x", padx=5, pady=2)
        
        # File controls
        file_frame = Frame(input_frame, bg='#121212')
        file_frame.pack(fill="x", pady=2)
        
        Label(file_frame, text="Input", bg='#121212', fg='white', 
              font=("Arial", 10, "bold")).pack(side="left", padx=5)
        
        Button(file_frame, text="Open File", command=self.open_file_dialog,
               bg='#3700B3', fg='white').pack(side="left", padx=2)
        Button(file_frame, text="Open Folder", command=self.open_folder_dialog,
               bg='#3700B3', fg='white').pack(side="left", padx=2)
        
        # Keep Download Transcript button
        Button(file_frame, text="Download Transcript", command=self.download_transcript,
               bg='#3700B3', fg='white').pack(side="left", padx=2)
        
        # Recording controls
        record_frame = Frame(input_frame, bg='#121212')
        record_frame.pack(fill="x", pady=2)
        
        Label(record_frame, text="Record", bg='#121212', fg='white').pack(side="left", padx=5)
        
        hotkey_menu = OptionMenu(record_frame, self.hotkey, 
                               "ctrl+shift+;", "ctrl+shift+r", "alt+r", "ctrl+space")
        hotkey_menu.config(bg='#3700B3', fg='white')
        hotkey_menu["menu"].config(bg='#3700B3', fg='white')
        hotkey_menu.pack(side="left", padx=2)
        
        Button(record_frame, text="Start/Stop", command=self.toggle_recording,
               bg='#3700B3', fg='white').pack(side="left", padx=2)
        
        self.recording_status = Label(record_frame, text="Ready", 
                                    bg='#121212', fg='white')
        self.recording_status.pack(side="left", padx=5)

    def create_settings_section(self, parent):
        """Compact settings panel"""
        settings_frame = Frame(parent, bg='#121212', bd=2, relief="groove")
        settings_frame.pack(fill="x", padx=5, pady=2)
        
        # Top row - Model settings
        top_row = Frame(settings_frame, bg='#121212')
        top_row.pack(fill="x", pady=2)
        
        Label(top_row, text="Model:", bg='#121212', fg='white').pack(side="left", padx=2)
        model_menu = OptionMenu(top_row, self.model_size, 
                              *["tiny", "base", "small", "medium", "large-v3", "distil-large-v3"])
        model_menu.config(bg='#3700B3', fg='white')
        model_menu["menu"].config(bg='#3700B3', fg='white')
        model_menu.pack(side="left", padx=2)
        
        Button(top_row, text="Load", command=self.reload_model,
               bg='#3700B3', fg='white').pack(side="right", padx=2)
        Button(top_row, text="Unload", command=self.unload_model,
               bg='#3700B3', fg='white').pack(side="right", padx=2)
        
        # Bottom row - Checkbuttons
        bottom_row = Frame(settings_frame, bg='#121212')
        bottom_row.pack(fill="x", pady=2)
        
        Checkbutton(bottom_row, text="Timestamps", variable=self.include_timestamps,
                   bg='#121212', fg='white', selectcolor='#121212').pack(side="left", padx=2)
        Checkbutton(bottom_row, text="Copy", variable=self.copy_to_clipboard,
                   bg='#121212', fg='white', selectcolor='#121212').pack(side="left", padx=2)

    def create_text_output_section(self, parent):
        # Create a frame for text output section
        text_frame = Frame(parent, bg='#121212', bd=2, relief="groove")
        text_frame.pack(fill="both", expand=True, pady=5)
        
        # Add a label
        Label(text_frame, text="Transcription Output", bg='#121212', fg='white', font=("Arial", 10, "bold")).pack(pady=5)
        
        # Create a frame for the text box and scrollbar
        text_box_frame = Frame(text_frame, bg='#121212')
        text_box_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Add a scrollbar
        scrollbar = Scrollbar(text_box_frame)
        scrollbar.pack(side="right", fill="y")
        
        # Add a text box with dark theme colors
        self.output_text = Text(text_box_frame, bg='#1E1E1E', fg='#FFFFFF', insertbackground='white',
                               wrap="word", yscrollcommand=scrollbar.set)
        self.output_text.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.output_text.yview)
        
        # Buttons for text operations
        button_frame = Frame(text_frame, bg='#121212')
        button_frame.pack(fill="x", pady=5)
        
        Button(button_frame, text="Copy All", command=self.copy_all_text, bg='#3700B3', fg='white').pack(side="left", padx=5)
        Button(button_frame, text="Clear", command=self.clear_text, bg='#3700B3', fg='white').pack(side="left", padx=5)
        Button(button_frame, text="Save As...", command=self.save_text_as, bg='#3700B3', fg='white').pack(side="left", padx=5)
    
    def create_control_section(self, parent):
        control_frame = Frame(parent, bg='#121212')
        control_frame.pack(fill="x", pady=5)
        
        Button(control_frame, text="Cancel Operation", command=self.cancel_operation, bg='#FF0000', fg='white').pack(side="left", padx=5)
        Button(control_frame, text="Quit", command=self.root.quit, bg='#555555', fg='white').pack(side="right", padx=5)
    
    def update_status(self, message):
        self.status_label.config(text=message)
        self.root.update_idletasks()
    
    def copy_all_text(self):
        """Copy all text from output_text to clipboard"""
        text_content = self.output_text.get("1.0", "end-1c")
        if text_content:
            pyperclip.copy(text_content)
            self.update_status("Text copied to clipboard")
        else:
            self.update_status("No text to copy")
    
    def clear_text(self):
        """Clear the text box"""
        self.output_text.delete("1.0", "end")
        self.update_status("Text cleared")
    
    def save_text_as(self):
        """Save the text content to a file"""
        text_content = self.output_text.get("1.0", "end-1c")
        if not text_content:
            self.update_status("No text to save")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
            
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(text_content)
            self.update_status(f"Text saved to {os.path.basename(file_path)}")
    
    def load_model(self):
        try:
            self.update_status(f"Loading model {self.model_size.get()} on {self.device}...")
            compute_type = self.compute_type.get()
            self.model = WhisperModel(self.model_size.get(), device=self.device, compute_type=compute_type)
            self.model_loaded = True  # Add this line
            self.update_status("Model loaded successfully.")
        except Exception as e:
            self.update_status(f"Error loading model: {str(e)}")
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
    
    def reload_model(self):
        if hasattr(self, 'load_model_thread') and self.load_model_thread.is_alive():
            self.update_status("Still loading previous model. Please wait...")
            return
        
        self.load_model_thread = threading.Thread(target=self.load_model)
        self.load_model_thread.daemon = True
        self.load_model_thread.start()
    
    def register_hotkey(self):
        # Unregister existing hotkey if any
        keyboard.unhook_all()
        
        # Register new hotkey
        current_hotkey = self.hotkey.get()
        keyboard.add_hotkey(current_hotkey, self.toggle_recording)
        self.update_status(f"Registered hotkey: {current_hotkey}")
    
    def open_file_dialog(self):
        file_path = filedialog.askopenfilename(filetypes=[
            ("Audio/Video files", "*.wav *.mp3 *.m4a *.mp4 *.avi *.mov *.ogg *.flac *.aac")
        ])
        if file_path:
            self.transcribe_file(file_path)
    
    def open_folder_dialog(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.transcribe_folder(folder_path)
    
    def drop_files(self, event):
        file_paths = self.root.tk.splitlist(event.data)
        for file_path in file_paths:
            if os.path.isdir(file_path):
                self.transcribe_folder(file_path)
            elif self.is_valid_media_file(file_path):
                self.transcribe_file(file_path)
            else:
                messagebox.showerror("Error", "Unsupported file format")
    
    def is_valid_media_file(self, file_path):
        """Check if the file is a supported audio/video format"""
        valid_extensions = {'.wav', '.mp3', '.m4a', '.mp4', '.avi', '.mov', '.ogg', '.flac', '.aac'}
        return os.path.splitext(file_path)[1].lower() in valid_extensions
    
    def convert_to_wav(self, input_file):
        """Convert any audio/video file to WAV format"""
        try:
            self.update_status(f"Converting {os.path.basename(input_file)} to WAV...")
            
            # Create temp file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_wav_path = temp_file.name
            temp_file.close()
            
            # Convert file using pydub
            audio = AudioSegment.from_file(input_file)
            audio = audio.set_channels(1)  # Convert to mono
            audio = audio.set_frame_rate(16000)  # Set sample rate to 16kHz
            audio.export(temp_wav_path, format="wav")
            
            return temp_wav_path
        except Exception as e:
            self.update_status(f"Error converting file: {str(e)}")
            raise
    
    def transcribe_file(self, file_path):
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded yet. Please wait.")
            return
            
        self.stop_transcription = False
        self.transcription_thread = threading.Thread(target=self._transcribe_file, args=(file_path,))
        self.transcription_thread.daemon = True
        self.transcription_thread.start()
        self.update_status(f"Transcribing: {os.path.basename(file_path)}")
    
    def _transcribe_file(self, file_path):
        try:
            start_time = time.time()
            
            # Convert file to WAV if it's not already
            temp_wav_path = None
            if not file_path.lower().endswith('.wav'):
                temp_wav_path = self.convert_to_wav(file_path)
                file_to_transcribe = temp_wav_path
            else:
                file_to_transcribe = file_path
            
            segments, info = self.model.transcribe(file_to_transcribe, beam_size=5)
            
            # Create output file path based on original file
            output_file_name = os.path.splitext(file_path)[0] + "_transcribed.txt"
            transcript = ""
            
            with open(output_file_name, 'w', encoding='utf-8') as f:
                include_timestamps = self.include_timestamps.get()
                if include_timestamps:
                    for segment in segments:
                        if self.stop_transcription:
                            return
                        line = f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}"
                        f.write(line + "\n")
                        transcript += line + "\n"
                else:
                    for segment in segments:
                        if self.stop_transcription:
                            return
                        f.write(f"{segment.text}\n")
                        transcript += segment.text + "\n"
            
            # Clean the output text file
            self.clean_text_file(output_file_name)
            
            # Reload the cleaned file to get the final transcript
            with open(output_file_name, 'r', encoding='utf-8') as f:
                cleaned_transcript = f.read()
            
            # Display transcript in the text box
            self.output_text.delete("1.0", "end")
            self.output_text.insert("1.0", cleaned_transcript)

            end_time = time.time()
            self.update_status(f"Transcription completed in {end_time - start_time:.2f} seconds")
            
            if self.copy_to_clipboard.get():
                pyperclip.copy(cleaned_transcript)
                self.update_status("Transcription copied to clipboard")
            
            # Remove the temporary file if created
            if temp_wav_path:
                try:
                    os.unlink(temp_wav_path)
                except:
                    pass
            
        except Exception as e:
            self.update_status(f"Error transcribing: {str(e)}")
            messagebox.showerror("Error", f"Failed to transcribe: {str(e)}")
            if temp_wav_path:
                try:
                    os.unlink(temp_wav_path)
                except:
                    pass
    
    def clean_text_file(self, file_path):
        """
        Reads a text file, removes unnecessary line breaks, trims leading/trailing spaces,
        and ensures the text ends with a period.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Replace line breaks with spaces, except for double line breaks (paragraphs).
            cleaned_content = content.replace('\n', ' ').replace('  ', ' ')
            
            # Strip leading and trailing spaces from the entire content.
            cleaned_content = cleaned_content.strip()
            
            # Ensure the text ends with a period.
            if not cleaned_content.endswith('.'):
                cleaned_content += '.'
            
            # Write the cleaned content back to the file.
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(cleaned_content)
            
            print(f"Cleaned: {file_path}")
        except Exception as e:
            print(f"Error cleaning {file_path}: {e}")
    
    def transcribe_folder(self, folder_path):
        self.stop_transcription = False
        self.transcription_thread = threading.Thread(target=self._transcribe_folder, args=(folder_path,))
        self.transcription_thread.daemon = True
        self.transcription_thread.start()
        self.update_status(f"Transcribing folder: {folder_path}")
    
    def _transcribe_folder(self, folder_path):
        media_files = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if self.is_valid_media_file(file):
                    media_files.append(os.path.join(root, file))
        
        if not media_files:
            self.update_status("No supported audio/video files found in the folder")
            return
        
        self.update_status(f"Found {len(media_files)} media files to transcribe")
        
        for i, file_path in enumerate(media_files):
            if self.stop_transcription:
                return
            self.update_status(f"Transcribing {i+1}/{len(media_files)}: {os.path.basename(file_path)}")
            self._transcribe_file(file_path)
    
    def record_audio(self):
        """Record audio from microphone and store it"""
        def callback(indata, frames, time, status):
            if status:
                print(f"Status: {status}")
            if self.recording:
                self.audio_data.append(indata.copy())
        
        try:
            with sd.InputStream(samplerate=self.sample_rate, channels=1, callback=callback):
                self.update_status("Recording... Press hotkey again to stop and transcribe")
                self.recording_status.config(text="Recording...", fg='#FF0000')
                while self.recording:
                    time.sleep(0.1)
                    # Update GUI while recording
                    self.root.update_idletasks()
        except Exception as e:
            self.update_status(f"Error recording audio: {str(e)}")
            self.recording_status.config(text="Error Recording", fg='#FF0000')
    
    def save_audio(self):
        """Save recorded audio to a temporary WAV file"""
        if not self.audio_data:
            self.update_status("No audio recorded")
            return None
            
        # Convert list of numpy arrays to one continuous array
        audio_concat = np.concatenate(self.audio_data, axis=0)
        
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_filename = temp_file.name
        
        # Save audio data to the temporary file
        sf.write(temp_filename, audio_concat, self.sample_rate)
        
        return temp_filename
    
    def play_start_sound(self):
        """Play a sound cue for starting recording."""
        data, samplerate = sf.read('start_sound.wav')  # Load the start sound
        sd.play(data, samplerate)  # Play the sound
        sd.wait()  # Wait until sound has finished playing

    def play_stop_sound(self):
        """Play a sound cue for stopping recording."""
        ################# Uncomment the following lines to enable a custom stop sound##################
        # data, samplerate = sf.read('stop_sound.wav')  # Load the stop sound
        # sd.play(data, samplerate)  # Play the sound
        # sd.wait()  # Wait until sound has finished playing
        print("Stop sound not enabled. You can uncomment the code to enable it.")

    def ensure_model_loaded(self):
        """Ensure the model is loaded before transcription"""
        if not self.model_loaded:
            self.update_status("Model not loaded. Loading model...")
            self.load_model()
            return self.model_loaded
        return True

    def toggle_recording(self):
        # Ensure model is loaded before recording
        if not self.ensure_model_loaded():
            self.update_status("Failed to load model. Cannot start recording.")
            return

        if self.model is None:
            self.update_status("Model not loaded yet. Please wait.")
            return
            
        if not self.recording:
            # Start recording
            self.recording = True
            self.audio_data = []
            self.recording_thread = threading.Thread(target=self.record_audio)
            self.recording_thread.daemon = True
            self.recording_thread.start()
            self.play_start_sound()  # Play start sound
        else:
            # Stop recording and transcribe
            self.recording = False
            self.recording_status.config(text="Processing...", fg='#FFA500')
            self.update_status("Processing audio...")
            self.play_stop_sound()  # Play stop sound (commented out)
            
            # Wait for recording thread to finish
            if self.recording_thread and self.recording_thread.is_alive():
                self.recording_thread.join(timeout=1.0)
            
            audio_file = self.save_audio()
            
            if audio_file:
                self.update_status("Transcribing recorded audio...")
                self.transcribe_recording_thread = threading.Thread(target=self._transcribe_recording, args=(audio_file,))
                self.transcribe_recording_thread.daemon = True
                self.transcribe_recording_thread.start()
    
    def _transcribe_recording(self, audio_file):
        try:
            start_time = time.time()
            segments, info = self.model.transcribe(audio_file, beam_size=5)
            
            # Generate transcript
            transcript = ""
            if self.include_timestamps.get():
                for segment in segments:
                    transcript += f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}\n"
            else:
                for segment in segments:
                    transcript += f"{segment.text}\n"
            
            end_time = time.time()
            
            # Clean up the transcript
            cleaned_transcript = transcript.replace('\n', ' ').replace('  ', ' ').strip()
            if not cleaned_transcript.endswith('.'):
                cleaned_transcript += '.'
            
            # Update text box with the transcript
            self.output_text.delete("1.0", "end")
            self.output_text.insert("1.0", cleaned_transcript)
            
            # Save transcript to file
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_dir = os.path.join(os.path.expanduser("~"), "Whisper_Transcriptions")
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"recorded_transcript_{timestamp}.txt")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(cleaned_transcript)
            
            # Copy to clipboard if enabled
            if self.copy_to_clipboard.get():
                pyperclip.copy(cleaned_transcript)
            
            # Clean up temporary file
            os.unlink(audio_file)
            
            self.update_status(f"Transcription completed in {end_time - start_time:.2f} seconds")
            self.recording_status.config(text="Not Recording", fg='white')
            
        except Exception as e:
            self.update_status(f"Error transcribing recording: {str(e)}")
            self.recording_status.config(text="Not Recording", fg='white')
    
    def cancel_operation(self):
        self.stop_transcription = True
        
        if self.recording:
            self.recording = False
            self.recording_status.config(text="Not Recording", fg='white')
        
        self.update_status("Operation cancelled")
        messagebox.showinfo("Info", "Operation cancelled")

    def unload_model(self):
        """Unload the Whisper model to free up memory"""
        if self.model is not None:
            self.model = None
            self.model_loaded = False
            torch.cuda.empty_cache()  # Clear CUDA cache if using GPU
            self.update_status("Model unloaded successfully")
        else:
            self.update_status("No model is currently loaded")

    def open_subtitle_dialog(self):
        """Open a file dialog for subtitle files and convert them"""
        file_path = filedialog.askopenfilename(filetypes=[
            ("Subtitle files", "*.vtt *.ttml")
        ])
        if file_path:
            self.convert_subtitle_file(file_path)

    def convert_subtitle_file(self, file_path):
        """Convert VTT or TTML subtitle file into a text transcript internally"""
        ext = os.path.splitext(file_path)[1].lower()
        output_file = None
        if ext == ".ttml":
            self.update_status("Converting TTML subtitle file internally...")
            output_file = self._convert_ttml_to_text(file_path)
        elif ext == ".vtt":
            self.update_status("Converting VTT subtitle file internally...")
            output_file = self._convert_vtt_to_text(file_path)
        else:
            messagebox.showerror("Error", "Unsupported subtitle format.")
            return

        if output_file and os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as f:
                content = f.read()
            self.output_text.delete("1.0", "end")
            self.output_text.insert("1.0", content)
            self.update_status("Subtitle conversion completed.")
        else:
            self.update_status("Conversion failed: output file not found.")

    def download_transcript(self):
        """Download subtitle from YouTube and convert to text with video title as name into a Transcripts folder"""
        from tkinter import simpledialog
        import glob
        youtube_url = simpledialog.askstring("Download Transcript", "Enter the YouTube video URL:")
        if not youtube_url:
            return

        self.update_status("Downloading subtitle...")
        import subprocess

        # Create 'Transcripts' directory in working directory
        transcripts_dir = os.path.join(os.getcwd(), "Transcripts")
        os.makedirs(transcripts_dir, exist_ok=True)

        output_template = os.path.join(transcripts_dir, "%(title)s.%(ext)s")

        # First try downloading TTML subtitles
        ttml_cmd = [
            "yt-dlp",
            "--skip-download",
            "--write-subs",
            "--write-auto-subs",
            "--sub-lang", "en",
            "--sub-format", "ttml",
            "--output", output_template,
            youtube_url
        ]
        try:
            subprocess.check_call(ttml_cmd)
        except Exception as e:
            self.update_status(f"Error downloading TTML subtitles: {e}")
            return

        # Search for downloaded subtitle files in the Transcripts directory
        ttml_files = glob.glob(os.path.join(transcripts_dir, "*.en.ttml"))
        vtt_files = glob.glob(os.path.join(transcripts_dir, "*.en.vtt"))
        output_file = None

        if ttml_files:
            ttml_file = ttml_files[0]
            self.update_status("TTML subtitles downloaded. Converting internally...")
            output_file = self._convert_ttml_to_text(ttml_file)
            if output_file:
                os.remove(ttml_file)  # cleanup
            else:
                return
        elif vtt_files:
            vtt_file = vtt_files[0]
            self.update_status("TTML subtitles not found. Converting VTT subtitles internally...")
            output_file = self._convert_vtt_to_text(vtt_file)
            if output_file:
                os.remove(vtt_file)  # cleanup
            else:
                return
        else:
            self.update_status("No subtitles found. Please check availability.")
            return

        # Read and display transcript
        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as f:
                content = f.read()
            self.output_text.delete("1.0", "end")
            self.output_text.insert("1.0", content)
            self.update_status("Transcript downloaded and converted successfully.")
            # Optionally copy to clipboard if enabled
            if self.copy_to_clipboard.get():
                pyperclip.copy(content)
        else:
            self.update_status("Transcript conversion failed: output file not found.")

    def _convert_ttml_to_text(self, ttml_file):
        """Converts a TTML subtitle file to a text transcript."""
        import xml.etree.ElementTree as ET
        try:
            tree = ET.parse(ttml_file)
            root = tree.getroot()
            segments = []
            include_ts = bool(self.include_timestamps.get())
            for elem in root.iter():
                if elem.tag.endswith('p'):
                    text = elem.text.strip() if elem.text else ""
                    if include_ts:
                        begin = elem.get('begin', '')
                        end = elem.get('end', '')
                        if begin and end:
                            segment = f"[{begin} -> {end}] {text}"
                        else:
                            segment = text
                    else:
                        segment = text
                    segment = segment.rstrip()
                    # Only add a period if there are multiple words and it doesn't end with punctuation.
                    if segment and segment[-1] not in ".!?" and len(segment.split()) > 1:
                        segment += "."
                    segments.append(segment)
            # New line separation when timestamps are included; otherwise use one block.
            transcript = "\n".join(segments) if include_ts else " ".join(segments)
            output_file = os.path.splitext(ttml_file)[0] + ".txt"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(transcript)
            return output_file
        except Exception as e:
            self.update_status(f"Error converting TTML subtitles: {e}")
            return None

    def _convert_vtt_to_text(self, vtt_file):
        """Converts a VTT subtitle file to a text transcript."""
        try:
            with open(vtt_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            segments = []
            include_ts = bool(self.include_timestamps.get())
            current_timestamp = ""
            for line in lines:
                line = line.strip()
                # Skip header lines
                if not line or line.upper().startswith("WEBVTT"):
                    continue
                if "-->" in line:
                    current_timestamp = line
                    continue
                if line.isdigit():
                    continue
                if include_ts and current_timestamp:
                    segment = f"[{current_timestamp}] {line}"
                    current_timestamp = ""
                else:
                    segment = line
                segment = segment.rstrip()
                # Only add a period if there are multiple words and it doesn't already end with punctuation.
                if segment and segment[-1] not in ".!?" and len(segment.split()) > 1:
                    segment += "."
                segments.append(segment)
            transcript = "\n".join(segments) if include_ts else " ".join(segments)
            output_file = os.path.splitext(vtt_file)[0] + ".txt"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(transcript)
            return output_file
        except Exception as e:
            self.update_status(f"Error converting VTT subtitles: {e}")
            return None

if __name__ == "__main__":
    # Set numpy multithreading limit to avoid conflicts with PyTorch
    os.environ["OMP_NUM_THREADS"] = "1"
    
    try:
        root = TkinterDnD.Tk()
        app = CombinedTranscriptionApp(root)
        root.mainloop()
    except Exception as e:
        messagebox.showerror("Fatal Error", f"Application error: {str(e)}")
