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
from tkinter import Tk, Label, Button, Checkbutton, IntVar, StringVar, Frame, OptionMenu, filedialog, messagebox
from tkinterdnd2 import DND_FILES, TkinterDnD
from faster_whisper import WhisperModel
import pyperclip
from pydub import AudioSegment
import mimetypes

class CombinedTranscriptionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Antistentorian")
        self.root.geometry("600x510")
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
        self.model_size = StringVar(value="tiny")
        
        # Recording variables
        self.recording = False
        self.audio_data = []
        self.sample_rate = 16000
        self.recording_thread = None
        
        # Transcription variables
        self.transcription_thread = None
        self.stop_transcription = False
        self.model = None
        
        # Check GPU availability
        self.has_gpu = self.check_gpu()
        self.device = "cuda" if self.has_gpu else "cpu"
        
        # Main frame
        main_frame = Frame(root, bg='#121212')
        main_frame.pack(pady=10, padx=20, fill="both", expand=True)
        
        # Create UI sections
        self.create_file_section(main_frame)
        self.create_recording_section(main_frame)
        self.create_settings_section(main_frame)
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
    
    def create_file_section(self, parent):
        file_frame = Frame(parent, bg='#121212', bd=2, relief="groove")
        file_frame.pack(fill="x", pady=5)
        
        Label(file_frame, text="File/Folder Transcription", bg='#121212', fg='white', font=("Arial", 10, "bold")).pack(pady=5)
        
        Label(file_frame, text="Drag and drop audio/video files or folders here", bg='#121212', fg='white').pack(pady=5)
        
        button_frame = Frame(file_frame, bg='#121212')
        button_frame.pack(pady=5)
        
        Button(button_frame, text="Open File", command=self.open_file_dialog, bg='#3700B3', fg='white').pack(side="left", padx=5)
        Button(button_frame, text="Open Folder", command=self.open_folder_dialog, bg='#3700B3', fg='white').pack(side="left", padx=5)
    
    def create_recording_section(self, parent):
        record_frame = Frame(parent, bg='#121212', bd=2, relief="groove")
        record_frame.pack(fill="x", pady=5)
        
        Label(record_frame, text="Real-time Transcription", bg='#121212', fg='white', font=("Arial", 10, "bold")).pack(pady=5)
        
        hotkey_frame = Frame(record_frame, bg='#121212')
        hotkey_frame.pack(pady=5)
        
        Label(hotkey_frame, text="Hotkey:", bg='#121212', fg='white').pack(side="left", padx=5)
        hotkey_options = ["ctrl+shift+;", "ctrl+shift+r", "alt+r", "ctrl+space"]
        hotkey_menu = OptionMenu(hotkey_frame, self.hotkey, *hotkey_options)
        hotkey_menu.config(bg='#3700B3', fg='white')
        hotkey_menu["menu"].config(bg='#3700B3', fg='white')
        hotkey_menu.pack(side="left", padx=5)
        
        Button(hotkey_frame, text="Update Hotkey", command=self.register_hotkey, bg='#3700B3', fg='white').pack(side="left", padx=5)
        
        Button(record_frame, text="Start/Stop Recording", command=self.toggle_recording, bg='#3700B3', fg='white').pack(pady=5)
        
        self.recording_status = Label(record_frame, text="Not Recording", bg='#121212', fg='white')
        self.recording_status.pack(pady=5)
    
    def create_settings_section(self, parent):
        settings_frame = Frame(parent, bg='#121212', bd=2, relief="groove")
        settings_frame.pack(fill="x", pady=5)
        
        Label(settings_frame, text="Settings", bg='#121212', fg='white', font=("Arial", 10, "bold")).pack(pady=5)
        
        options_frame = Frame(settings_frame, bg='#121212')
        options_frame.pack(pady=5)
        
        # Left column
        left_col = Frame(options_frame, bg='#121212')
        left_col.pack(side="left", padx=10)
        
        Checkbutton(left_col, text="Include Timestamps", variable=self.include_timestamps, 
                   bg='#121212', fg='white', selectcolor='#121212').pack(anchor="w")
        
        Checkbutton(left_col, text="Open File After", variable=self.open_file_after, 
                   bg='#121212', fg='white', selectcolor='#121212').pack(anchor="w")
        
        Checkbutton(left_col, text="Copy to Clipboard", variable=self.copy_to_clipboard, 
                   bg='#121212', fg='white', selectcolor='#121212').pack(anchor="w")
        
        # Right column
        right_col = Frame(options_frame, bg='#121212')
        right_col.pack(side="left", padx=10)
        
        model_frame = Frame(right_col, bg='#121212')
        model_frame.pack(anchor="w", fill="x")
        
        Label(model_frame, text="Model:", bg='#121212', fg='white').pack(side="left")
        model_options = ["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3", "distil-large-v3"]
        model_menu = OptionMenu(model_frame, self.model_size, *model_options)
        model_menu.config(bg='#3700B3', fg='white')
        model_menu["menu"].config(bg='#3700B3', fg='white')
        model_menu.pack(side="left", padx=5)
        
        compute_frame = Frame(right_col, bg='#121212')
        compute_frame.pack(anchor="w", fill="x", pady=5)
        
        Label(compute_frame, text="Compute:", bg='#121212', fg='white').pack(side="left")
        compute_options = ["float16", "int8", "int8_float16"]
        compute_menu = OptionMenu(compute_frame, self.compute_type, *compute_options)
        compute_menu.config(bg='#3700B3', fg='white')
        compute_menu["menu"].config(bg='#3700B3', fg='white')
        compute_menu.pack(side="left", padx=5)
        
        Button(right_col, text="Apply Settings", command=self.reload_model, bg='#3700B3', fg='white').pack(pady=5)
    
    def create_control_section(self, parent):
        control_frame = Frame(parent, bg='#121212')
        control_frame.pack(fill="x", pady=5)
        
        Button(control_frame, text="Cancel Operation", command=self.cancel_operation, bg='#FF0000', fg='white').pack(side="left", padx=5)
        Button(control_frame, text="Quit", command=self.root.quit, bg='#555555', fg='white').pack(side="right", padx=5)
    
    def update_status(self, message):
        self.status_label.config(text=message)
        self.root.update_idletasks()
    
    def load_model(self):
        try:
            self.update_status(f"Loading model {self.model_size.get()} on {self.device}...")
            compute_type = self.compute_type.get()
            self.model = WhisperModel(self.model_size.get(), device=self.device, compute_type=compute_type)
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

            end_time = time.time()
            self.update_status(f"Transcription completed in {end_time - start_time:.2f} seconds")
            
            if self.copy_to_clipboard.get():
                pyperclip.copy(transcript)
                self.update_status("Transcription copied to clipboard")
            
            # Remove the default behavior of opening the file after transcription
            # if self.open_file_after.get():
            #     try:
            #         if sys.platform == 'win32':
            #             os.startfile(output_file_name)
            #         elif sys.platform == 'darwin':  # macOS
            #             subprocess.call(['open', output_file_name])
            #         else:  # Linux
            #             subprocess.call(['xdg-open', output_file_name])
            #     except Exception as e:
            #         self.update_status(f"Failed to open file: {str(e)}")
            
            # Clean up temporary file if created
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
        # Uncomment the following lines to enable a custom stop sound
        # data, samplerate = sf.read('stop_sound.wav')  # Load the stop sound
        # sd.play(data, samplerate)  # Play the sound
        # sd.wait()  # Wait until sound has finished playing
        print("Stop sound not enabled. You can uncomment the code to enable it.")

    def toggle_recording(self):
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
            
            # Save transcript to file
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_dir = os.path.join(os.path.expanduser("~"), "Whisper_Transcriptions")
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"recorded_transcript_{timestamp}.txt")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(transcript)
            
            # Copy to clipboard if enabled
            if self.copy_to_clipboard.get():
                pyperclip.copy(transcript)
            
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

if __name__ == "__main__":
    # Set numpy multithreading limit to avoid conflicts with PyTorch
    os.environ["OMP_NUM_THREADS"] = "1"
    
    try:
        root = TkinterDnD.Tk()
        app = CombinedTranscriptionApp(root)
        root.mainloop()
    except Exception as e:
        messagebox.showerror("Fatal Error", f"Application error: {str(e)}")
