"""Models module for Kokoro TTS Local"""
from typing import Optional, Tuple, List
import torch
from kokoro import KPipeline
import os
import json
import codecs
from pathlib import Path
import numpy as np
import shutil
import threading

# Set environment variables for proper encoding
os.environ["PYTHONIOENCODING"] = "utf-8"
# Disable symlinks warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Setup for safer monkey-patching
import atexit
import signal
import sys

# Track whether patches have been applied
_patches_applied = {
    'json_load': False,
    'load_voice': False
}

def _cleanup_monkey_patches():
    """Restore original functions that were monkey-patched"""
    try:
        if _patches_applied['json_load'] and _original_json_load is not None:
            restore_json_load()
            _patches_applied['json_load'] = False
            print("Restored original json.load function")
    except Exception as e:
        print(f"Warning: Error restoring json.load: {e}")
        
    try:
        if _patches_applied['load_voice']:
            restore_original_load_voice()
            _patches_applied['load_voice'] = False
            print("Restored original KPipeline.load_voice function")
    except Exception as e:
        print(f"Warning: Error restoring KPipeline.load_voice: {e}")

# Register cleanup for normal exit
atexit.register(_cleanup_monkey_patches)

# Register cleanup for signals
for sig in [signal.SIGINT, signal.SIGTERM]:
    try:
        signal.signal(sig, lambda signum, frame: (
            print(f"\nReceived signal {signum}, cleaning up..."), 
            _cleanup_monkey_patches(),
            sys.exit(1)
        ))
    except (ValueError, AttributeError):
        # Some signals might not be available on all platforms
        pass

# List of available voice files
VOICE_FILES = [
    # American Female voices
    "af_alloy.pt", "af_aoede.pt", "af_bella.pt", "af_jessica.pt",
    "af_kore.pt", "af_nicole.pt", "af_nova.pt", "af_river.pt",
    "af_sarah.pt", "af_sky.pt",
    # American Male voices
    "am_adam.pt", "am_echo.pt", "am_eric.pt", "am_fenrir.pt",
    "am_liam.pt", "am_michael.pt", "am_onyx.pt", "am_puck.pt",
    "am_santa.pt",
    # British Female voices
    "bf_alice.pt", "bf_emma.pt", "bf_isabella.pt", "bf_lily.pt",
    # British Male voices
    "bm_daniel.pt", "bm_fable.pt", "bm_george.pt", "bm_lewis.pt",
    # Special voices
    "el_dora.pt", "em_alex.pt", "em_santa.pt",
    "ff_siwis.pt",
    "hf_alpha.pt", "hf_beta.pt",
    "hm_omega.pt", "hm_psi.pt",
    "jf_sara.pt", "jm_nicola.pt",
    "jf_alpha.pt", "jf_gongtsuene.pt", "jf_nezumi.pt", "jf_tebukuro.pt",
    "jm_kumo.pt",
    "pf_dora.pt", "pm_alex.pt", "pm_santa.pt",
    "zf_xiaobei.pt", "zf_xiaoni.pt", "zf_xiaoqiao.pt", "zf_xiaoyi.pt"
]

# Patch KPipeline's load_voice method to use weights_only=False
original_load_voice = KPipeline.load_voice

def patched_load_voice(self, voice_path):
    """Load voice model with weights_only=False for compatibility"""
    if not os.path.exists(voice_path):
        raise FileNotFoundError(f"Voice file not found: {voice_path}")
    voice_name = Path(voice_path).stem
    try:
        voice_model = torch.load(voice_path, weights_only=False)
        if voice_model is None:
            raise ValueError(f"Failed to load voice model from {voice_path}")
        # Ensure device is set
        if not hasattr(self, 'device'):
            self.device = 'cpu'
        # Move model to device and store in voices dictionary
        self.voices[voice_name] = voice_model.to(self.device)
        return self.voices[voice_name]
    except Exception as e:
        print(f"Error loading voice {voice_name}: {e}")
        raise

# Apply the patch
KPipeline.load_voice = patched_load_voice
_patches_applied['load_voice'] = True

# Store original function for restoration if needed
def restore_original_load_voice():
    global _patches_applied
    if _patches_applied['load_voice']:
        KPipeline.load_voice = original_load_voice
        _patches_applied['load_voice'] = False

def patch_json_load():
    """Patch json.load to handle UTF-8 encoded files with special characters"""
    global _patches_applied, _original_json_load
    original_load = json.load
    _original_json_load = original_load  # Store for restoration
    
    def custom_load(fp, *args, **kwargs):
        try:
            # Try reading with UTF-8 encoding
            if hasattr(fp, 'buffer'):
                content = fp.buffer.read().decode('utf-8')
            else:
                content = fp.read()
            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                raise
        except UnicodeDecodeError:
            # If UTF-8 fails, try with utf-8-sig for files with BOM
            fp.seek(0)
            content = fp.read()
            if isinstance(content, bytes):
                content = content.decode('utf-8-sig', errors='replace')
            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                raise
    
    json.load = custom_load
    _patches_applied['json_load'] = True
    return original_load  # Return original for restoration

# Store the original load function for potential restoration
_original_json_load = None

def restore_json_load():
    """Restore the original json.load function"""
    global _original_json_load, _patches_applied
    if _original_json_load is not None and _patches_applied['json_load']:
        json.load = _original_json_load
        _original_json_load = None
        _patches_applied['json_load'] = False

def load_config(config_path: str) -> dict:
    """Load configuration file with proper encoding handling"""
    try:
        with codecs.open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except UnicodeDecodeError:
        # Fallback to utf-8-sig if regular utf-8 fails
        with codecs.open(config_path, 'r', encoding='utf-8-sig') as f:
            return json.load(f)

# Initialize espeak-ng
phonemizer_available = False  # Global flag to track if phonemizer is working
try:
    from phonemizer.backend.espeak.wrapper import EspeakWrapper
    from phonemizer import phonemize
    import espeakng_loader
    
    # Make library available first
    library_path = espeakng_loader.get_library_path()
    data_path = espeakng_loader.get_data_path()
    espeakng_loader.make_library_available()
    
    # Set up espeak-ng paths
    EspeakWrapper.library_path = library_path
    EspeakWrapper.data_path = data_path
    
    # Verify espeak-ng is working
    try:
        test_phonemes = phonemize('test', language='en-us')
        if test_phonemes:
            phonemizer_available = True
            print("Phonemizer successfully initialized")
        else:
            print("Note: Phonemization returned empty result")
            print("TTS will work, but phoneme visualization will be disabled")
    except Exception as e:
        # Continue without espeak functionality
        print(f"Note: Phonemizer not available: {e}")
        print("TTS will work, but phoneme visualization will be disabled")

except ImportError as e:
    print(f"Note: Phonemizer packages not installed: {e}")
    print("TTS will work, but phoneme visualization will be disabled")
    # Rather than automatically installing packages, inform the user
    print("If you want phoneme visualization, manually install required packages:")
    print("pip install espeakng-loader phonemizer-fork")

# Initialize pipeline globally with thread safety
_pipeline = None
_pipeline_lock = threading.RLock()  # Reentrant lock for thread safety

def download_voice_files(voice_files=None, repo_version="main", required_count=1):
    """Download voice files from Hugging Face.
    
    Args:
        voice_files: Optional list of voice files to download. If None, download all VOICE_FILES.
        repo_version: Version/tag of the repository to use (default: "main")
        required_count: Minimum number of voices required (default: 1)
        
    Returns:
        List of successfully downloaded voice files
        
    Raises:
        ValueError: If fewer than required_count voices could be downloaded
    """
    # Use absolute path for voices directory
    voices_dir = Path(os.path.abspath("voices"))
    voices_dir.mkdir(exist_ok=True)
    
    # Import here to avoid startup dependency
    from huggingface_hub import hf_hub_download
    downloaded_voices = []
    failed_voices = []
    
    # If specific voice files are requested, use those. Otherwise use all.
    files_to_download = voice_files if voice_files is not None else VOICE_FILES
    total_files = len(files_to_download)
    
    print(f"\nDownloading voice files... ({total_files} total files)")
    
    # Check for existing voice files first
    existing_files = []
    for voice_file in files_to_download:
        voice_path = voices_dir / voice_file
        if voice_path.exists():
            print(f"Voice file {voice_file} already exists")
            downloaded_voices.append(voice_file)
            existing_files.append(voice_file)
    
    # Remove existing files from the download list
    files_to_download = [f for f in files_to_download if f not in existing_files]
    if not files_to_download and downloaded_voices:
        print(f"All required voice files already exist ({len(downloaded_voices)} files)")
        return downloaded_voices
        
    # Proceed with downloading missing files
    retry_count = 3
    try:
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            for voice_file in files_to_download:
                # Full path where the voice file should be
                voice_path = voices_dir / voice_file
                
                # Try with retries
                for attempt in range(retry_count):
                    try:
                        print(f"Downloading {voice_file}... (attempt {attempt+1}/{retry_count})")
                        # Download to a temporary location first
                        temp_path = hf_hub_download(
                            repo_id="hexgrad/Kokoro-82M",
                            filename=f"voices/{voice_file}",
                            local_dir=temp_dir,
                            force_download=True,
                            revision=repo_version
                        )
                        
                        # Move the file to the correct location
                        os.makedirs(os.path.dirname(str(voice_path)), exist_ok=True)
                        shutil.copy2(temp_path, str(voice_path))  # Use copy2 instead of move
                        
                        # Verify file integrity
                        if os.path.getsize(str(voice_path)) > 0:
                            downloaded_voices.append(voice_file)
                            print(f"Successfully downloaded {voice_file}")
                            break  # Success, exit retry loop
                        else:
                            print(f"Warning: Downloaded file {voice_file} has zero size, retrying...")
                            os.remove(str(voice_path))  # Remove invalid file
                            if attempt == retry_count - 1:
                                failed_voices.append(voice_file)
                    except (IOError, OSError, ValueError, FileNotFoundError, ConnectionError) as e:
                        print(f"Warning: Failed to download {voice_file} (attempt {attempt+1}): {e}")
                        if attempt == retry_count - 1:
                            failed_voices.append(voice_file)
                            print(f"Error: Failed all {retry_count} attempts to download {voice_file}")
    except Exception as e:
        print(f"Error during voice download process: {e}")
        import traceback
        traceback.print_exc()
    
    # Report results
    if failed_voices:
        print(f"Warning: Failed to download {len(failed_voices)} voice files: {', '.join(failed_voices)}")
    
    if not downloaded_voices:
        error_msg = "No voice files could be downloaded. Please check your internet connection."
        print(f"Error: {error_msg}")
        raise ValueError(error_msg)
    elif len(downloaded_voices) < required_count:
        error_msg = f"Only {len(downloaded_voices)} voice files could be downloaded, but {required_count} were required."
        print(f"Error: {error_msg}")
        raise ValueError(error_msg)
    else:
        print(f"Successfully processed {len(downloaded_voices)} voice files")
    
    return downloaded_voices

def build_model(model_path: str, device: str, repo_version: str = "main") -> KPipeline:
    """Build and return the Kokoro pipeline with proper encoding configuration
    
    Args:
        model_path: Path to the model file or None to use default
        device: Device to use ('cuda' or 'cpu')
        repo_version: Version/tag of the repository to use (default: "main")
        
    Returns:
        Initialized KPipeline instance
    """
    global _pipeline, _pipeline_lock
    
    # Use a lock for thread safety
    with _pipeline_lock:
        # Double-check pattern to avoid race conditions
        if _pipeline is not None:
            return _pipeline
            
        try:
            # Patch json loading before initializing pipeline
            patch_json_load()
            
            # Download model if it doesn't exist
            if model_path is None:
                model_path = 'kokoro-v1_0.pth'
            
            model_path = os.path.abspath(model_path)
            if not os.path.exists(model_path):
                print(f"Downloading model file {model_path}...")
                try:
                    from huggingface_hub import hf_hub_download
                    model_path = hf_hub_download(
                        repo_id="hexgrad/Kokoro-82M",
                        filename="kokoro-v1_0.pth",
                        local_dir=".",
                        force_download=True,
                        revision=repo_version
                    )
                    print(f"Model downloaded to {model_path}")
                except Exception as e:
                    print(f"Error downloading model: {e}")
                    raise ValueError(f"Could not download model: {e}") from e
            
            # Download config if it doesn't exist
            config_path = os.path.abspath("config.json")
            if not os.path.exists(config_path):
                print("Downloading config file...")
                try:
                    config_path = hf_hub_download(
                        repo_id="hexgrad/Kokoro-82M",
                        filename="config.json",
                        local_dir=".",
                        force_download=True,
                        revision=repo_version
                    )
                    print(f"Config downloaded to {config_path}")
                except Exception as e:
                    print(f"Error downloading config: {e}")
                    raise ValueError(f"Could not download config: {e}") from e
            
            # Download voice files - require at least one voice
            try:
                downloaded_voices = download_voice_files(repo_version=repo_version, required_count=1)
            except ValueError as e:
                print(f"Error: Voice files download failed: {e}")
                raise ValueError("Voice files download failed") from e
            
            # Validate language code
            lang_code = 'a'  # 'a' for American English
            if lang_code not in ['a', 'b']:  # Simple validation of supported codes
                print(f"Warning: Unsupported language code '{lang_code}'. Using 'a' (American English).")
                lang_code = 'a'
                
            # Initialize pipeline with validated language code
            pipeline_instance = KPipeline(lang_code=lang_code)
            if pipeline_instance is None:
                raise ValueError("Failed to initialize KPipeline - pipeline is None")
                
            # Store device parameter for reference in other operations
            pipeline_instance.device = device
            
            # Initialize voices dictionary if it doesn't exist
            if not hasattr(pipeline_instance, 'voices'):
                pipeline_instance.voices = {}
            
            # Try to load the first available voice with improved error handling
            voice_loaded = False
            for voice_file in downloaded_voices:
                voice_path = os.path.abspath(os.path.join("voices", voice_file))
                if os.path.exists(voice_path):
                    try:
                        pipeline_instance.load_voice(voice_path)
                        print(f"Successfully loaded voice: {voice_file}")
                        voice_loaded = True
                        break  # Successfully loaded a voice
                    except Exception as e:
                        print(f"Warning: Failed to load voice {voice_file}: {e}")
                        continue
            
            if not voice_loaded:
                print("Warning: Could not load any voice models")
            
            # Set the global _pipeline only after successful initialization
            _pipeline = pipeline_instance
            
        except Exception as e:
            print(f"Error initializing pipeline: {e}")
            # Restore original json.load on error
            restore_json_load()
            raise
            
        return _pipeline

def list_available_voices() -> List[str]:
    """List all available voice models"""
    # Always use absolute path for consistency
    voices_dir = Path(os.path.abspath("voices"))
    
    # Create voices directory if it doesn't exist
    if not voices_dir.exists():
        print(f"Creating voices directory at {voices_dir}")
        voices_dir.mkdir(exist_ok=True)
        return []
    
    # Get all .pt files in the voices directory
    voice_files = list(voices_dir.glob("*.pt"))
    
    # If we found voice files, return them
    if voice_files:
        return [f.stem for f in voice_files]
        
    # If no voice files in standard location, check if we need to do a one-time migration
    # This is legacy support for older installations
    alt_voices_path = Path(".") / "voices"
    if alt_voices_path.exists() and alt_voices_path.is_dir() and alt_voices_path != voices_dir:
        print(f"Checking alternative voice location: {alt_voices_path.absolute()}")
        alt_voice_files = list(alt_voices_path.glob("*.pt"))
        
        if alt_voice_files:
            print(f"Found {len(alt_voice_files)} voice files in alternate location")
            print("Moving files to the standard voices directory...")
            
            # Process files in a batch for efficiency
            files_moved = 0
            for voice_file in alt_voice_files:
                target_path = voices_dir / voice_file.name
                if not target_path.exists():
                    try:
                        # Use copy2 to preserve metadata, then remove original if successful
                        shutil.copy2(str(voice_file), str(target_path))
                        files_moved += 1
                    except (OSError, IOError) as e:
                        print(f"Error copying {voice_file.name}: {e}")
            
            if files_moved > 0:
                print(f"Successfully moved {files_moved} voice files")
                return [f.stem for f in voices_dir.glob("*.pt")]
    
    print("No voice files found. Please run the application again to download voices.")
    return []

def load_voice(voice_name: str, device: str) -> torch.Tensor:
    """Load a voice model in a thread-safe manner
    
    Args:
        voice_name: Name of the voice to load (with or without .pt extension)
        device: Device to use ('cuda' or 'cpu')
        
    Returns:
        Loaded voice model tensor
        
    Raises:
        ValueError: If voice file not found or loading fails
    """
    pipeline = build_model(None, device)
    
    # Format voice path correctly - strip .pt if it was included
    voice_name = voice_name.replace('.pt', '')
    voice_path = os.path.abspath(os.path.join("voices", f"{voice_name}.pt"))
    
    if not os.path.exists(voice_path):
        raise ValueError(f"Voice file not found: {voice_path}")
        
    # Use a lock to ensure thread safety when loading voices
    with _pipeline_lock:
        # Check if voice is already loaded
        if hasattr(pipeline, 'voices') and voice_name in pipeline.voices:
            return pipeline.voices[voice_name]
            
        # Load voice if not already loaded
        return pipeline.load_voice(voice_path)

def generate_speech(
    model: KPipeline,
    text: str,
    voice: str,
    lang: str = 'a',
    device: str = 'cpu',
    speed: float = 1.0
) -> Tuple[Optional[torch.Tensor], Optional[str]]:
    """Generate speech using the Kokoro pipeline in a thread-safe manner
    
    Args:
        model: KPipeline instance
        text: Text to synthesize
        voice: Voice name (e.g. 'af_bella')
        lang: Language code ('a' for American English, 'b' for British English)
        device: Device to use ('cuda' or 'cpu')
        speed: Speech speed multiplier (default: 1.0)
        
    Returns:
        Tuple of (audio tensor, phonemes string) or (None, None) on error
    """
    global _pipeline_lock
    
    try:
        if model is None:
            raise ValueError("Model is None - pipeline not properly initialized")
        
        # Format voice name and path
        voice_name = voice.replace('.pt', '')
        voice_path = os.path.abspath(os.path.join("voices", f"{voice_name}.pt"))
        
        # Check if voice file exists
        if not os.path.exists(voice_path):
            raise ValueError(f"Voice file not found: {voice_path}")
        
        # Thread-safe initialization of model properties and voice loading
        with _pipeline_lock:
            # Initialize voices dictionary if it doesn't exist
            if not hasattr(model, 'voices'):
                model.voices = {}
                
            # Ensure device is set
            if not hasattr(model, 'device'):
                model.device = device
                
            # Ensure voice is loaded before generating
            if voice_name not in model.voices:
                print(f"Loading voice {voice_name}...")
                try:
                    model.load_voice(voice_path)
                    if voice_name not in model.voices:
                        raise ValueError("Voice load succeeded but voice not in model.voices dictionary")
                except Exception as e:
                    raise ValueError(f"Failed to load voice {voice_name}: {e}")
        
        # Generate speech (outside the lock for better concurrency)
        print(f"Generating speech with device: {model.device}")
        generator = model(
            text, 
            voice=voice_path,
            speed=speed,
            split_pattern=r'\n+'
        )
        
        # Get first generated segment and convert numpy array to tensor if needed
        for gs, ps, audio in generator:
            if audio is not None:
                if isinstance(audio, np.ndarray):
                    audio = torch.from_numpy(audio).float()
                return audio, ps
            
        return None, None
    except (ValueError, FileNotFoundError, RuntimeError, KeyError, AttributeError, TypeError) as e:
        print(f"Error generating speech: {e}")
        return None, None
    except Exception as e:
        print(f"Unexpected error during speech generation: {e}")
        import traceback
        traceback.print_exc()
        return None, None