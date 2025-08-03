"""
Automated transcription monitor for lecture workflow
Watches input folder for new media files and auto-transcribes them
Enhanced with conversation detection and formatting
"""

import os
import json
import time
import assemblyai as aai
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import logging
from datetime import datetime
import subprocess
import shutil
import re

# Configuration
CONFIG_FILE = "transcription_config.json"
DEFAULT_CONFIG = {
    "watch_folder": "./input",
    "output_folder": "./output", 
    "processed_folder": "./processed",
    "api_key_path": "/home/drkeithcox/assemblyai.key",
    "supported_extensions": [".mp4", ".mp3", ".wav", ".m4a", ".flac", ".webm"],
    "processing_delay": 2,  # seconds to wait after file appears (for complete upload)
    "conversation_mode": False,  # Set to True to enable conversation detection
    "expected_speakers": 2,  # Number of speakers to expect in conversation mode
    "conversation_keywords": ["notebooklm", "conversation", "discussion", "dialogue", "interview"]  # Keywords in filename to auto-detect conversations
}

class TranscriptionHandler(FileSystemEventHandler):
    def __init__(self, config):
        self.config = config
        self.transcriber = None
        self.setup_logging()
        self.setup_transcriber()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('transcription_monitor.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_transcriber(self):
        """Initialize AssemblyAI transcriber"""
        try:
            with open(self.config["api_key_path"], 'r') as f:
                api_key = f.read().strip()
            aai.settings.api_key = api_key
            self.transcriber = aai.Transcriber()
            self.logger.info("✅ AssemblyAI transcriber initialized")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize transcriber: {e}")
            raise
            
    def on_created(self, event):
        """Handle new file creation events"""
        if event.is_directory:
            return
            
        file_path = Path(event.src_path)
        
        # Check if it's a supported media file
        if file_path.suffix.lower() in self.config["supported_extensions"]:
            self.logger.info(f"📁 New media file detected: {file_path.name}")
            
            # Wait for file to be completely uploaded
            time.sleep(self.config["processing_delay"])
            
            # Process the file
            self.process_file(file_path)
    
    def convert_to_mp3(self, input_file, output_file):
        """Convert audio/video file to MP3 using ffmpeg"""
        try:
            # Check if ffmpeg is available
            if not shutil.which('ffmpeg'):
                self.logger.error("❌ ffmpeg not found. Please install ffmpeg for audio conversion.")
                return False
                
            self.logger.info(f"🎵 Converting to MP3: {input_file.name}")
            
            # Use ffmpeg to convert to MP3
            cmd = [
                'ffmpeg', '-i', str(input_file),
                '-acodec', 'mp3',
                '-ab', '128k',  # 128kbps bitrate
                '-ar', '44100',  # 44.1kHz sample rate
                '-y',  # overwrite output file if exists
                str(output_file)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info(f"✅ MP3 conversion successful: {output_file.name}")
                return True
            else:
                self.logger.error(f"❌ MP3 conversion failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error during MP3 conversion: {e}")
            return False
    
    def should_convert_to_mp3(self, file_path):
        """Check if file should be converted to MP3 - convert all .wav files"""
        return file_path.suffix.lower() == '.wav'
    
    def is_conversation_file(self, file_path):
        """Determine if this file should be processed as a conversation"""
        # Check if global conversation mode is enabled
        if self.config["conversation_mode"]:
            return True
            
        # Check if filename contains conversation keywords
        filename_lower = file_path.name.lower()
        for keyword in self.config["conversation_keywords"]:
            if keyword in filename_lower:
                return True
                
        return False
    
    def format_conversation_transcript(self, transcript):
        """Format transcript with speaker diarization as a conversation"""
        if not hasattr(transcript, 'utterances') or not transcript.utterances:
            # Fallback to regular transcript if no speaker data
            return transcript.text
        
        conversation_text = []
        current_speaker = None
        
        for utterance in transcript.utterances:
            speaker = utterance.speaker
            text = utterance.text.strip()
            
            if not text:
                continue
                
            # Format timestamp
            start_seconds = utterance.start / 1000
            timestamp = f"[{int(start_seconds//60):02d}:{int(start_seconds%60):02d}]"
            
            # Add speaker line if speaker changed
            if speaker != current_speaker:
                if current_speaker is not None:
                    conversation_text.append("")  # Add blank line between speakers
                conversation_text.append(f"{speaker} {timestamp}: {text}")
                current_speaker = speaker
            else:
                # Continue same speaker
                conversation_text.append(f"    {text}")
        
        return "\n".join(conversation_text)
    
    def process_file(self, file_path):
        """Process a single media file"""
        try:
            is_conversation = self.is_conversation_file(file_path)
            mode_indicator = "💬" if is_conversation else "🎬"
            
            self.logger.info(f"{mode_indicator} Starting transcription: {file_path.name} {'(conversation mode)' if is_conversation else ''}")
            
            # Create output directories
            output_dir = Path(self.config["output_folder"])
            processed_dir = Path(self.config["processed_folder"])
            output_dir.mkdir(exist_ok=True)
            processed_dir.mkdir(exist_ok=True)
            
            # Generate output filename - use original filename with .txt extension
            suffix = "_conversation" if is_conversation else ""
            output_filename = file_path.stem + suffix + ".txt"
            txt_file = output_dir / output_filename
            
            # Check if we should convert to MP3 (all .wav files)
            mp3_file = None
            if self.should_convert_to_mp3(file_path):
                mp3_filename = file_path.stem + ".mp3"
                mp3_file = output_dir / mp3_filename
                
                # Convert to MP3
                if not self.convert_to_mp3(file_path, mp3_file):
                    self.logger.warning(f"⚠️ MP3 conversion failed, continuing with transcription")
                    mp3_file = None
            
            # Transcribe the file with appropriate settings
            self.logger.info(f"🔄 Transcribing {file_path.name}...")
            
            if is_conversation:
                # Enable speaker diarization for conversations
                self.logger.info(f"🗣️ Conversation mode: expecting {self.config['expected_speakers']} speakers")
                config = aai.TranscriptionConfig(
                    speaker_labels=True,
                    speakers_expected=self.config["expected_speakers"]
                )
                transcript = self.transcriber.transcribe(str(file_path), config=config)
            else:
                # Regular transcription
                transcript = self.transcriber.transcribe(str(file_path))
            
            if transcript.status == aai.TranscriptStatus.error:
                raise Exception(f"Transcription error: {transcript.error}")
            
            # Format the transcript based on mode
            if is_conversation:
                formatted_text = self.format_conversation_transcript(transcript)
            else:
                formatted_text = transcript.text
            
            # Write .txt file with formatted transcript
            with open(txt_file, "w") as f:
                f.write(formatted_text)
            
            # Move processed file
            processed_file = processed_dir / file_path.name
            file_path.rename(processed_file)
            
            self.logger.info(f"✅ Successfully processed {file_path.name} → {output_filename}")
            if mp3_file and mp3_file.exists():
                self.logger.info(f"🎵 MP3 file created: {mp3_file.name}")
            
        except Exception as e:
            self.logger.error(f"❌ Error processing {file_path.name}: {e}")

def load_config():
    """Load or create configuration file"""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
        # Merge with defaults for any missing keys
        for key, value in DEFAULT_CONFIG.items():
            if key not in config:
                config[key] = value
    else:
        config = DEFAULT_CONFIG.copy()
        
    # Save config (creates file if new, updates if missing keys)
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)
        
    return config

def setup_directories(config):
    """Create necessary directories"""
    for folder_key in ["watch_folder", "output_folder", "processed_folder"]:
        Path(config[folder_key]).mkdir(exist_ok=True)

def main():
    print("🎬 Auto Transcription Monitor Starting...")
    
    # Load configuration
    config = load_config()
    setup_directories(config)
    
    print(f"📁 Watching folder: {config['watch_folder']}")
    print(f"📤 Output folder: {config['output_folder']}")
    print(f"✅ Processed files moved to: {config['processed_folder']}")
    
    if config["conversation_mode"]:
        print(f"💬 Conversation mode: ENABLED (expecting {config['expected_speakers']} speakers)")
    else:
        print(f"💬 Conversation mode: AUTO-DETECT (keywords: {', '.join(config['conversation_keywords'])})")
    
    # Setup file system watcher
    event_handler = TranscriptionHandler(config)
    observer = Observer()
    observer.schedule(event_handler, config["watch_folder"], recursive=False)
    
    # Start monitoring
    observer.start()
    print(f"👀 Monitoring started. Drop media files into {config['watch_folder']} folder...")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping monitor...")
        observer.stop()
    
    observer.join()
    print("✅ Monitor stopped")

if __name__ == "__main__":
    main()