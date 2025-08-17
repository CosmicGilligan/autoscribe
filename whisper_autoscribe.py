"""
Automated transcription monitor for lecture workflow using OpenAI Whisper
Watches input folder for new media files and auto-transcribes them
Enhanced with speaker detection and formatting
"""

import os
import json
import time
import whisper
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import logging
from datetime import datetime
import subprocess
import shutil
import re
from typing import Optional

# Configuration
CONFIG_FILE = "transcription_config.json"
DEFAULT_CONFIG = {
    "watch_folder": "./input",
    "output_folder": "./output", 
    "processed_folder": "./processed",
    "supported_extensions": [".mp4", ".mp3", ".wav", ".m4a", ".flac", ".webm", ".mov", ".avi"],
    "processing_delay": 2,  # seconds to wait after file appears (for complete upload)
    "whisper_model": "medium",  # tiny, base, small, medium, large
    "language": "en",  # Language code or "auto" for auto-detection
    "task": "transcribe",  # "transcribe" or "translate" (translate converts to English)
    "conversation_mode": False,  # Set to True to enable conversation detection
    "conversation_keywords": ["notebooklm", "conversation", "discussion", "dialogue", "interview"],
    "output_formats": ["txt", "srt"],  # Available: txt, srt, vtt, json
    "verbose": True  # Show detailed processing info
}

class WhisperTranscriptionHandler(FileSystemEventHandler):
    def __init__(self, config):
        self.config = config
        self.model = None
        self.setup_logging()
        self.load_whisper_model()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('whisper_transcription_monitor.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_whisper_model(self):
        """Initialize Whisper model"""
        try:
            model_name = self.config["whisper_model"]
            self.logger.info(f"🤖 Loading Whisper model: {model_name}")
            self.model = whisper.load_model(model_name)
            self.logger.info("✅ Whisper model loaded successfully")
        except Exception as e:
            self.logger.error(f"❌ Failed to load Whisper model: {e}")
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
    
    def format_time_srt(self, seconds):
        """Convert seconds to SRT time format (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"
    
    def detect_speaker_changes(self, segments, threshold=2.0):
        """
        Simple speaker detection based on pause duration and audio characteristics
        This is a basic implementation - for better results, use pyannote.audio
        """
        speakers = []
        current_speaker = "Speaker 1"
        speaker_count = 1
        
        for i, segment in enumerate(segments):
            # Check for long pause (potential speaker change)
            if i > 0:
                pause_duration = segment['start'] - segments[i-1]['end']
                if pause_duration > threshold:
                    # Potential speaker change
                    speaker_count += 1
                    current_speaker = f"Speaker {(speaker_count - 1) % 2 + 1}"  # Alternate between 2 speakers
            
            speakers.append(current_speaker)
        
        return speakers
    
    def format_conversation_transcript(self, result):
        """Format transcript with basic speaker detection as a conversation"""
        if not result.get('segments'):
            return result['text']
        
        segments = result['segments']
        is_conversation = self.is_conversation_file(Path("dummy"))  # We'll pass this info differently
        
        if is_conversation:
            speakers = self.detect_speaker_changes(segments)
        else:
            speakers = [""] * len(segments)  # No speakers for regular transcription
        
        conversation_text = []
        current_speaker = None
        
        for i, segment in enumerate(segments):
            speaker = speakers[i] if is_conversation else ""
            text = segment['text'].strip()
            
            if not text:
                continue
                
            # Format timestamp
            start_seconds = segment['start']
            timestamp = f"[{int(start_seconds//60):02d}:{int(start_seconds%60):02d}]"
            
            # Add speaker line if speaker changed or first segment
            if speaker != current_speaker and speaker:
                if current_speaker is not None:
                    conversation_text.append("")  # Add blank line between speakers
                conversation_text.append(f"{speaker} {timestamp}: {text}")
                current_speaker = speaker
            elif speaker:
                # Continue same speaker
                conversation_text.append(f"    {text}")
            else:
                # Regular transcription without speakers
                conversation_text.append(f"{timestamp}: {text}")
        
        return "\n".join(conversation_text)
    
    def save_srt_file(self, result, output_path):
        """Save transcript as SRT subtitle file"""
        with open(output_path, "w", encoding="utf-8") as f:
            for i, segment in enumerate(result["segments"], 1):
                start_time = self.format_time_srt(segment["start"])
                end_time = self.format_time_srt(segment["end"])
                text = segment["text"].strip()
                f.write(f"{i}\n{start_time} --> {end_time}\n{text}\n\n")
    
    def save_json_file(self, result, output_path):
        """Save full Whisper result as JSON"""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    
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
            
            # Check if we should convert to MP3 (all .wav files)
            mp3_file = None
            if self.should_convert_to_mp3(file_path):
                mp3_filename = file_path.stem + ".mp3"
                mp3_file = output_dir / mp3_filename
                
                # Convert to MP3
                if not self.convert_to_mp3(file_path, mp3_file):
                    self.logger.warning(f"⚠️ MP3 conversion failed, continuing with transcription")
                    mp3_file = None
            
            # Transcribe the file with Whisper
            self.logger.info(f"🔄 Transcribing {file_path.name} with Whisper...")
            
            # Set up transcription options
            options = {
                "verbose": self.config.get("verbose", True),
                "task": self.config.get("task", "transcribe")
            }
            
            # Set language if specified (not "auto")
            if self.config.get("language") != "auto":
                options["language"] = self.config.get("language", "en")
            
            # Perform transcription
            result = self.model.transcribe(str(file_path), **options)
            
            # Generate output files based on configured formats
            base_filename = file_path.stem
            if is_conversation:
                base_filename += "_conversation"
            
            for output_format in self.config["output_formats"]:
                if output_format == "txt":
                    # Save formatted text transcript
                    txt_file = output_dir / f"{base_filename}.txt"
                    if is_conversation:
                        formatted_text = self.format_conversation_transcript(result)
                    else:
                        formatted_text = result["text"]
                    
                    with open(txt_file, "w", encoding="utf-8") as f:
                        f.write(formatted_text)
                    
                    self.logger.info(f"📝 Text transcript saved: {txt_file.name}")
                
                elif output_format == "srt":
                    # Save SRT subtitle file
                    srt_file = output_dir / f"{base_filename}.srt"
                    self.save_srt_file(result, srt_file)
                    self.logger.info(f"🎬 SRT subtitles saved: {srt_file.name}")
                
                elif output_format == "json":
                    # Save full JSON result
                    json_file = output_dir / f"{base_filename}.json"
                    self.save_json_file(result, json_file)
                    self.logger.info(f"📊 JSON data saved: {json_file.name}")
            
            # Move processed file
            processed_file = processed_dir / file_path.name
            file_path.rename(processed_file)
            
            # Log transcription stats
            duration = result.get("segments", [{}])[-1].get("end", 0) if result.get("segments") else 0
            self.logger.info(f"✅ Successfully processed {file_path.name} (duration: {duration:.1f}s)")
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
    print("🎬 Whisper Auto Transcription Monitor Starting...")
    
    # Load configuration
    config = load_config()
    setup_directories(config)
    
    print(f"📁 Watching folder: {config['watch_folder']}")
    print(f"📤 Output folder: {config['output_folder']}")
    print(f"✅ Processed files moved to: {config['processed_folder']}")
    print(f"🤖 Whisper model: {config['whisper_model']}")
    print(f"🌍 Language: {config['language']}")
    print(f"📄 Output formats: {', '.join(config['output_formats'])}")
    
    if config["conversation_mode"]:
        print(f"💬 Conversation mode: ENABLED")
    else:
        print(f"💬 Conversation mode: AUTO-DETECT (keywords: {', '.join(config['conversation_keywords'])})")
    
    # Setup file system watcher
    event_handler = WhisperTranscriptionHandler(config)
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