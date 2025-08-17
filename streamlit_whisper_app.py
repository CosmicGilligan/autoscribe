"""
Streamlit Web UI for Whisper Auto-Transcription Monitor
Real-time transcription with visual progress and configuration
"""

import streamlit as st
import os
import json
import time
import whisper
import threading
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import logging
import subprocess
import shutil
from datetime import datetime
import queue
import pandas as pd
from typing import Dict, List, Optional
import sys
import io
from contextlib import redirect_stdout, redirect_stderr

# Configuration
CONFIG_FILE = "streamlit_transcription_config.json"
DEFAULT_CONFIG = {
    "watch_folder": "./input",
    "output_folder": "./output", 
    "processed_folder": "./processed",
    "supported_extensions": [".mp4", ".mp3", ".wav", ".m4a", ".flac", ".webm", ".mov", ".avi"],
    "processing_delay": 2,
    "whisper_model": "medium",
    "language": "en",
    "task": "transcribe",
    "conversation_mode": False,
    "conversation_keywords": ["notebooklm", "conversation", "discussion", "dialogue", "interview"],
    "output_formats": ["txt", "srt"],
    "verbose": True
}

# Global variables for thread communication
if 'file_queue' not in st.session_state:
    st.session_state.file_queue = queue.Queue()
if 'processing_status' not in st.session_state:
    st.session_state.processing_status = {}
if 'observer' not in st.session_state:
    st.session_state.observer = None
if 'monitor_running' not in st.session_state:
    st.session_state.monitor_running = False
if 'transcription_log' not in st.session_state:
    st.session_state.transcription_log = []

class StreamlitTranscriptionHandler(FileSystemEventHandler):
    def __init__(self, config, status_queue):
        self.config = config
        self.model = None
        self.status_queue = status_queue
        self.load_whisper_model()
        
    def load_whisper_model(self):
        """Initialize Whisper model"""
        try:
            model_name = self.config["whisper_model"]
            self.status_queue.put({"type": "info", "message": f"🤖 Loading Whisper model: {model_name}..."})
            
            # Custom progress callback for model loading
            def progress_hook(current, total):
                if total > 0:
                    percent = (current / total) * 100
                    self.status_queue.put({
                        "type": "progress", 
                        "message": f"📥 Downloading model: {percent:.1f}%",
                        "progress": percent
                    })
            
            self.model = whisper.load_model(model_name, download_root=None)
            self.status_queue.put({"type": "success", "message": "✅ Whisper model loaded successfully"})
        except Exception as e:
            self.status_queue.put({"type": "error", "message": f"❌ Failed to load Whisper model: {e}"})
            raise
            
    def on_created(self, event):
        """Handle new file creation events"""
        if event.is_directory:
            return
            
        file_path = Path(event.src_path)
        
        # Check if it's a supported media file
        if file_path.suffix.lower() in self.config["supported_extensions"]:
            self.status_queue.put({
                "type": "file_detected", 
                "message": f"New media file detected: {file_path.name}",
                "file": str(file_path)
            })
            
            # Wait for file to be completely uploaded
            time.sleep(self.config["processing_delay"])
            
            # Process the file
            self.process_file(file_path)
    
    def format_time_srt(self, seconds):
        """Convert seconds to SRT time format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"
    
    def save_outputs(self, result, base_path, is_conversation):
        """Save transcription outputs in configured formats"""
        outputs = []
        
        for output_format in self.config["output_formats"]:
            if output_format == "txt":
                txt_file = base_path.with_suffix('.txt')
                if is_conversation:
                    # Simple conversation formatting
                    formatted_text = self.format_conversation_text(result)
                else:
                    formatted_text = result["text"]
                
                with open(txt_file, "w", encoding="utf-8") as f:
                    f.write(formatted_text)
                outputs.append(str(txt_file))
            
            elif output_format == "srt":
                srt_file = base_path.with_suffix('.srt')
                with open(srt_file, "w", encoding="utf-8") as f:
                    for i, segment in enumerate(result["segments"], 1):
                        start_time = self.format_time_srt(segment["start"])
                        end_time = self.format_time_srt(segment["end"])
                        text = segment["text"].strip()
                        f.write(f"{i}\n{start_time} --> {end_time}\n{text}\n\n")
                outputs.append(str(srt_file))
            
            elif output_format == "json":
                json_file = base_path.with_suffix('.json')
                with open(json_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                outputs.append(str(json_file))
        
        return outputs
    
    def format_conversation_text(self, result):
        """Basic conversation formatting"""
        if not result.get('segments'):
            return result['text']
        
        formatted_lines = []
        current_speaker = "Speaker 1"
        speaker_count = 1
        
        for i, segment in enumerate(result['segments']):
            # Simple speaker detection based on pauses
            if i > 0:
                pause = segment['start'] - result['segments'][i-1]['end']
                if pause > 2.0:  # 2 second pause suggests speaker change
                    speaker_count += 1
                    current_speaker = f"Speaker {(speaker_count - 1) % 2 + 1}"
            
            timestamp = f"[{int(segment['start']//60):02d}:{int(segment['start']%60):02d}]"
            text = segment['text'].strip()
            formatted_lines.append(f"{current_speaker} {timestamp}: {text}")
        
        return "\n".join(formatted_lines)
    
    def process_file(self, file_path):
        """Process a single media file with progress updates"""
        try:
            filename = file_path.name
            is_conversation = any(keyword in filename.lower() 
                                for keyword in self.config["conversation_keywords"])
            
            self.status_queue.put({
                "type": "processing_start",
                "message": f"🎬 Starting transcription: {filename}",
                "file": str(file_path),
                "conversation": is_conversation
            })
            
            # Create output directories
            output_dir = Path(self.config["output_folder"])
            processed_dir = Path(self.config["processed_folder"])
            output_dir.mkdir(exist_ok=True)
            processed_dir.mkdir(exist_ok=True)
            
            # Set up transcription options - disable verbose to prevent terminal output
            options = {
                "verbose": False,  # Disable terminal output
                "task": self.config.get("task", "transcribe")
            }
            
            if self.config.get("language") != "auto":
                options["language"] = self.config.get("language", "en")
            
            # Progress updates for GUI
            self.status_queue.put({
                "type": "progress",
                "message": f"🎙️ Loading audio for {filename}...",
                "progress": 10
            })
            
            # Capture any stdout/stderr during transcription
            captured_output = io.StringIO()
            captured_error = io.StringIO()
            
            self.status_queue.put({
                "type": "transcribing", 
                "message": f"🔄 Transcribing {filename}... Please wait, this may take several minutes",
                "file": str(file_path)
            })
            
            # Perform transcription with output capture
            try:
                with redirect_stdout(captured_output), redirect_stderr(captured_error):
                    result = self.model.transcribe(str(file_path), **options)
                
                # Send any captured output to GUI if needed
                output_text = captured_output.getvalue()
                if output_text.strip():
                    self.status_queue.put({
                        "type": "info",
                        "message": f"📊 Transcription details: {output_text[:100]}..." if len(output_text) > 100 else output_text.strip()
                    })
                    
            except Exception as transcription_error:
                error_text = captured_error.getvalue()
                raise Exception(f"Transcription failed: {transcription_error}. Error output: {error_text}")
            
            self.status_queue.put({
                "type": "progress", 
                "message": f"📝 Processing transcript for {filename}...",
                "progress": 85
            })
            
            self.status_queue.put({
                "type": "progress",
                "message": f"💾 Saving output files for {filename}...",
                "progress": 90
            })
            
            # Save outputs
            base_filename = file_path.stem
            if is_conversation:
                base_filename += "_conversation"
            
            base_path = output_dir / base_filename
            output_files = self.save_outputs(result, base_path, is_conversation)
            
            # Move processed file
            processed_file = processed_dir / file_path.name
            file_path.rename(processed_file)
            
            # Calculate duration and stats
            duration = result.get("segments", [{}])[-1].get("end", 0) if result.get("segments") else 0
            word_count = len(result["text"].split())
            
            # Show completion with stats
            self.status_queue.put({
                "type": "processing_complete",
                "message": f"✅ Completed {filename} → {len(output_files)} files created ({duration:.1f}s audio, {word_count:,} words)",
                "file": str(file_path),
                "duration": duration,
                "output_files": output_files,
                "word_count": word_count,
                "conversation": is_conversation,
                "progress": 100
            })
            
            # Also send individual file notifications
            for output_file in output_files:
                file_type = Path(output_file).suffix.upper()[1:]  # Remove dot, make uppercase
                self.status_queue.put({
                    "type": "info",
                    "message": f"📄 {file_type} file saved: {Path(output_file).name}"
                })
            
        except Exception as e:
            self.status_queue.put({
                "type": "error",
                "message": f"❌ Error processing {file_path.name}: {str(e)}",
                "file": str(file_path)
            })

def load_config():
    """Load or create configuration file"""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
        for key, value in DEFAULT_CONFIG.items():
            if key not in config:
                config[key] = value
    else:
        config = DEFAULT_CONFIG.copy()
    return config

def save_config(config):
    """Save configuration to file"""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)

def setup_directories(config):
    """Create necessary directories"""
    for folder_key in ["watch_folder", "output_folder", "processed_folder"]:
        Path(config[folder_key]).mkdir(exist_ok=True)

def start_monitoring(config):
    """Start the file monitoring in a separate thread"""
    if st.session_state.monitor_running:
        return
    
    setup_directories(config)
    
    event_handler = StreamlitTranscriptionHandler(config, st.session_state.file_queue)
    observer = Observer()
    observer.schedule(event_handler, config["watch_folder"], recursive=False)
    observer.start()
    
    st.session_state.observer = observer
    st.session_state.monitor_running = True

def stop_monitoring():
    """Stop the file monitoring"""
    if st.session_state.observer and st.session_state.monitor_running:
        st.session_state.observer.stop()
        st.session_state.observer.join()
        st.session_state.observer = None
        st.session_state.monitor_running = False

def main():
    st.set_page_config(
        page_title="Whisper Transcription Monitor",
        page_icon="🎬",
        layout="wide"
    )
    
    st.title("🎬 Whisper Auto-Transcription Monitor")
    st.markdown("Real-time media file transcription with OpenAI Whisper")
    
    # Load configuration
    config = load_config()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        model_options = ["tiny", "base", "small", "medium", "large"]
        config["whisper_model"] = st.selectbox(
            "Whisper Model",
            model_options,
            index=model_options.index(config["whisper_model"]),
            help="Larger models are more accurate but slower"
        )
        
        # Language selection
        language_options = {
            "Auto-detect": "auto",
            "English": "en",
            "Spanish": "es", 
            "French": "fr",
            "German": "de",
            "Italian": "it",
            "Portuguese": "pt",
            "Chinese": "zh",
            "Japanese": "ja",
            "Korean": "ko"
        }
        
        selected_lang = st.selectbox(
            "Language",
            list(language_options.keys()),
            index=list(language_options.values()).index(config["language"])
        )
        config["language"] = language_options[selected_lang]
        
        # Task selection
        config["task"] = st.selectbox(
            "Task",
            ["transcribe", "translate"],
            index=0 if config["task"] == "transcribe" else 1,
            help="Translate converts everything to English"
        )
        
        # Output formats
        format_options = ["txt", "srt", "json"]
        config["output_formats"] = st.multiselect(
            "Output Formats",
            format_options,
            default=config["output_formats"]
        )
        
        # Conversation mode
        config["conversation_mode"] = st.checkbox(
            "Always use conversation mode",
            value=config["conversation_mode"]
        )
        
        # Folder settings
        st.subheader("📁 Folders")
        config["watch_folder"] = st.text_input("Watch Folder", value=config["watch_folder"])
        config["output_folder"] = st.text_input("Output Folder", value=config["output_folder"])
        config["processed_folder"] = st.text_input("Processed Folder", value=config["processed_folder"])
        
        # Save configuration
        if st.button("💾 Save Configuration"):
            save_config(config)
            st.success("Configuration saved!")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📊 Monitor Status")
        
        # Control buttons
        col_start, col_stop = st.columns(2)
        
        with col_start:
            if st.button("▶️ Start Monitoring", disabled=st.session_state.monitor_running):
                with st.spinner("Starting monitoring system..."):
                    start_monitoring(config)
                    # Add a persistent success message
                    st.session_state.transcription_log.append({
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "type": "info",
                        "message": f"🟢 Monitoring started! Watching: {config['watch_folder']}",
                        "file": "",
                        "duration": 0,
                        "word_count": 0
                    })
        
        with col_stop:
            if st.button("⏹️ Stop Monitoring", disabled=not st.session_state.monitor_running):
                stop_monitoring()
                st.session_state.transcription_log.append({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "type": "info", 
                    "message": "🔴 Monitoring stopped",
                    "file": "",
                    "duration": 0,
                    "word_count": 0
                })
        
        # Status indicator with current status
        status_col1, status_col2 = st.columns([1, 3])
        with status_col1:
            if st.session_state.monitor_running:
                st.success("🟢 Running")
            else:
                st.error("🔴 Stopped")
        
        with status_col2:
            if st.session_state.monitor_running:
                st.info(f"👀 Watching: `{config['watch_folder']}`")
            else:
                st.info("Click 'Start Monitoring' to begin")
        
        # Current processing status
        current_progress = st.empty()
        
        # Process any new messages from the queue
        new_messages = []
        while not st.session_state.file_queue.empty():
            try:
                message = st.session_state.file_queue.get_nowait()
                new_messages.append(message)
            except queue.Empty:
                break
        
        # Handle new messages
        for message in new_messages:
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            log_entry = {
                "time": timestamp,
                "type": message["type"],
                "message": message["message"],
                "file": message.get("file", ""),
                "duration": message.get("duration", 0),
                "word_count": message.get("word_count", 0)
            }
            
            st.session_state.transcription_log.append(log_entry)
            
            # Show progress for current processing
            if message["type"] in ["processing_start", "transcribing", "progress"]:
                progress_value = message.get("progress", 0)
                with current_progress:
                    if progress_value > 0:
                        st.progress(progress_value / 100)
                    st.info(message["message"])
            elif message["type"] == "processing_complete":
                with current_progress:
                    st.success(message["message"])
                    st.balloons()  # Celebration for completed transcription!
            elif message["type"] == "error":
                with current_progress:
                    st.error(message["message"])
        
        # Keep only last 100 entries
        if len(st.session_state.transcription_log) > 100:
            st.session_state.transcription_log = st.session_state.transcription_log[-100:]
        
        # Display recent activity
        if st.session_state.transcription_log:
            st.subheader("📋 Activity Log")
            
            # Create DataFrame for better display
            df = pd.DataFrame(st.session_state.transcription_log)
            
            # Display the log with better formatting
            if not df.empty:
                # Show last 15 entries, most recent first
                display_df = df[['time', 'message']].tail(15).iloc[::-1]
                
                # Custom styling for different message types
                for _, row in display_df.iterrows():
                    time_str = row['time']
                    message = row['message']
                    
                    # Find the corresponding log entry to get the type
                    log_entry = next((log for log in reversed(st.session_state.transcription_log) 
                                    if log['time'] == time_str and log['message'] == message), None)
                    
                    if log_entry:
                        msg_type = log_entry['type']
                        if msg_type == 'error':
                            st.error(f"🕐 {time_str} - {message}")
                        elif msg_type == 'processing_complete':
                            st.success(f"🕐 {time_str} - {message}")
                        elif msg_type in ['processing_start', 'transcribing', 'progress']:
                            st.info(f"🕐 {time_str} - {message}")
                        else:
                            st.write(f"🕐 {time_str} - {message}")
        else:
            st.info("👋 No activity yet. Start monitoring and drop some media files!")
    
    with col2:
        st.header("ℹ️ Information")
        
        # Current configuration summary
        st.subheader("⚙️ Current Settings")
        st.code(f"""Model: {config['whisper_model']}
Language: {selected_lang}
Task: {config['task']}
Formats: {', '.join(config['output_formats'])}
Conversation: {'Always' if config['conversation_mode'] else 'Auto-detect'}""")
        
        # Folder status
        st.subheader("📁 Folder Status")
        
        watch_path = Path(config['watch_folder'])
        output_path = Path(config['output_folder'])
        processed_path = Path(config['processed_folder'])
        
        # Watch folder
        if watch_path.exists():
            files_in_watch = list(watch_path.glob("*"))
            media_files = [f for f in files_in_watch if f.suffix.lower() in config['supported_extensions']]
            st.success(f"📥 **Watch:** {len(media_files)} files pending")
            if media_files:
                with st.expander("View pending files"):
                    for file in media_files:
                        st.text(f"• {file.name}")
        else:
            st.error("📥 **Watch:** Folder missing")
        
        # Output folder
        if output_path.exists():
            output_files = list(output_path.glob("*"))
            st.success(f"📤 **Output:** {len(output_files)} files")
            if output_files:
                with st.expander("View output files"):
                    for file in sorted(output_files, key=lambda x: x.stat().st_mtime, reverse=True)[:10]:
                        st.text(f"• {file.name}")
        else:
            st.error("📤 **Output:** Folder missing")
        
        # Processed folder
        if processed_path.exists():
            processed_files = list(processed_path.glob("*"))
            st.success(f"✅ **Processed:** {len(processed_files)} files")
        else:
            st.error("✅ **Processed:** Folder missing")
        
        # Statistics
        completed = [log for log in st.session_state.transcription_log if log['type'] == 'processing_complete']
        if completed:
            st.subheader("📈 Session Statistics")
            total_duration = sum(log['duration'] for log in completed)
            total_words = sum(log['word_count'] for log in completed)
            
            col_stats1, col_stats2 = st.columns(2)
            with col_stats1:
                st.metric("Files Processed", len(completed))
                st.metric("Total Words", f"{total_words:,}")
            with col_stats2:
                st.metric("Total Duration", f"{total_duration:.1f}s")
                if total_duration > 0:
                    wps = total_words / total_duration if total_duration > 0 else 0
                    st.metric("Words/Second", f"{wps:.1f}")
        
        # Model info
        st.subheader("🤖 Model Information")
        model_sizes = {
            "tiny": "~39 MB, fastest",
            "base": "~74 MB, good speed", 
            "small": "~244 MB, balanced",
            "medium": "~769 MB, recommended",
            "large": "~1550 MB, best quality"
        }
        
        current_model = config['whisper_model']
        st.info(f"**{current_model.title()}**: {model_sizes.get(current_model, 'Unknown')}")
    
    # Auto-refresh every 1 second for more responsive UI
    time.sleep(1)
    st.rerun()

if __name__ == "__main__":
    main()