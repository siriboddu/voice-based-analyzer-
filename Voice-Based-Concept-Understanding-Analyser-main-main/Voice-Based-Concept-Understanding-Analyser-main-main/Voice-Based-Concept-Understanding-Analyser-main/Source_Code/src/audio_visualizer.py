import os
import tempfile
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment

class AudioVisualizer:
    def _load_audio_safely(self, file_path):
        """Loads any audio format safely using pydub and returns a normalized float array and sample rate."""
        # Dynamically determine the format from the file extension
        ext = os.path.splitext(file_path)[1].lower().replace('.', '')
        if ext == 'm4a':
            ext = 'mp4'  # pydub handles m4a as mp4 container structures

        # Load audio segment
        audio_segment = AudioSegment.from_file(file_path, format=ext)
        
        # Convert to mono if multi-channel
        if audio_segment.channels > 1:
            audio_segment = audio_segment.set_channels(1)

        # Extract raw data to a numpy float array
        sr = audio_segment.frame_rate
        samples = np.array(audio_segment.get_array_of_samples())
        
        # Normalize to float32 between -1.0 and 1.0 (matching librosa style expectations)
        if audio_segment.sample_width == 2:
            y = samples.astype(np.float32) / 32768.0
        elif audio_segment.sample_width == 4:
            y = samples.astype(np.float32) / 2147483648.0
        else:
            y = samples.astype(np.float32) / 128.0

        return y, sr

    def create_waveform(self, uploaded_file):
        temp_path = None
        try:
            # Dynamically extract the correct file extension (.mp3, .wav, .m4a)
            suffix = os.path.splitext(uploaded_file.name)[1].lower()
            
            # Save the uploaded file with its native matching extension
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
                temp.write(uploaded_file.getbuffer())
                temp_path = temp.name

            # Load audio data safely via our custom native decoder
            y, sr = self._load_audio_safely(temp_path)

            # Create waveform figure
            fig, ax = plt.subplots(figsize=(12, 3))
            librosa.display.waveshow(y, sr=sr, ax=ax, color="#1f77b4")
            ax.set_title("Uploaded Audio Waveform")
            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("Amplitude")
            plt.tight_layout()
            return fig

        except Exception as e:
            if 'fig' in locals():
                plt.close(fig)
            raise e

        finally:
            # Clean up the temp file after loading array parameters into RAM
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

    def save_waveform(self, audio_path):
        os.makedirs("assets", exist_ok=True)
        waveform_path = os.path.join("assets", "waveform.png")

        # Load safely using our native pydub decoder method
        y, sr = self._load_audio_safely(audio_path)
        
        fig, ax = plt.subplots(figsize=(10, 3))
        librosa.display.waveshow(y, sr=sr, ax=ax)
        ax.set_title("Audio Waveform")
        plt.tight_layout()
        
        fig.savefig(waveform_path, dpi=300)
        plt.close(fig)
        return waveform_path