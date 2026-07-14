import os
import re
import librosa
import numpy as np
from pydub import AudioSegment

class AudioFeatures:
    """
    Extract useful speaking metrics from an audio file.
    """

    def _load_audio_safely(self, file_path):
        """Helper to decode any audio format safely via pydub."""
        ext = os.path.splitext(file_path)[1].lower().replace('.', '')
        if ext == 'm4a':
            ext = 'mp4'
        
        audio_segment = AudioSegment.from_file(file_path, format=ext)
        if audio_segment.channels > 1:
            audio_segment = audio_segment.set_channels(1)

        sr = audio_segment.frame_rate
        samples = np.array(audio_segment.get_array_of_samples())
        
        if audio_segment.sample_width == 2:
            y = samples.astype(np.float32) / 32768.0
        elif audio_segment.sample_width == 4:
            y = samples.astype(np.float32) / 2147483648.0
        else:
            y = samples.astype(np.float32) / 128.0

        return y, sr

    def analyze(self, audio_path, transcript):
        try:
            # Read audio data directly using our pydub wrapper safely
            audio, sample_rate = self._load_audio_safely(audio_path)

            # Calculate duration safely
            duration = librosa.get_duration(y=audio, sr=sample_rate)

            # Voice Energy calculation
            rms = librosa.feature.rms(y=audio)[0]
            average_energy = float(np.mean(rms))

            # Word Count calculation
            words = transcript.split()
            total_words = len(words)

            # Words Per Minute calculation
            wpm = round((total_words / duration) * 60) if duration > 0 else 0

            # Filler Words Counting
            fillers = ["um", "uh", "like", "so", "actually", "basically", "you know"]
            transcript_lower = transcript.lower()
            filler_count = 0

            for filler in fillers:
                filler_count += len(re.findall(r"\b" + re.escape(filler) + r"\b", transcript_lower))

            # Pause Ratio calculation
            estimated_speaking_time = total_words * 0.45
            if duration > 0:
                pause_ratio = max(0, (duration - estimated_speaking_time) / duration)
            else:
                pause_ratio = 0
            pause_ratio = round(pause_ratio * 100, 2)

            return {
                "duration": round(duration, 2),
                "word_count": total_words,
                "wpm": wpm,
                "energy": round(average_energy, 4),
                "pause_ratio": pause_ratio,
                "filler_count": filler_count
            }

        except Exception as e:
            raise Exception(f"Audio Analysis Error: {str(e)}")