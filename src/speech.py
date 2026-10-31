"""Speech-to-Text (STT) and Text-to-Speech (TTS) module using Gnani Vachana SDK.

Supports both batch CLI execution (run.py) and interactive Gradio Web UI (app.py).
"""
import os
import wave
import struct

try:
    from gnani.stt import GnaniSTTClient
    from gnani.tts import GnaniTTSClient
except ImportError:
    GnaniSTTClient = None
    GnaniTTSClient = None


def _write_silent_wav(output_path: str, duration_sec: float = 1.0, sample_rate: int = 16000) -> str:
    """Create a valid silent WAV file if Gnani TTS is unavailable or unconfigured."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    num_samples = int(duration_sec * sample_rate)
    with wave.open(output_path, "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(struct.pack("<" + "h" * num_samples, *([0] * num_samples)))
    return output_path


def transcribe_audio(audio_path: str, lang: str = "en-IN") -> str:
    """Transcribe a customer audio file (.wav) using Gnani STT.

    Args:
        audio_path: Path to the recorded or uploaded .wav file.
        lang: Gnani language code (default: 'en-IN').

    Returns:
        Transcribed text string.
    """
    if not audio_path or not os.path.exists(audio_path):
        return ""

    if GnaniSTTClient and os.environ.get("GNANI_API_KEY"):
        try:
            stt = GnaniSTTClient()  # Reads GNANI_API_KEY from environment
            transcript_data = stt.transcribe(audio_path, language_code=lang)
            if isinstance(transcript_data, dict):
                return transcript_data.get("transcript", "").strip()
            return str(transcript_data).strip()
        except Exception as e:
            print(f"[Gnani STT Warning] Transcription error: {e}")
            return ""
    return "Mock STT Transcript: Where is my order? (Configure GNANI_API_KEY for live STT)"


def synthesize_speech(
    text: str,
    output_path: str = "reply.wav",
    lang: str = "en-IN",
    voice: str = "Kaveri",
) -> str:
    """Convert agent text reply into spoken audio (.wav) using Gnani TTS.

    Explicitly returns `output_path` so both `run.py` and the Gradio frontend (`app.py`)
    can locate and play the generated audio file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    if GnaniTTSClient and os.environ.get("GNANI_API_KEY") and text and text.strip():
        try:
            tts = GnaniTTSClient()  # Reads GNANI_API_KEY from environment
            tts.synthesize(
                text,
                voice=voice,
                model="timbre-v2.5",
                language=lang,
                output_file=output_path,
            )
            return output_path
        except Exception as e:
            print(f"[Gnani TTS Warning] Synthesis error: {e}. Generating fallback silent WAV.")
            return _write_silent_wav(output_path)
    else:
        return _write_silent_wav(output_path)
