import io
import os
import queue
import tempfile
import threading
import wave
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


class SpeechRecognizer(ABC):
    @abstractmethod
    def transcribe_file(self, audio_path: Path) -> str:
        ...

    @abstractmethod
    def transcribe_bytes(self, audio_bytes: bytes) -> str:
        ...

    @abstractmethod
    def is_available(self) -> bool:
        ...


class VoskRecognizer(SpeechRecognizer):
    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path
        self._model = None
        self._recognizer = None

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from vosk import Model, KaldiRecognizer
        except ImportError:
            raise RuntimeError(
                "Vosk is required for offline speech recognition. "
                "Install with: poetry add vosk"
            )
        model_dir = self.model_path or Path.home() / ".nyaya_mitra" / "vosk-model"
        if not model_dir.exists():
            raise RuntimeError(
                f"Vosk model not found at {model_dir}. "
                f"Download from https://alphacephei.com/vosk/models and extract to {model_dir}"
            )
        self._model = Model(str(model_dir))
        self._recognizer = KaldiRecognizer(self._model, 16000)

    def transcribe_file(self, audio_path: Path) -> str:
        self._load_model()
        try:
            from vosk import KaldiRecognizer
        except ImportError:
            raise RuntimeError("Vosk is not installed")
        wf = wave.open(str(audio_path), "rb")
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
            raise ValueError(
                "Audio must be mono 16-bit PCM at 16kHz. "
                "Use sox/ffmpeg to convert."
            )
        rec = KaldiRecognizer(self._model, wf.getframerate())
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            rec.AcceptWaveform(data)
        result = rec.FinalResult()
        wf.close()
        return self._parse_result(result)

    def transcribe_bytes(self, audio_bytes: bytes) -> str:
        self._load_model()
        try:
            from vosk import KaldiRecognizer
        except ImportError:
            raise RuntimeError("Vosk is not installed")
        rec = KaldiRecognizer(self._model, 16000)
        rec.AcceptWaveform(audio_bytes)
        result = rec.FinalResult()
        return self._parse_result(result)

    def is_available(self) -> bool:
        try:
            import vosk  # noqa: F401
            return True
        except ImportError:
            return False

    def _parse_result(self, result: str) -> str:
        import json
        try:
            data = json.loads(result)
            return data.get("text", "")
        except (json.JSONDecodeError, KeyError):
            return ""


class WhisperRecognizer(SpeechRecognizer):
    def __init__(self, model_size: str = "tiny"):
        self.model_size = model_size
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        try:
            import whisper
        except ImportError:
            raise RuntimeError(
                "Whisper is required. Install with: poetry add openai-whisper"
            )
        self._model = whisper.load_model(self.model_size)

    def transcribe_file(self, audio_path: Path) -> str:
        self._load_model()
        result = self._model.transcribe(str(audio_path))
        return result.get("text", "").strip()

    def transcribe_bytes(self, audio_bytes: bytes) -> str:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name
        try:
            return self.transcribe_file(Path(tmp_path))
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def is_available(self) -> bool:
        try:
            import whisper  # noqa: F401
            return True
        except ImportError:
            return False
