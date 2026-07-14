from pathlib import Path
from typing import Optional

from app.core.voice.stt import SpeechRecognizer, VoskRecognizer, WhisperRecognizer
from app.core.voice.tts import TextToSpeech, Pyttsx3Engine


class VoiceEngine:
    def __init__(
        self,
        stt_backend: str = "vosk",
        tts_backend: str = "pyttsx3",
        vosk_model_path: Optional[Path] = None,
        whisper_model_size: str = "tiny",
    ):
        self.stt = self._create_stt(stt_backend, vosk_model_path, whisper_model_size)
        self.tts = self._create_tts(tts_backend)

    def _create_stt(
        self,
        backend: str,
        vosk_model_path: Optional[Path],
        whisper_model_size: str,
    ) -> SpeechRecognizer:
        if backend == "whisper":
            return WhisperRecognizer(model_size=whisper_model_size)
        return VoskRecognizer(model_path=vosk_model_path)

    def _create_tts(self, backend: str) -> TextToSpeech:
        if backend == "pyttsx3":
            return Pyttsx3Engine()
        return Pyttsx3Engine()

    def transcribe(self, audio_path: Path) -> str:
        return self.stt.transcribe_file(audio_path)

    def transcribe_bytes(self, audio_bytes: bytes) -> str:
        return self.stt.transcribe_bytes(audio_bytes)

    def speak(self, text: str) -> None:
        self.tts.speak(text)

    def speak_to_file(self, text: str, output_path: Path) -> Path:
        return self.tts.speak_to_file(text, output_path)

    def is_stt_available(self) -> bool:
        return self.stt.is_available()

    def is_tts_available(self) -> bool:
        return self.tts.is_available()


voice_engine = VoiceEngine()
