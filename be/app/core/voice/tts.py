import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


class TextToSpeech(ABC):
    @abstractmethod
    def speak(self, text: str) -> None:
        ...

    @abstractmethod
    def speak_to_file(self, text: str, output_path: Path) -> Path:
        ...

    @abstractmethod
    def is_available(self) -> bool:
        ...

    @abstractmethod
    def set_voice(self, voice_id: str) -> None:
        ...

    @abstractmethod
    def set_rate(self, rate: int) -> None:
        ...

    @abstractmethod
    def set_volume(self, volume: float) -> None:
        ...


class Pyttsx3Engine(TextToSpeech):
    def __init__(self):
        self._engine = None
        self._voice_id: Optional[str] = None
        self._rate: int = 150
        self._volume: float = 1.0

    def _get_engine(self):
        if self._engine is None:
            try:
                import pyttsx3
            except ImportError:
                raise RuntimeError(
                    "pyttsx3 is required for text-to-speech. "
                    "Install with: poetry add pyttsx3"
                )
            self._engine = pyttsx3.init()
            if self._voice_id:
                self._engine.setProperty("voice", self._voice_id)
            self._engine.setProperty("rate", self._rate)
            self._engine.setProperty("volume", self._volume)
        return self._engine

    def speak(self, text: str) -> None:
        engine = self._get_engine()
        engine.say(text)
        engine.runAndWait()

    def speak_to_file(self, text: str, output_path: Path) -> Path:
        engine = self._get_engine()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        engine.save_to_file(text, str(output_path))
        engine.runAndWait()
        return output_path.resolve()

    def is_available(self) -> bool:
        try:
            import pyttsx3  # noqa: F401
            return True
        except ImportError:
            return False

    def set_voice(self, voice_id: str) -> None:
        self._voice_id = voice_id
        if self._engine is not None:
            self._engine.setProperty("voice", voice_id)

    def set_rate(self, rate: int) -> None:
        self._rate = rate
        if self._engine is not None:
            self._engine.setProperty("rate", rate)

    def set_volume(self, volume: float) -> None:
        self._volume = max(0.0, min(1.0, volume))
        if self._engine is not None:
            self._engine.setProperty("volume", self._volume)

    def list_voices(self) -> list[dict]:
        engine = self._get_engine()
        voices = engine.getProperty("voices")
        return [
            {"id": v.id, "name": v.name, "languages": v.languages}
            for v in voices
        ]
