import base64
import os
from pathlib import Path
from typing import Optional

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from app.infra.logging import logger


class DocumentEncryptor:
    def __init__(self, key_path: Optional[Path] = None):
        self.key_path = key_path
        self._fernet: Optional[Fernet] = None

    def _derive_key(self, passphrase: str, salt: bytes) -> bytes:
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=600000,
        )
        return base64.urlsafe_b64encode(kdf.derive(passphrase.encode()))

    def initialize(self, passphrase: str) -> None:
        salt = os.urandom(16)
        key = self._derive_key(passphrase, salt)
        self._fernet = Fernet(key)
        if self.key_path:
            self.key_path.parent.mkdir(parents=True, exist_ok=True)
            self.key_path.write_bytes(salt + key)
            logger.info("Encryption key derived and saved to %s", self.key_path)

    def load_key(self, passphrase: str) -> bool:
        if not self.key_path or not self.key_path.exists():
            return False
        data = self.key_path.read_bytes()
        salt = data[:16]
        stored_key = data[16:]
        key = self._derive_key(passphrase, salt)
        if key != stored_key:
            logger.warning("Passphrase mismatch")
            return False
        self._fernet = Fernet(key)
        return True

    def encrypt(self, data: bytes) -> bytes:
        if not self._fernet:
            raise RuntimeError("Encryptor not initialized")
        return self._fernet.encrypt(data)

    def decrypt(self, data: bytes) -> bytes:
        if not self._fernet:
            raise RuntimeError("Encryptor not initialized")
        return self._fernet.decrypt(data)
