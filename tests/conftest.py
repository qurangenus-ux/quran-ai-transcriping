"""
Test configuration and shared fixtures.

Mocks heavy ML/audio dependencies so unit tests can run without
installing the full ML stack (librosa, torch, transformers, etc.).
"""

import sys
from unittest.mock import MagicMock

# Mock heavy ML/audio libraries before any app module imports them.
_MOCK_MODULES = [
    "numpy",
    "librosa",
    "librosa.core",
    "librosa.feature",
    "torch",
    "torchaudio",
    "transformers",
    "soundfile",
    "resampy",
    "scipy",
    "scipy.signal",
    "scipy.spatial",
    "dtw",
    "pydub",
    "pydub.audio_segment",
    "pydub.silence",
    "quran_ayah_lookup",
    "rapidfuzz",
    "rapidfuzz.fuzz",
    "rapidfuzz.process",
]
for _mod in _MOCK_MODULES:
    sys.modules.setdefault(_mod, MagicMock())
