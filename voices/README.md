# Reference voices for OmniVoice (voice cloning)

Put these files here; the TTS worker (`voice_env/tts_server.py`) clones them at startup:

| file | content |
|---|---|
| `ar_ref.wav` | 8–10 s of clean Standard Arabic speech, one speaker, no music or echo (mono, 16–48 kHz) |
| `ar_ref.txt` | the exact words spoken in `ar_ref.wav` (UTF-8) |
| `en_ref.wav` | the same for English |
| `en_ref.txt` | the exact words spoken in `en_ref.wav` |

Without them OmniVoice uses its built-in voice. Restart the worker after adding or changing a file.
