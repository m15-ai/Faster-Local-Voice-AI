# Copyright (c) 2025 m15.ai
# 
# License: MIT
#
# Description:
#
# This is a real-time, local voice AI system optimized to run on an 8GB Ubuntu
# laptop with no GPU, achieving less than 1 second STT to TTS latency. It
# leverages a WebSocket client/server architecture, utilizes Gemma3:1b via
# Ollama for the language model, Vosk for offline speech-to-text, and Piper for
# text-to-speech. The system also employs JACK/PipeWire for low-latency I/O.
# The project aims for full localization with interruptions still on the roadmap.

import json
import os
import errno
import numpy as np

def load_config():
    CONFIG_PATH = os.path.join(os.path.dirname(__file__), ".config.json")
    DEFAULTS = {
  	"volume": 95,
  	"mic_name": "",
  	"audio_output_device": "",
  	"model_name": "gemma3:1b",
  	"voice": "alan-low.onnx",
  	"mute_mic_during_playback": True,
  	"fade_duration_ms": 50,
  	"history_length": 0,
  	"system_prompt": "You are a helpful voice assistant. Always provide short responses."
	}

    try:
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, "r") as f:
                    cfg = json.load(f)
                    print("[Config] Loaded from file:", CONFIG_PATH)
                    return {**DEFAULTS, **cfg}
            except Exception as e:
                print("[Config] Failed to load config, using defaults:", e)
        else:
            print("[Config] Config file not found, using defaults.")
    except Exception as e:
        print(f"[Config] Error loading config: {e}")

    print("[Config] Using defaults only.")
    return DEFAULTS
    
def get_voice_sample_rate(voice_name):
    json_path = os.path.join("voices", voice_name + ".json")
    try:
        with open(json_path, "r") as f:
            meta = json.load(f)
        return meta.get("audio", {}).get("sample_rate", 16000)
    except Exception:
        return 16000  # default fallback

def apply_fade(audio_bytes, fade_ms, sample_rate=48000, channels=2, apply_in=True, apply_out=True):
    if fade_ms == 0 or not (apply_in or apply_out):
        return audio_bytes

    fade_samples = int((fade_ms / 1000.0) * sample_rate)
    total_samples = len(audio_bytes) // 2  # int16 = 2 bytes

    if total_samples < 2 * fade_samples:
        return audio_bytes

    audio = np.frombuffer(audio_bytes, dtype=np.int16).copy()

    if apply_in:
        fade_in = np.linspace(0.0, 1.0, fade_samples)
        for i in range(fade_samples):
            audio[i * channels:(i + 1) * channels] = (
                audio[i * channels:(i + 1) * channels] * fade_in[i]
            ).astype(np.int16)

    if apply_out:
        fade_out = np.linspace(1.0, 0.0, fade_samples)
        for i in range(fade_samples):
            audio[-(i + 1) * channels:-(i) * channels if i > 0 else None] = (
                audio[-(i + 1) * channels:-(i) * channels if i > 0 else None] * fade_out[i]
            ).astype(np.int16)

    return audio.tobytes()
