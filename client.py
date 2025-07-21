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

import asyncio
import jack
import numpy as np
import queue
import json
import websockets
import subprocess
import soxr
import time
import os
from utils import load_config, apply_fade
import threading

# Shared state and queues
audio_q = queue.Queue()
playback_q = queue.Queue()
outgoing_ws = None
MUTE_MIC = True
fade_duration = 0
mic_enabled = True

# JACK setup
client = jack.Client("AIClient")
inport_l = client.inports.register("mic_l")
inport_r = client.inports.register("mic_r")
outport_l = client.outports.register("playback_l")
outport_r = client.outports.register("playback_r")

# Buffers
audio_buffer = np.zeros(0, dtype=np.float32)
playback_chunk = b""
playback_buffer = np.zeros(0, dtype=np.float32)

def wait_for_port(port_name, timeout=3.0):
    """Wait for a JACK port to appear."""
    start = time.time()
    while time.time() - start < timeout:
        result = subprocess.run(["pw-link", "-l"], capture_output=True, text=True)
        if port_name in result.stdout:
            return True
        time.sleep(0.1)
    return False

def connect_jack_ports():
    port_map = [
        ("system:capture_1", "AIClient:mic_l"),
        ("system:capture_2", "AIClient:mic_r"),
        ("AIClient:playback_l", "system:playback_1"),
        ("AIClient:playback_r", "system:playback_2"),
    ]
    for src, dst in port_map:
        try:
            subprocess.run(["jack_connect", src, dst], check=True)
            print(f"[Jack] Connected: {src} → {dst}")
        except subprocess.CalledProcessError as e:
            print(f"[Jack] Failed to connect: {src} → {dst}: {e}")

@client.set_process_callback
def process(frames):
    global playback_buffer
    # Microphone capture (stereo or mono merge)
    in_l = inport_l.get_array()
    in_r = inport_r.get_array()
    mic_mono = ((in_l + in_r) * 0.5).copy()  # mix down to mono
    if mic_enabled:
        audio_q.put(mic_mono)

    # Debug VU meter
    # if np.abs(mic_mono).mean() > 0.01:
    #    print(f"[Mic] signal level: {np.abs(mic_mono).mean():.4f}")

    if playback_buffer.size < frames * 2:
        # Top up buffer if we don't have enough
        while playback_buffer.size < frames * 2 and not playback_q.empty():
            chunk = playback_q.get()
            if isinstance(chunk, bytes):
                new_data = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768
                playback_buffer = np.concatenate((playback_buffer, new_data))
            elif chunk == "__END__":
                try:
                    if event_loop and outgoing_ws:
                        asyncio.run_coroutine_threadsafe(outgoing_ws.send("__done__"), event_loop)
                except Exception as e:
                    print(f"[Error] Failed to notify server: {e}")
                playback_buffer = np.zeros(0, dtype=np.float32)

    # Write to output
    if playback_buffer.size >= frames * 2:
        outport_l.get_array()[:] = playback_buffer[::2][:frames]
        outport_r.get_array()[:] = playback_buffer[1::2][:frames]
        playback_buffer = playback_buffer[frames * 2:]
    else:
        outport_l.get_array().fill(0)
        outport_r.get_array().fill(0)

async def send_audio(ws, config):
    global mic_enabled
    rate = int(client.samplerate)
    config["mic_rate"] = rate
    while True:
        data = await asyncio.to_thread(audio_q.get)
        if not MUTE_MIC or mic_enabled:
            # print(f"[Send] Raw mic frame: {data.shape}, mean={data.mean():.4f}")
            resampled_np = soxr.resample(data, rate, 16000)
            clipped = np.clip(resampled_np, -1.0, 1.0)
            int16_data = (clipped * 32767).astype(np.int16)
            await ws.send(int16_data.tobytes())

async def receive_audio(ws, config):
    global fade_duration
    global mic_enabled
    buffer = bytearray()
    is_first_chunk = True
    fade_duration = config.get("fade_duration_ms", 0)

    try:
        async for message in ws:
            if isinstance(message, bytes):
                buffer += message
                if len(buffer) >= 48000:
                    chunk = bytes(buffer)
                    if is_first_chunk:
                        mic_enabled = False
                        print(f"[Mic] Mic muted: {not mic_enabled}")
                        if fade_duration > 0:
                            chunk = apply_fade(chunk, fade_duration, apply_in=True, apply_out=False)
                        is_first_chunk = False
                    playback_q.put(chunk)

                    #print(f"[Playback] Chunk bytes: {len(chunk)}")
                    
                    buffer = bytearray()
            elif isinstance(message, str) and message.strip() == "__END__":
                if buffer:
                    chunk = bytes(buffer)
                    if fade_duration > 0:
                        chunk = apply_fade(chunk, fade_duration, apply_in=False, apply_out=True)
                    playback_q.put(chunk)
                    
                    #print(f"[Playback] Chunk bytes: {len(chunk)}")

                buffer = bytearray()
                playback_q.put("__END__")

                # Delay mic unmute until buffer drains
                async def unmute_when_done():
                    while not playback_q.empty() or playback_buffer.size > 0:
                        await asyncio.sleep(0.05)
                    global mic_enabled
                    mic_enabled = True
                    print(f"[Mic] Mic muted: {not mic_enabled}")
                asyncio.create_task(unmute_when_done())
                is_first_chunk = True

                #mic_enabled = True
                #print(f"[Mic] Mic muted: {not mic_enabled}")
                #is_first_chunk = True
    finally:
        pass

async def main():
    global outgoing_ws, MUTE_MIC
    global event_loop

    loop = asyncio.get_running_loop()
    event_loop = loop

    config = load_config()
    MUTE_MIC = config.get("mute_mic_during_playback", True)

    volume = config.get("volume")
    if isinstance(volume, int) and 0 <= volume <= 100:
        try:
            subprocess.run(["amixer", "set", "Master", f"{volume}%"], check=True)
        except Exception as e:
            print(f"[Warning] Failed to set volume: {e}")

    uri = "ws://localhost:8765"
    async with websockets.connect(uri, ping_timeout=120, ping_interval=30) as ws:
        print("[Client] Connected to WebSocket server.")

        await ws.send(json.dumps({
            "type": "config_sync",
            "config": config
        }))

        outgoing_ws = ws

        # Start JACK client
        client.activate()
        print("[Client] JACK client activated")

        connect_jack_ports() 

        await asyncio.gather(
            send_audio(ws, config),
            receive_audio(ws, config)
        )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("[Client] Exiting.")
    finally:
        client.deactivate()
        client.close()
