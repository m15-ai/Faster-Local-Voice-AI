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
import websockets
import json
import re
import os
import time
from vosk import Model, KaldiRecognizer
from utils import load_config
from utils import get_voice_sample_rate
import numpy as np

RATE = 16000
CHANNELS = 1
MODEL_PATH = "vosk-model"
PIPER_PATH = "/home/m15/bin/piper/piper"

LOW_EFFORT_UTTERANCES = {"huh", "uh", "um", "erm", "hmm", "he's", "but", "the"}

vosk_model = Model(MODEL_PATH)

# Configuration constants for TTS streaming
FLUSH_INTERVAL = 0.2  # Time (seconds) to wait before flushing response text to TTS
MAX_RESPONSE_LENGTH = 75  # Maximum characters in response text before flushing to TTS
MIN_BATCH_SENTENCES = 1  # Minimum number of sentences to batch before processing
BATCH_FLUSH_INTERVAL = 0.75  # Time (seconds) to wait before flushing batched sentences

def clean_response(text):
    # Normalize apostrophes and quotes
    text = text.replace('’', "'").replace('`', "'").replace("''", "'")
    # Remove emojis
    text = re.sub(r'[\U0001F000-\U0001FFFF\U00002700-\U000027BF\U00002600-\U000026FF]+', '', text)
    text = re.sub(r"[\*]+", '', text)
    text = re.sub(r"\(.*?\)", '', text)
    text = re.sub(r"<.*?>", '', text)
    text = text.replace('\n', ' ').strip()
    # Normalize multiple spaces, preserve single spaces
    text = re.sub(r'\s{2,}', ' ', text)
    # Ensure no extra spaces before punctuation
    text = re.sub(r'\s+([.?!])', r'\1', text)
    # Fix contractions (e.g., "don ' t" → "don't")
    text = re.sub(r"(\w+)\s*'\s*(\w+)", r"\1'\2", text)
    # Insert space after sentence-ending punctuation if missing
    text = re.sub(r'([.!?])([^\s.!?])', r'\1 \2', text)
    # Handle colons: ensure space before and after, unless part of a time (e.g., 12:30)
    text = re.sub(r'(?<!\d):(?!\d)', r' : ', text)  # Add spaces around colons not in numbers
    # Remove any resulting double spaces from colon spacing
    text = re.sub(r'\s{2,}', ' ', text)
    return text

def split_sentences(text):
    # Custom sentence splitter: split on .!? followed by space or end, excluding abbreviations
    sentence_endings = r'(?<=[.!?])(?:\s+|$)(?!(?:Mr|Mrs|Ms|Dr|Sr|Jr|Prof|St|Ave|Inc|Corp|[0-9])\.)'
    sentences = [s.strip() for s in re.split(sentence_endings, text) if s.strip()]
    # Ensure space after sentence-ending punctuation
    result = []
    for i, sentence in enumerate(sentences):
        if i < len(sentences) - 1 and not sentence.endswith((' ', '\n')):
            sentence += ' '
        result.append(sentence)
    return result

async def monitor_piper_stderr(stderr_pipe):
    while True:
        line = await stderr_pipe.readline()
        if not line:
            break
        #print(f"\033[38;5;196m[Piper STDERR] {line.decode().strip()}\033[0m")

async def query_ollama(model, messages):
    payload = {
        "model": model,
        "messages": messages,
        "stream": False
    }
    proc = await asyncio.create_subprocess_exec(
        'curl', '-s', '-X', 'POST', 'http://localhost:11434/api/chat',
        '-H', 'Content-Type: application/json',
        '-d', json.dumps(payload),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await proc.communicate()
    if stderr:
        print(f"\033[38;5;196m[Ollama Query STDERR] {stderr.decode().strip()}\033[0m")
    result = json.loads(stdout)
    response = result['message']['content']
    print(f"\033[38;5;200m[Debug] Raw Ollama response: '{response}'\033[0m")
    return clean_response(response)

async def stream_ollama_response(model, messages):
    payload = {
        "model": model,
        "messages": messages,
        "options": {"num_ctx": 2048, "temperature": 0.7, "top_k": 40},  # Limit context window, reduce randomness for faster output
        "stream": True
    }
    proc = await asyncio.create_subprocess_exec(
        'curl', '-N', '-s', '-X', 'POST', 'http://localhost:11434/api/chat',
        '-H', 'Content-Type: application/json',
        '-d', json.dumps(payload),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    async for line in proc.stdout:
        chunk = line.decode().strip()
        if not chunk:
            continue
        try:
            json_chunk = json.loads(chunk)
            token = json_chunk.get("message", {}).get("content", "")
            #print(f"\033[38;5;200m[Debug] Raw token: '{token}'\033[0m")
            await asyncio.sleep(0.01)  # Yield to event loop to prevent starvation of other coroutines (e.g., TTS processing)
            token = re.sub(r'[\U0001F000-\U0001FFFF\U00002700-\U000027BF\U00002600-\U000026FF]+', '', token)
            token = token.replace('’', "'").replace('`', "'").replace("''", "'")
            if token:
                yield token
        except json.JSONDecodeError as e:
            print(f"\033[38;5;196m[Ollama Error] JSON decode error: {e}, chunk: {chunk}\033[0m")
            continue
    stderr, _ = await proc.communicate()
    if stderr:
        print(f"\033[38;5;196m[Ollama STDERR] {stderr.decode().strip()}\033[0m")
    await proc.wait()

async def stream_tts(text, piper_proc, voice):
    sample_rate = get_voice_sample_rate(voice)
    piper_proc.stdin.write(text.encode() + b'\n')
    await piper_proc.stdin.drain()
    raw_pcm = b""
    
    start_time = time.time()
    while True:
        try:
            chunk = await asyncio.wait_for(piper_proc.stdout.read(4096), timeout=0.5)
            if chunk:
                raw_pcm += chunk
                start_time = time.time()  # Reset idle timer on new data
        except asyncio.TimeoutError:
            if time.time() - start_time > 1.0:  # Break if idle for >1 second
                break
    #print(f"\033[38;5;200m[Debug] Piper PCM output size for '{text}': {len(raw_pcm)} bytes\033[0m")
    
    # Normalize amplitude to prevent clipping
    pcm_array = np.frombuffer(raw_pcm, dtype=np.int16)
    if len(pcm_array) > 0:
        max_amplitude = np.max(np.abs(pcm_array))
        if max_amplitude > 0:
            scale = 22937 / max_amplitude
            pcm_array = (pcm_array * scale).astype(np.int16)
            raw_pcm = pcm_array.tobytes()
    #print(f"\033[38;5;200m[Debug] Normalized PCM size for '{text}': {len(raw_pcm)} bytes\033[0m")
    
    # Adaptive silence padding: 150ms for short utterances (<0.5s), 20ms otherwise
    duration_secs = len(raw_pcm) / (sample_rate * 2)
    silence_ms = 150 if duration_secs < 0.5 else 20
    raw_pcm += b"\x00" * int(sample_rate * silence_ms / 1000 * 2)

    # Minimal SoX pipeline
    sox_cmd = [
        "sox",
        "-t", "raw", "-r", str(sample_rate), "-c", "1", "-b", "16", "-e", "signed-integer", "-",
        "-r", "48000", "-c", "2", "-t", "raw", "-",
        "gain", "-3"
    ]
    sox_proc = await asyncio.create_subprocess_exec(
        *sox_cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    sox_stdout, sox_stderr = await sox_proc.communicate(input=raw_pcm)
    if sox_stderr:
        print(f"\033[38;5;196m[SoX STDERR] {sox_stderr.decode().strip()}\033[0m")
    #print(f"\033[38;5;200m[Debug] SoX output size for '{text}': {len(sox_stdout)} bytes\033[0m")
    
    for i in range(0, len(sox_stdout), 2048):
        yield sox_stdout[i:i+2048]

async def process_connection(websocket):
    recognizer = KaldiRecognizer(vosk_model, RATE)
    session_config = None
    piper_proc = None
    last_stt_end_time = None
    first_tts_sent = False
    chat_history = []

    async for message in websocket:
        if isinstance(message, str):
            if message.strip() == "__done__":
                continue
            try:
                data = json.loads(message)
                if data.get("type") == "config_sync":
                    session_config = data.get("config", {})
                    print("[Server] Config synced:", session_config.get("voice"))
                    voice_model_path = f"voices/{session_config['voice']}"
                    if not os.path.exists(voice_model_path):
                        print(f"[ERROR] Voice model not found: {voice_model_path}")
                        await websocket.send("__ERROR__: Voice model not found.")
                        continue
                    try:
                        piper_proc = await asyncio.create_subprocess_exec(
                            PIPER_PATH, '--model', voice_model_path, '--output_raw',
                            stdin=asyncio.subprocess.PIPE,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE
                        )
                        asyncio.create_task(monitor_piper_stderr(piper_proc.stderr))
                    except Exception as e:
                        print(f"[ERROR] Failed to start Piper: {e}")
                        await websocket.send("__ERROR__: Piper failed to start.")
                        continue
            except json.JSONDecodeError:
                continue
            except Exception as e:
                print("[Server] Unexpected error:", e)
                continue
            continue

        if not isinstance(message, bytes):
            continue

        if session_config is None:
            continue

        if recognizer.AcceptWaveform(message):
            try:
                result = json.loads(recognizer.Result())
            except (json.JSONDecodeError, ValueError) as e:
                print(f"\033[38;5;196m[Vosk Error] Failed to parse recognizer result: {e}\033[0m")
                continue

            last_stt_end_time = time.time()
            first_tts_sent = False
            user_text = result.get("text", "").strip()
            if not user_text:
                continue

            cleaned = user_text.lower().strip(".,!? ")
            if cleaned in LOW_EFFORT_UTTERANCES:
                recognizer = KaldiRecognizer(vosk_model, RATE)
                continue

            print(f"\033[38;5;35m[User]: {user_text}\033[0m")
            chat_history.append({"role": "user", "content": user_text})
            context = [{"role": "system", "content": session_config.get("system_prompt", "You are a helpful voice assistant. Provide concise, meaningful responses.")}] + chat_history[-session_config.get("history_length", 0):]
            
            response_text = ""
            full_response = ""
            last_flush_time = time.time()
            batch_sentences = []

            async for token in stream_ollama_response(session_config["model_name"], context):
                response_text += token
                
                current_time = time.time()
                if current_time - last_flush_time >= FLUSH_INTERVAL or len(response_text) > MAX_RESPONSE_LENGTH:
                    response_text = clean_response(response_text)
                    sentences = split_sentences(response_text)
                    if sentences:
                        *complete_sentences, response_text = sentences
                        batch_sentences.extend(complete_sentences)
                        # Process batched sentences if enough accumulated or time threshold reached
                        if len(batch_sentences) >= MIN_BATCH_SENTENCES or current_time - last_flush_time >= BATCH_FLUSH_INTERVAL:
                            # Combine sentences with a space, preserving punctuation
                            combined_segment = " ".join(clean_response(sentence).strip() for sentence in batch_sentences if sentence.strip() and not re.fullmatch(r"[.?!\-–—…]+", sentence))
                            if combined_segment:
                                if not first_tts_sent and last_stt_end_time:
                                    tts_latency = int((current_time - last_stt_end_time) * 1000)
                                    print(f"\033[38;5;220m[Perf] STT → TTS latency: {tts_latency} ms\033[0m")
                                    first_tts_sent = True
                                print(f"\033[38;5;75m[AI]: {combined_segment}\033[0m")
                                full_response += combined_segment + " "
                                async for chunk in stream_tts(combined_segment, piper_proc, session_config["voice"]):
                                    await websocket.send(chunk)
                            batch_sentences = []
                            last_flush_time = current_time
                    #print(f"\033[38;5;200m[Debug] response_text after flush: '{response_text}'\033[0m")

            response_text = clean_response(response_text)
            sentences = split_sentences(response_text)
            if response_text and not sentences:
                sentences = [response_text]
            batch_sentences.extend(sentences)

            # Process remaining batched sentences
            combined_segment = " ".join(clean_response(sentence).strip() for sentence in batch_sentences if sentence.strip() and not re.fullmatch(r"[.?!\-–—…]+", sentence))
            if combined_segment:
                if not first_tts_sent and last_stt_end_time:
                    tts_latency = int((time.time() - last_stt_end_time) * 1000)
                    print(f"\033[38;5;220m[Perf] STT → TTS latency: {tts_latency} ms\033[0m")
                    first_tts_sent = True
                print(f"\033[38;5;75m[AI]: {combined_segment}\033[0m")
                full_response += combined_segment + " "
                async for chunk in stream_tts(combined_segment, piper_proc, session_config["voice"]):
                    await websocket.send(chunk)
            #print(f"\033[38;5;200m[Debug] Final response_text: '{response_text}'\033[0m")

            # Fallback for empty or invalid responses
            if not full_response.strip() or re.fullmatch(r'[^\w\s]+', full_response.strip()):
                fallback_response = "Sorry, I didn't catch that. Could you repeat?"
                print(f"\033[38;5;75m[AI]: {fallback_response}\033[0m")
                async for chunk in stream_tts(fallback_response, piper_proc, session_config["voice"]):
                    await websocket.send(chunk)
                full_response = fallback_response

            # After full_response is complete
            truncated_response = full_response.strip()[:200] + " [truncated]" if len(full_response) > 200 else full_response.strip()
            chat_history.append({"role": "assistant", "content": truncated_response})
            await websocket.send("__END__")

    try:
        if piper_proc and piper_proc.stdin:
            piper_proc.stdin.close()
            await piper_proc.wait()
    except Exception as e:
        print(f"[Piper] Shutdown error: {e}")

async def warm_up_ollama(model_name, system_prompt):
    print(f"[Server] Warming up Ollama model: {model_name}")
    try:
        response = await query_ollama(model_name, [{"role": "system", "content": system_prompt}, {"role": "user", "content": "Hello"}])
        print("[Server] Ollama model is warm.")
    except Exception as e:
        print(f"[Server] Ollama warm-up failed: {e}")

async def main():
    config = load_config()
    await warm_up_ollama(config["model_name"], config.get("system_prompt", "You are a helpful voice assistant. Provide concise, meaningful responses."))
    print("[Server] Listening on ws://0.0.0.0:8765 ...")
    async with websockets.serve(process_connection, "0.0.0.0", 8765, ping_timeout=None, ping_interval=None):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())