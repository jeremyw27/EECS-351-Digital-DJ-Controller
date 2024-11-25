#!/usr/bin/env python3
"""
Play an audio file using a limited amount of memory.

This example uses numpy and wave modules to read the audio file. 

"""
import argparse
import queue
import sys
import threading
import wave

import numpy as np
import sounddevice as sd

def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    '-l', '--list-devices', action='store_true',
    help='show list of audio devices and exit'
)
args, remaining = parser.parse_known_args()
if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    parents=[parser]
)
parser.add_argument(
    'filename', metavar='FILENAME',
    help='audio file to be played back'
)
parser.add_argument(
    '-d', '--device', type=int_or_str,
    help='output device (numeric ID or substring)'
)
parser.add_argument(
    '-b', '--blocksize', type=int, default=2048,
    help='block size (default: %(default)s)'
)
parser.add_argument(
    '-q', '--buffersize', type=int, default=20,
    help='number of blocks used for buffering (default: %(default)s)'
)
args = parser.parse_args(remaining)
if args.blocksize == 0:
    parser.error('blocksize must not be zero')
if args.buffersize < 1:
    parser.error('buffersize must be at least 1')

q = queue.Queue(maxsize=args.buffersize)
event = threading.Event()

def apply_effects(audio_data, volume=.3):
    # Apply a simple volume effect (scaling the amplitude)
    return audio_data * volume

def callback(outdata, frames, time, status):
    assert frames == args.blocksize
    if status.output_underflow:
        print('Output underflow: increase blocksize?', file=sys.stderr)
        raise sd.CallbackAbort
    assert not status
    try:
        data = q.get_nowait()
    except queue.Empty:
        print('Buffer is empty: increase buffersize?', file=sys.stderr)
        raise sd.CallbackAbort
    
    data = apply_effects(data)

    if len(data) < len(outdata):
        outdata[:len(data)] = data
        outdata[len(data):] = b'\x00' * (len(outdata) - len(data))
        raise sd.CallbackStop
    else:
        outdata[:] = data

def read_wav_file(filename, blocksize):
    with wave.open(filename, 'rb') as wf:
        samplerate = wf.getframerate()
        channels = wf.getnchannels()
        dtype = np.int16  # Assuming 16-bit PCM
        while True:
            data = wf.readframes(blocksize)
            if not data:
                break
            yield np.frombuffer(data, dtype=dtype).reshape(-1, channels).astype('float32') / 32768.0

try:
    wave_gen = read_wav_file(args.filename, args.blocksize)
    samplerate = None
    channels = None
    for _ in range(args.buffersize):
        data = next(wave_gen, None)
        if data is None:
            break
        if samplerate is None or channels is None:
            with wave.open(args.filename, 'rb') as wf:
                samplerate = wf.getframerate()
                channels = wf.getnchannels()
        q.put_nowait(data)  # Pre-fill queue
    
    stream = sd.OutputStream(
        samplerate=samplerate, blocksize=args.blocksize,
        device=args.device, channels=channels, dtype='float32',
        callback=callback, finished_callback=event.set
    )
    with stream:
        timeout = args.blocksize * args.buffersize / samplerate
        while data is not None:
            data = next(wave_gen, None)
            if data is not None:
                q.put(data, timeout=timeout)
        event.wait()  # Wait until playback is finished
except KeyboardInterrupt:
    parser.exit('\nInterrupted by user')
except queue.Full:
    # A timeout occurred, i.e. there was an error in the callback
    parser.exit(1)
except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))