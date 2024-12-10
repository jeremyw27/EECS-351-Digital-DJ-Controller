import argparse
import queue
import sys
import threading
import wave
import tkinter as tk
import numpy as np
import sounddevice as sd
from scipy.signal import butter, lfilter, lfilter_zi

class SliderApp:
    def __init__(self, master, volume_lock, volume_var, echo_delay_lock, delay_var, echo_feedback_lock, feedback_var):
        self.volume_lock = volume_lock
        self.volume_var = volume_var
        self.echo_delay_lock = echo_delay_lock
        self.delay_var = delay_var
        self.echo_feedback_lock = echo_feedback_lock
        self.feedback_var = feedback_var

        self.master = master
        master.title("Volume and Echo Slider")

        self.volume_label = tk.Label(master, text="Current Volume:")
        self.volume_label.pack()
        self.volume_slider = tk.Scale(master, from_=0, to=100, orient=tk.HORIZONTAL, length=400)
        self.volume_slider.set(50)
        self.volume_slider.pack()

        self.echo_delay_label = tk.Label(master, text="Echo Delay (s):")
        self.echo_delay_label.pack()
        self.echo_delay_slider = tk.Scale(master, from_=0, to=2, orient=tk.HORIZONTAL, length=400, resolution=0.01)
        self.echo_delay_slider.set(0.3)
        self.echo_delay_slider.pack()

        self.echo_feedback_label = tk.Label(master, text="Echo Feedback:")
        self.echo_feedback_label.pack()
        self.echo_feedback_slider = tk.Scale(master, from_=0, to=1, orient=tk.HORIZONTAL, length=400, resolution=0.01)
        self.echo_feedback_slider.set(0.5)
        self.echo_feedback_slider.pack()

        self.current_volume = tk.DoubleVar(value=self.volume_slider.get())
        self.volume_slider.config(variable=self.current_volume)

        self.current_echo_delay = tk.DoubleVar(value=self.echo_delay_slider.get())
        self.echo_delay_slider.config(variable=self.current_echo_delay)

        self.current_echo_feedback = tk.DoubleVar(value=self.echo_feedback_slider.get())
        self.echo_feedback_slider.config(variable=self.current_echo_feedback)

        self.volume_slider.bind("<Motion>", self.update_volume)
        self.volume_slider.bind("<ButtonRelease-1>", self.update_volume)
        self.echo_delay_slider.bind("<Motion>", self.update_echo_delay)
        self.echo_delay_slider.bind("<ButtonRelease-1>", self.update_echo_delay)
        self.echo_feedback_slider.bind("<Motion>", self.update_echo_feedback)
        self.echo_feedback_slider.bind("<ButtonRelease-1>", self.update_echo_feedback)

    def update_volume(self, event=None):
        with self.volume_lock:
            self.volume_var[0] = self.current_volume.get() / 100.0
        print(f"Updated volume: {self.volume_var[0]}")

    def update_echo_delay(self, event=None):
        with self.echo_delay_lock:
            self.delay_var[0] = self.current_echo_delay.get()
        print(f"Updated Echo Delay: {self.delay_var[0]}")

    def update_echo_feedback(self, event=None):
        with self.echo_feedback_lock:
            self.feedback_var[0] = self.current_echo_feedback.get()
        print(f"Updated Echo Feedback: {self.feedback_var[0]}")

def create_slider_app(volume_lock, volume_var, echo_delay_lock, delay_var, echo_feedback_lock, feedback_var):
    root = tk.Tk()
    app = SliderApp(root, volume_lock, volume_var, echo_delay_lock, delay_var, echo_feedback_lock, feedback_var)
    return app, root

def apply_effects(audio_data, volume, sample_rate, delay, feedback):
    data = audio_data * volume
    delay_samples = int(sample_rate * delay)
    buffer = np.zeros((delay_samples, data.shape[1]), dtype=np.float32)
    buffer_index = 0
    feedback = np.clip(feedback, 0.0, 1.0)

    print(f"Applying echo: delay_samples={delay_samples}, feedback={feedback}")

    for i in range(len(data)):
        for channel in range(data.shape[1]):
            delayed_sample = buffer[buffer_index, channel]
            buffer[buffer_index, channel] = data[i, channel] + delayed_sample * feedback
            data[i, channel] += delayed_sample
        buffer_index = (buffer_index + 1) % delay_samples

    return np.clip(data, -1.0, 1.0).astype(np.float32)

def callback(outdata, frames, time, status, volume_lock, volume, echo_delay_lock, delay_var, echo_feedback_lock, feedback_var):
    if status:
        print(status, file=sys.stderr)

    try:
        data = q.get_nowait()
    except queue.Empty:
        raise sd.CallbackAbort

    with volume_lock:
        current_volume = volume[0]
    with echo_delay_lock:
        current_delay = delay_var[0]
    with echo_feedback_lock:
        current_feedback = feedback_var[0]

    print(f"Callback with volume={current_volume}, delay={current_delay}, feedback={current_feedback}")

    data = apply_effects(data, current_volume, samplerate, current_delay, current_feedback)
    outdata[:] = data

def read_wav_file(filename, blocksize):
    with wave.open(filename, 'rb') as wf:
        global samplerate
        samplerate = wf.getframerate()
        channels = wf.getnchannels()
        while True:
            data = wf.readframes(blocksize)
            if not data:
                break
            yield np.frombuffer(data, dtype=np.int16).reshape(-1, channels).astype(np.float32) / 32768.0

def file_reader_thread(wave_gen, timeout):
    try:
        while True:
            data = next(wave_gen, None)
            if data is None:
                break
            q.put(data, timeout=timeout)
    except Exception as e:
        print(f"Exception in file_reader_thread: {e}", file=sys.stderr)

def int_or_str(text):
    try:
        return int(text)
    except ValueError:
        return text

parser = argparse.ArgumentParser(description='Play a sound file with echo effect.')
parser.add_argument('filename', metavar='FILENAME', help='audio file to be played back')
parser.add_argument('-d', '--device', type=int_or_str, help='output device (numeric ID or substring)')
parser.add_argument('-b', '--blocksize', type=int, default=2048, help='block size (default: %(default)s)')
parser.add_argument('-q', '--buffersize', type=int, default=20, help='number of blocks used for buffering (default: %(default)s)')
args = parser.parse_args()
if args.blocksize <= 0:
    parser.error('blocksize must be positive')
if args.buffersize < 1:
    parser.error('buffersize must be at least 1')

BLOCK_SIZE = args.blocksize
BUFFER_SIZE = args.buffersize

q = queue.Queue(maxsize=BUFFER_SIZE)
event = threading.Event()

volume_lock = threading.Lock()
volume = [0.5]
echo_delay_lock = threading.Lock()
delay_var = [0.3]
echo_feedback_lock = threading.Lock()
feedback_var = [0.5]

try:
    wave_gen = read_wav_file(args.filename, BLOCK_SIZE)
    for _ in range(BUFFER_SIZE):
        data = next(wave_gen, None)
        if data is None:
            break
        q.put_nowait(data)

    stream = sd.OutputStream(
        samplerate=samplerate, blocksize=BLOCK_SIZE,
        device=args.device, channels=2, dtype='float32',
        callback=lambda *args: callback(*args, volume_lock=volume_lock, volume=volume, echo_delay_lock=echo_delay_lock, delay_var=delay_var, echo_feedback_lock=echo_feedback_lock, feedback_var=feedback_var),
        finished_callback=event.set
    )

    with stream:
        timeout = BLOCK_SIZE * BUFFER_SIZE / samplerate
        reader_thread = threading.Thread(target=file_reader_thread, args=(wave_gen, timeout))
        reader_thread.start()

        app, root = create_slider_app(volume_lock, volume, echo_delay_lock, delay_var, echo_feedback_lock, feedback_var)
        root.mainloop()

        reader_thread.join()
        event.wait()
except KeyboardInterrupt:
    print('\nInterrupted by user', file=sys.stderr)
except queue.Full:
    print('Queue full', file=sys.stderr)
except Exception as e:
    print(f'Error: {e}', file=sys.stderr)
    # Uncomment the following line for detailed debugging:
    # traceback.print_exc()