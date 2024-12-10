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
    def __init__(self, master, volume_lock, volume_var, lpf_lock, lpf_var, hpf_lock, hpf_var, echo_delay_lock, delay_var, echo_feedback_lock, feedback_var):
        self.volume_lock = volume_lock
        self.volume_var = volume_var
        self.lpf_lock = lpf_lock
        self.lpf_var = lpf_var
        self.hpf_lock = hpf_lock
        self.hpf_var = hpf_var
        self.echo_delay_lock = echo_delay_lock
        self.delay_var = delay_var
        self.echo_feedback_lock = echo_feedback_lock
        self.feedback_var = feedback_var

        self.master = master
        master.title("Volume, LPF, HPF, and Echo Slider")

        self.volume_label = tk.Label(master, text="Current Volume:")
        self.volume_label.pack()
        self.volume_slider = tk.Scale(master, from_=0, to=100, orient=tk.HORIZONTAL, length=400)
        self.volume_slider.set(50)
        self.volume_slider.pack()

        self.lpf_label = tk.Label(master, text="LPF Cutoff Frequency:")
        self.lpf_label.pack()
        self.lpf_slider = tk.Scale(master, from_=20, to=20000, orient=tk.HORIZONTAL, length=400)
        self.lpf_slider.set(5000)
        self.lpf_slider.pack()

        self.hpf_label = tk.Label(master, text="HPF Cutoff Frequency:")
        self.hpf_label.pack()
        self.hpf_slider = tk.Scale(master, from_=20, to=20000, orient=tk.HORIZONTAL, length=400)
        self.hpf_slider.set(20)
        self.hpf_slider.pack()

        self.echo_delay_label = tk.Label(master, text="Echo Delay (s):")
        self.echo_delay_label.pack()
        self.echo_delay_slider = tk.Scale(master, from_=0, to=2, orient=tk.HORIZONTAL, length=400, resolution=0.01)
        self.echo_delay_slider.set(0.3)
        self.echo_delay_slider.pack()

        self.echo_feedback_label = tk.Label(master, text="Echo Feedback:")
        self.echo_feedback_label.pack()
        self.echo_feedback_slider = tk.Scale(master, from_=0, to=2, orient=tk.HORIZONTAL, length=400, resolution=0.01)
        self.echo_feedback_slider.set(0.5)
        self.echo_feedback_slider.pack()

        self.current_volume = tk.DoubleVar(value=self.volume_slider.get())
        self.volume_slider.config(variable=self.current_volume)
        self.current_lpf = tk.DoubleVar(value=self.lpf_slider.get())
        self.lpf_slider.config(variable=self.current_lpf)
        self.current_hpf = tk.DoubleVar(value=self.hpf_slider.get())
        self.hpf_slider.config(variable=self.current_hpf)
        self.current_echo_delay = tk.DoubleVar(value=self.echo_delay_slider.get())
        self.echo_delay_slider.config(variable=self.current_echo_delay)
        self.current_echo_feedback = tk.DoubleVar(value=self.echo_feedback_slider.get())
        self.echo_feedback_slider.config(variable=self.current_echo_feedback)

        self.volume_slider.bind("<Motion>", self.update_volume)
        self.volume_slider.bind("<ButtonRelease-1>", self.update_volume)
        self.lpf_slider.bind("<Motion>", self.update_lpf)
        self.lpf_slider.bind("<ButtonRelease-1>", self.update_lpf)
        self.hpf_slider.bind("<Motion>", self.update_hpf)
        self.hpf_slider.bind("<ButtonRelease-1>", self.update_hpf)
        self.echo_delay_slider.bind("<Motion>", self.update_echo_delay)
        self.echo_delay_slider.bind("<ButtonRelease-1>", self.update_echo_delay)
        self.echo_feedback_slider.bind("<Motion>", self.update_echo_feedback)
        self.echo_feedback_slider.bind("<ButtonRelease-1>", self.update_echo_feedback)

    def update_volume(self, event=None):
        with self.volume_lock:
            self.volume_var[0] = self.current_volume.get() / 100.0
        print(f"Updated volume: {self.volume_var[0]}")

    def update_lpf(self, event=None):
        with self.lpf_lock:
            self.lpf_var[0] = self.current_lpf.get()
        print(f"Updated LPF: {self.lpf_var[0]}")

    def update_hpf(self, event=None):
        with self.hpf_lock:
            self.hpf_var[0] = self.current_hpf.get()
        print(f"Updated HPF: {self.hpf_var[0]}")

    def update_echo_delay(self, event=None):
        with self.echo_delay_lock:
            self.delay_var[0] = self.current_echo_delay.get()
        print(f"Updated Echo Delay: {self.delay_var[0]}")

    def update_echo_feedback(self, event=None):
        with self.echo_feedback_lock:
            self.feedback_var[0] = self.current_echo_feedback.get()
        print(f"Updated Echo Feedback: {self.feedback_var[0]}")

def create_slider_app(volume_lock, volume_var, lpf_lock, lpf_var, hpf_lock, hpf_var, echo_delay_lock, delay_var, echo_feedback_lock, feedback_var):
    root = tk.Tk()
    app = SliderApp(root, volume_lock, volume_var, lpf_lock, lpf_var, hpf_lock, hpf_var, echo_delay_lock, delay_var, echo_feedback_lock, feedback_var)
    return app, root

def butter_filter(cutoff, fs, btype, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype=btype, analog=False)
    return b, a

def apply_filter(data, b, a):
    y = np.zeros_like(data)
    zi = lfilter_zi(b, a)
    for i in range(data.shape[1]):
        y[:, i], _ = lfilter(b, a, data[:, i], zi=zi*data[0, i])
    return y

def apply_effects(audio_data, volume, lpf_value, hpf_value, sample_rate, delay=0.3, feedback=0.5):
    data = audio_data * volume

    if lpf_value < 20000:
        b_lpf, a_lpf = butter_filter(lpf_value, sample_rate, 'low')
        data = apply_filter(data, b_lpf, a_lpf)

    if hpf_value > 20:
        b_hpf, a_hpf = butter_filter(hpf_value, sample_rate, 'high')
        data = apply_filter(data, b_hpf, a_hpf)

    # Initialize the delay buffer
    echo_delay_samples = int(sample_rate * delay)
    buffer = np.zeros((echo_delay_samples, data.shape[1]), dtype=np.float32)
    buffer_index = 0

    feedback = np.clip(feedback, 0.0, 1.0)

    print(f"Applying echo effect with echo_delay_samples={echo_delay_samples} and feedback={feedback}")

    for i in range(len(data)):
        out_sample = data[i] + feedback * buffer[buffer_index]
        buffer[buffer_index] = out_sample
        buffer_index = (buffer_index + 1) % echo_delay_samples
        data[i] = out_sample

    data = np.clip(data, -1.0, 1.0)
    return data.astype(np.float32)

def callback(outdata, frames, time, status, volume_lock, volume, lpf_lock, lpf, hpf_lock, hpf, echo_delay_lock, delay_var, echo_feedback_lock, feedback_var):
    assert frames == BLOCK_SIZE
    if status.output_underflow:
        print('Output underflow: increase blocksize?', file=sys.stderr)
        raise sd.CallbackAbort
    assert not status

    try:
        data = q.get_nowait()
    except queue.Empty:
        print('Buffer is empty: increase buffersize?', file=sys.stderr)
        raise sd.CallbackAbort

    with volume_lock:
        current_volume = volume[0]
    with lpf_lock:
        current_lpf = lpf[0]
    with hpf_lock:
        current_hpf = hpf[0]

    with echo_delay_lock:
        current_delay = delay_var[0]
    with echo_feedback_lock:
        current_feedback = feedback_var[0]

    print(f"Callback processing with volume={current_volume}, lpf={current_lpf}, hpf={current_hpf}, delay={current_delay}, feedback={current_feedback}")

    try:
        data = apply_effects(data, current_volume, current_lpf, current_hpf, samplerate, delay=current_delay, feedback=current_feedback)
    except Exception as e:
        print(f"Error applying effects: {e}", file=sys.stderr)
        raise sd.CallbackAbort

    if len(data) < len(outdata):
        outdata[:len(data)] = data
        outdata[len(data):] = np.zeros((len(outdata) - len(data), data.shape[1]), dtype=outdata.dtype)
        raise sd.CallbackStop
    else:
        outdata[:] = data

def read_wav_file(filename, blocksize):
    with wave.open(filename, 'rb') as wf:
        global samplerate
        samplerate = wf.getframerate()
        print("Sample rate:", samplerate)
        channels = wf.getnchannels()
        dtype = np.int16
        while True:
            data = wf.readframes(blocksize)
            if not data:
                break
            yield np.frombuffer(data, dtype=dtype).reshape(-1, channels).astype(np.float32) / 32768.0

def file_reader_thread(wave_gen, timeout):
    try:
        while True:
            data = next(wave_gen, None)
            if data is None:
                break
            q.put(data, timeout=timeout)
    except Exception as e:
        print(f"Exception in file_reader_thread: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()

def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-l', '--list-devices', action='store_true', help='show list of audio devices and exit')
args, remaining = parser.parse_known_args()
if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, parents=[parser])
parser.add_argument('filename', metavar='FILENAME', help='audio file to be played back')
parser.add_argument('-d', '--device', type=int_or_str, help='output device (numeric ID or substring)')
parser.add_argument('-b', '--blocksize', type=int, default=2048, help='block size (default: %(default)s)')
parser.add_argument('-q', '--buffersize', type=int, default=30, help='number of blocks used for buffering (default: %(default)s)')
args = parser.parse_args(remaining)
if args.blocksize == 0:
    parser.error('blocksize must not be zero')
if args.buffersize < 1:
    parser.error('buffersize must be at least 1')

BLOCK_SIZE = args.blocksize
BUFFER_SIZE = args.buffersize

q = queue.Queue(maxsize=BUFFER_SIZE)
event = threading.Event()

volume_lock = threading.Lock()
volume = [0.5]
lpf_lock = threading.Lock()
lpf = [5000]
hpf_lock = threading.Lock()
hpf = [20]
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
        callback=lambda *args: callback(*args, 
                                        volume_lock=volume_lock, volume=volume, 
                                        lpf_lock=lpf_lock, lpf=lpf, 
                                        hpf_lock=hpf_lock, hpf=hpf, 
                                        echo_delay_lock=echo_delay_lock, delay_var=delay_var, 
                                        echo_feedback_lock=echo_feedback_lock, feedback_var=feedback_var),
        finished_callback=event.set
    )

    with stream:
        timeout = BLOCK_SIZE * BUFFER_SIZE / samplerate
        reader_thread = threading.Thread(target=file_reader_thread, args=(wave_gen, timeout))
        reader_thread.start()

        app, root = create_slider_app(volume_lock, volume, lpf_lock, lpf, hpf_lock, hpf, echo_delay_lock, delay_var, echo_feedback_lock, feedback_var)
        root.mainloop()

        reader_thread.join()
        event.wait()
except KeyboardInterrupt:
    parser.exit('\nInterrupted by user')
except queue.Full:
    parser.exit(1)
except Exception as e:
    import traceback
    traceback.print_exc()
    parser.exit(type(e).__name__ + ': ' + str(e))