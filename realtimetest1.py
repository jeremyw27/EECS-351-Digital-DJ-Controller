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
    def __init__(self, master, volume_lock, volume_var, lpf_lock, lpf_var, hpf_lock, hpf_var):
        self.volume_lock = volume_lock
        self.volume_var = volume_var
        self.lpf_lock = lpf_lock
        self.lpf_var = lpf_var
        self.hpf_lock = hpf_lock
        self.hpf_var = hpf_var
        self.master = master
        master.title("Volume, LPF, and HPF Slider")

        self.volume_label = tk.Label(master, text="Current Volume:")
        self.volume_label.pack()

        self.volume_slider = tk.Scale(master, from_=0, to=100, orient=tk.HORIZONTAL, length=400)
        self.volume_slider.set(50)  # Setting initial volume to 50%
        self.volume_slider.pack()

        self.lpf_label = tk.Label(master, text="LPF Cutoff Frequency:")
        self.lpf_label.pack()

        self.lpf_slider = tk.Scale(master, from_=20, to=20000, orient=tk.HORIZONTAL, length=400)  # Low-pass filter slider
        self.lpf_slider.set(5000)
        self.lpf_slider.pack()

        self.hpf_label = tk.Label(master, text="HPF Cutoff Frequency:")
        self.hpf_label.pack()

        self.hpf_slider = tk.Scale(master, from_=20, to=20000, orient=tk.HORIZONTAL, length=400)  # High-pass filter slider
        self.hpf_slider.set(20)
        self.hpf_slider.pack()

        self.current_volume = tk.DoubleVar(value=self.volume_slider.get())
        self.volume_slider.config(variable=self.current_volume)
        
        self.current_lpf = tk.DoubleVar(value=self.lpf_slider.get())
        self.lpf_slider.config(variable=self.current_lpf)

        self.current_hpf = tk.DoubleVar(value=self.hpf_slider.get())
        self.hpf_slider.config(variable=self.current_hpf)

        self.volume_slider.bind("<Motion>", self.update_volume)
        self.volume_slider.bind("<ButtonRelease-1>", self.update_volume)
        self.lpf_slider.bind("<Motion>", self.update_lpf)
        self.lpf_slider.bind("<ButtonRelease-1>", self.update_lpf)
        self.hpf_slider.bind("<Motion>", self.update_hpf)
        self.hpf_slider.bind("<ButtonRelease-1>", self.update_hpf)

    def update_volume(self, event=None):
        with self.volume_lock:
            self.volume_var[0] = self.current_volume.get() / 100.0

    def update_lpf(self, event=None):
        with self.lpf_lock:
            self.lpf_var[0] = self.current_lpf.get()

    def update_hpf(self, event=None):
        with self.hpf_lock:
            self.hpf_var[0] = self.current_hpf.get()

def create_slider_app(volume_lock, volume_var, lpf_lock, lpf_var, hpf_lock, hpf_var):
    root = tk.Tk()
    app = SliderApp(root, volume_lock, volume_var, lpf_lock, lpf_var, hpf_lock, hpf_var)
    return app, root

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
volume = [0.5]  #Using a list to allow mutable access
lpf_lock = threading.Lock()
lpf = [5000]  
hpf_lock = threading.Lock()
hpf = [20]   

lpfslider, root = create_slider_app(volume_lock, volume, lpf_lock, lpf, hpf_lock, hpf)

#calculate filter parameters
def butter_filter(cutoff, fs, btype, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype=btype, analog=False)
    return b, a

#apply fillter with initialization
def apply_filter(data, b, a):
    #apply the filter for each channel one at a time
    y = np.zeros_like(data)  #initialize an array to store filtered data
    zi = lfilter_zi(b, a)  #initialize filter state
    for i in range(data.shape[1]): 
        y[:, i], _ = lfilter(b, a, data[:, i], zi=zi*data[0, i]) #apply filter with initial state
    return y

def apply_effects(audio_data, volume, lpf_value, hpf_value, sample_rate):
    #volume
    data = audio_data * volume

    #low pass filter
    if lpf_value < 20000:
        b_lpf, a_lpf = butter_filter(lpf_value, sample_rate, 'low')
        data = apply_filter(data, b_lpf, a_lpf)

    #high pass filter
    if hpf_value > 20:
        b_hpf, a_hpf = butter_filter(hpf_value, sample_rate, 'high')
        data = apply_filter(data, b_hpf, a_hpf)
    
    return data.astype(np.float32)

def callback(outdata, frames, time, status):
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
    
    #receive volume and cut off freqs from ui
    with volume_lock:
        current_volume = volume[0]
    with lpf_lock:
        current_lpf = lpf[0]
    with hpf_lock:
        current_hpf = hpf[0]
    
    #effects
    try:
        data = apply_effects(data, current_volume, current_lpf, current_hpf, samplerate)
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
        channels = wf.getnchannels()
        dtype = np.int16  #assume 16-bit PCM
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
        print(f"Exception in file_reader_thread: {e}")

try:
    wave_gen = read_wav_file(args.filename, BLOCK_SIZE)
    for _ in range(BUFFER_SIZE):
        data = next(wave_gen, None)
        if data is None:
            break
        q.put_nowait(data)  #prefill queue
    
    stream = sd.OutputStream(
        samplerate=samplerate, blocksize=BLOCK_SIZE,
        device=args.device, channels=2, dtype='float32',
        callback=callback, finished_callback=event.set
    )
    with stream:
        timeout = BLOCK_SIZE * BUFFER_SIZE / samplerate
        reader_thread = threading.Thread(target=file_reader_thread, args=(wave_gen, timeout))
        reader_thread.start()
        
        root.mainloop()  #run tkinter event loop
        
        reader_thread.join()  #ensure the reader thread completes
        event.wait()  #wait until playback is finished
except KeyboardInterrupt:
    parser.exit('\nInterrupted by user')
except queue.Full:
    parser.exit(1)
except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))