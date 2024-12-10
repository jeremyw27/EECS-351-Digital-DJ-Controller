"""
To play this file run "python .\livefiltering.py song.wav"  where your song is the song.wav, do "python .\livefiltering.py -help" for help
you need tkinter, sounddevice, numpy, and scipy
"""

import time
import math
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
    def __init__(self, master, volume_lock, volume_var, lpf_lock, lpf_var, hpf_lock, hpf_var, low_gain_lock, low_gain_var, mid_gain_lock, mid_gain_var, high_gain_lock, high_gain_var, pause_lock, pause_var):
        self.volume_lock = volume_lock
        self.volume_var = volume_var
        self.lpf_lock = lpf_lock
        self.lpf_var = lpf_var
        self.hpf_lock = hpf_lock
        self.hpf_var = hpf_var
        self.low_gain_lock = low_gain_lock
        self.low_gain_var = low_gain_var
        self.mid_gain_lock = mid_gain_lock
        self.mid_gain_var = mid_gain_var
        self.high_gain_lock = high_gain_lock
        self.high_gain_var = high_gain_var
        self.pause_lock = pause_lock
        self.pause_var = pause_var
        self.master = master
        master.title("Volume, LPF, HPF, and Gain Sliders")

        self.volume_label = tk.Label(master, text="Current Volume:")
        self.volume_label.pack()

        self.volume_slider = tk.Scale(master, from_=0, to=100, orient=tk.HORIZONTAL, length=400)
        self.volume_slider.set(50)  #initial volume at 50%
        self.volume_slider.pack()

        self.lpf_label = tk.Label(master, text="LPF Cutoff Frequency:")
        self.lpf_label.pack()

        self.lpf_slider = tk.Scale(master, from_=20, to=20000, orient=tk.HORIZONTAL, length=400)  #lpf slider
        self.lpf_slider.set(5000)
        self.lpf_slider.pack()

        self.hpf_label = tk.Label(master, text="HPF Cutoff Frequency:")
        self.hpf_label.pack()

        self.hpf_slider = tk.Scale(master, from_=20, to=20000, orient=tk.HORIZONTAL, length=400)  #hpf slider
        self.hpf_slider.set(20)
        self.hpf_slider.pack()

        self.low_gain_label = tk.Label(master, text="Low Frequency Gain:")
        self.low_gain_label.pack()

        self.low_gain_slider = tk.Scale(master, from_=-20, to=20, orient=tk.HORIZONTAL, length=400)  #low eq slider
        self.low_gain_slider.set(0)  #0 initial gain
        self.low_gain_slider.pack()

        self.mid_gain_label = tk.Label(master, text="Mid Frequency Gain:")
        self.mid_gain_label.pack()

        self.mid_gain_slider = tk.Scale(master, from_=-20, to=20, orient=tk.HORIZONTAL, length=400)  #mid eq slider
        self.mid_gain_slider.set(0)  #0 initial gain
        self.mid_gain_slider.pack()

        self.high_gain_label = tk.Label(master, text="High Frequency Gain:")
        self.high_gain_label.pack()

        self.high_gain_slider = tk.Scale(master, from_=-20, to=20, orient=tk.HORIZONTAL, length=400)  #high eq slider
        self.high_gain_slider.set(0)  #0 initial gain
        self.high_gain_slider.pack()

        

        self.current_volume = tk.DoubleVar(value=self.volume_slider.get())
        self.volume_slider.config(variable=self.current_volume)
        
        self.current_lpf = tk.DoubleVar(value=self.lpf_slider.get())
        self.lpf_slider.config(variable=self.current_lpf)

        self.current_hpf = tk.DoubleVar(value=self.hpf_slider.get())
        self.hpf_slider.config(variable=self.current_hpf)

        self.current_low_gain = tk.DoubleVar(value=self.low_gain_slider.get())
        self.low_gain_slider.config(variable=self.current_low_gain)

        self.current_mid_gain = tk.DoubleVar(value=self.mid_gain_slider.get())
        self.mid_gain_slider.config(variable=self.current_mid_gain)

        self.current_high_gain = tk.DoubleVar(value=self.high_gain_slider.get())
        self.high_gain_slider.config(variable=self.current_high_gain)

        self.volume_slider.bind("<Motion>", self.update_volume)
        self.volume_slider.bind("<ButtonRelease-1>", self.update_volume)
        self.lpf_slider.bind("<Motion>", self.update_lpf)
        self.lpf_slider.bind("<ButtonRelease-1>", self.update_lpf)
        self.hpf_slider.bind("<Motion>", self.update_hpf)
        self.hpf_slider.bind("<ButtonRelease-1>", self.update_hpf)
        self.low_gain_slider.bind("<Motion>", self.update_low_gain)
        self.low_gain_slider.bind("<ButtonRelease-1>", self.update_low_gain)
        self.mid_gain_slider.bind("<Motion>", self.update_mid_gain)
        self.mid_gain_slider.bind("<ButtonRelease-1>", self.update_mid_gain)
        self.high_gain_slider.bind("<Motion>", self.update_high_gain)
        self.high_gain_slider.bind("<ButtonRelease-1>", self.update_high_gain)

    def update_volume(self, event=None):
        with self.volume_lock:
            self.volume_var[0] = self.current_volume.get() / 100.0

    def update_lpf(self, event=None):
        with self.lpf_lock:
            self.lpf_var[0] = self.current_lpf.get()

    def update_hpf(self, event=None):
        with self.hpf_lock:
            self.hpf_var[0] = self.current_hpf.get()

    def update_low_gain(self, event=None):
        with self.low_gain_lock:
            self.low_gain_var[0] = self.current_low_gain.get()

    def update_mid_gain(self, event=None):
        with self.mid_gain_lock:
            self.mid_gain_var[0] = self.current_mid_gain.get()

    def update_high_gain(self, event=None):
        with self.high_gain_lock:
            self.high_gain_var[0] = self.current_high_gain.get()

    def toggle_pause(self):
        with self.pause_lock:
            self.pause_var[0] = not self.pause_var[0]
        self.pause_button.config(text="Resume" if self.pause_var[0] else "Pause")

def create_slider_app(volume_lock, volume_var, lpf_lock, lpf_var, hpf_lock, hpf_var, low_gain_lock, low_gain_var, mid_gain_lock, mid_gain_var, high_gain_lock, high_gain_var, pause_lock, pause_var):
    root = tk.Tk()
    app = SliderApp(root, volume_lock, volume_var, lpf_lock, lpf_var, hpf_lock, hpf_var, low_gain_lock, low_gain_var, mid_gain_lock, mid_gain_var, high_gain_lock, high_gain_var, pause_lock, pause_var)
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
volume = [0.5]   #use lists so it's mutable
lpf_lock = threading.Lock()
lpf = [5000]  
hpf_lock = threading.Lock()
hpf = [20]    
low_gain_lock = threading.Lock()
low_gain = [0]  
mid_gain_lock = threading.Lock()
mid_gain = [0]  
high_gain_lock = threading.Lock()
high_gain = [0] 
pause_lock = threading.Lock()
pause = [False]

slider_app, root = create_slider_app(volume_lock, volume, lpf_lock, lpf, hpf_lock, hpf, low_gain_lock, low_gain, mid_gain_lock, mid_gain, high_gain_lock, high_gain, pause_lock, pause)

#calc hpf and lpf filter parameters
def butter_filter(cutoff, fs, btype, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype=btype, analog=False)
    return b, a

def higheqfilter(gain, fs):  
    pi = 3.14159
    if(gain>=0):   #parametric eq
       
        f0 = 12500
        Bf = 7500
        G = gain
        GB = 0.5 * gain
        G0 = 0
        beta = math.tan(Bf/2*pi/(fs/2))*math.sqrt(abs((10**(GB/20))**2 
            - (10**(G0/20))**2))/math.sqrt(abs(10**(G/20)**2 - (10**(GB/20))**2))
        b = [0] * 3
        a = [0] * 3
        b[0] = (10**(G0/20) + 10**(G/20)*beta)/(1 + beta)
        b[1] = -2*10**(G0/20)*math.cos(f0*pi/(fs/2))/(1 + beta)
        b[2] = (10**(G0/20)-10**(G/20)*beta)/(1 + beta)

        a[0] = 1
        a[1] = -2 * math.cos(f0*pi/(fs/2))/(1 + beta)
        a[2] = (1 - beta)/(1 + beta)

    else:   
        
        f0 = 12500
        bf = 1
        b = [0] * 3
        a = [0] * 3
        G = 10**(gain/40)
        omega0 = 2*pi*f0/fs
        alpha = math.sin(omega0)*math.sinh((math.log(2)/2)*bf*(omega0/math.sin(omega0)))
        
        b[0] = 1+alpha*G
        b[1] = -2*math.cos(omega0)
        b[2] = 1-alpha*G
        a[0] = 1+alpha/G
        a[1] = -2*math.cos(omega0)
        a[2] = 1-alpha/G

    return b, a

def mideqfilter(gain, fs):
    pi = 3.14159
    if(gain>=0):
       
        f0 = 2600
        Bf = 2400
        G = gain
        GB = 0.5 * gain
        G0 = 0
        beta = math.tan(Bf/2*pi/(fs/2))*math.sqrt(abs((10**(GB/20))**2 
            - (10**(G0/20))**2))/math.sqrt(abs(10**(G/20)**2 - (10**(GB/20))**2))
        b = [0] * 3
        a = [0] * 3
        b[0] = (10**(G0/20) + 10**(G/20)*beta)/(1 + beta)
        b[1] = -2*10**(G0/20)*math.cos(f0*pi/(fs/2))/(1 + beta)
        b[2] = (10**(G0/20)-10**(G/20)*beta)/(1 + beta)

        a[0] = 1
        a[1] = -2 * math.cos(f0*pi/(fs/2))/(1 + beta)
        a[2] = (1 - beta)/(1 + beta)
    else:
        f0=2600
        bf = 2
        b = [0] * 3
        a = [0] * 3
        G = 10**(gain/40)
        omega0 = 2*pi*f0/fs
        alpha = math.sin(omega0)*math.sinh((math.log(2)/2)*bf*(omega0/math.sin(omega0)))
        b[0] = 1+alpha*G
        b[1] = -2*math.cos(omega0)
        b[2] = 1-alpha*G
        a[0] = 1+alpha/G
        a[1] = -2*math.cos(omega0)
        a[2] = 1-alpha/G

    return b, a

def loweqfilter(gain, fs):
    pi = 3.14159
    if(gain>=0):  
       
        f0 = 100
        Bf = 50
        G = gain
        GB = 0.5 * gain
        G0 = 0
        beta = math.tan(Bf/2*pi/(fs/2))*math.sqrt(abs((10**(GB/20))**2 
            - (10**(G0/20))**2))/math.sqrt(abs(10**(G/20)**2 - (10**(GB/20))**2))
        b = [0] * 3
        a = [0] * 3
        b[0] = (10**(G0/20) + 10**(G/20)*beta)/(1 + beta)
        b[1] = -2*10**(G0/20)*math.cos(f0*pi/(fs/2))/(1 + beta)
        b[2] = (10**(G0/20)-10**(G/20)*beta)/(1 + beta)

        a[0] = 1
        a[1] = -2 * math.cos(f0*pi/(fs/2))/(1 + beta)
        a[2] = (1 - beta)/(1 + beta)

    else:  
        f0 = 100
        bf = 2
        b = [0] * 3
        a = [0] * 3
        G = 10**(gain/40)
        omega0 = 2*pi*f0/fs
        alpha = math.sin(omega0)*math.sinh((math.log(2)/2)*bf*(omega0/math.sin(omega0)))
        b[0] = 1+alpha*G
        b[1] = -2*math.cos(omega0)
        b[2] = 1-alpha*G
        a[0] = 1+alpha/G
        a[1] = -2*math.cos(omega0)
        a[2] = 1-alpha/G

    return b, a

#Apply filter with Initialization
def apply_filter(data, b, a):
    #apply filter to each channel one at a time
    y = np.zeros_like(data)  #array to store filtered data
    zi = lfilter_zi(b, a)  #initialize filter state
    for i in range(data.shape[1]):  #go through each channel
        y[:, i], _ = lfilter(b, a, data[:, i], zi=zi*data[0, i])  #actually apply filter with inital conditions
    return y

def apply_effects(audio_data, volume, lpf_value, hpf_value, sample_rate, low_gain, mid_gain, high_gain):
    
    data = audio_data * volume
    a = []
    b = []
    filterapplied = False
    
    
    filterapplied = True  #lpf
    b, a = butter_filter(lpf_value, sample_rate, 'low')  #I would do the if statement to check this but it was making it mad, because b and a weren't initalized right sometimes idk this was easier
        

    
    if hpf_value > 20: #hpf
        
        b_hpf, a_hpf = butter_filter(hpf_value, sample_rate, 'high')
        if(filterapplied):
            b = np.convolve(b, b_hpf)  #convolve filter coefficient so I can do one filter call at the end to save processing time
            a = np.convolve(a, a_hpf)
        else:
            b = b_hpf
            a = a_hpf
        filterapplied = True
        

    if low_gain != 0:
        
        b_leq, a_leq = loweqfilter(low_gain, sample_rate)
        if(filterapplied):
            b = np.convolve(b, b_leq)
            a = np.convolve(a, a_leq)
        else:
            b = b_leq
            a = a_leq
        filterapplied = True
        

    if mid_gain != 0:
        
        b_meq, a_meq = mideqfilter(mid_gain, sample_rate)
        if(filterapplied):
            b = np.convolve(b, b_meq)
            a = np.convolve(a, a_meq)
        else:
            b = b_meq
            a = a_meq
        filterapplied = True
        
        

    if high_gain != 0:
        
        b_heq, a_heq = higheqfilter(high_gain, sample_rate)
        if(filterapplied):
            b = np.convolve(b, b_heq)
            a = np.convolve(a, a_heq)

        else:
            b = b_heq
            a = a_heq
        filterapplied = True
        

   
    if(filterapplied):
        data = apply_filter(data, b, a)
    return data.astype(np.float32)

def callback(outdata, frames, time, status):
    assert frames == BLOCK_SIZE
    if status.output_underflow:
        print('Output underflow: increase blocksize?', file=sys.stderr)
        raise sd.CallbackAbort
    assert not status

    with pause_lock:
        is_paused = pause[0]
    
    if not is_paused:
        try:
            data = q.get_nowait()  #get data from queue
        except queue.Empty:
            print('Buffer is empty: increase buffersize?', file=sys.stderr)
            raise sd.CallbackAbort
    else:
        data = np.zeros((frames, n_channels), dtype='float32')  #if playback is paused just feed it zeros

    #apply volume and filters from shared variables
    with volume_lock:
        current_volume = volume[0]
    with lpf_lock:
        current_lpf = lpf[0]
    with hpf_lock:
        current_hpf = hpf[0]
    with low_gain_lock:
        current_low_gain = low_gain[0]
    with mid_gain_lock:
        current_mid_gain = mid_gain[0]
    with high_gain_lock:
        current_high_gain = high_gain[0]

    if not is_paused: #only apply effects if it's not paused
        try:
            data = apply_effects(data, current_volume, current_lpf, current_hpf, samplerate, current_low_gain, current_mid_gain, current_high_gain)  
        except Exception as e:
            print(f"Error applying effects: {e}", file=sys.stderr)
            raise sd.CallbackAbort
    
    if len(data) < len(outdata):
        outdata[:len(data)] = data #makes sure always passing array of correct size to audio callback funciton, it crashes otherwise
        outdata[len(data):] = np.zeros((len(outdata) - len(data), data.shape[1]), dtype=outdata.dtype) 
        raise sd.CallbackStop
    else:
        outdata[:] = data

def read_wav_file(filename, blocksize):  #open wav file
    with wave.open(filename, 'rb') as wf:
        global samplerate  
        global n_channels
        samplerate = wf.getframerate() #get parameters
        n_channels = wf.getnchannels()
        dtype = np.int16  #assuming 16-bit PCM

        while True:
            data = wf.readframes(blocksize)
            if not data:
                break
            yield np.frombuffer(data, dtype=dtype).reshape(-1, n_channels).astype(np.float32) / 32768.0  #I found this line in the documentation for sound device

def file_reader_thread(wave_gen, timeout):
    try:
        while True:
            
            with pause_lock:
                
                if pause[0]:
                    
                    time.sleep(.1)
                    continue
            
            data = next(wave_gen, None) #keep looking through chunk iterator
            if data is None:
                
                break
            q.put(data, timeout=timeout)  #put data into queue
            
    except Exception as e:
        print(f"Exception in file_reader_thread: {e}")

try:
    wave_gen = read_wav_file(args.filename, BLOCK_SIZE)  #open song and return iterator to chunks of song
    for _ in range(BUFFER_SIZE):
        data = next(wave_gen, None)  #get initial data
        if data is None:
            break
        q.put_nowait(data)  #prefill data in queue
    
    stream = sd.OutputStream(
        samplerate=samplerate, blocksize=BLOCK_SIZE,
        device=args.device, channels=n_channels, dtype='float32',
        callback=callback, finished_callback=event.set  #open output stream
    )
    with stream:
        timeout = BLOCK_SIZE * BUFFER_SIZE / samplerate  #time inbetween audio callback calls
        reader_thread = threading.Thread(target=file_reader_thread, args=(wave_gen, timeout))  #run audio callback thread
        reader_thread.start() #start that shit
        
        root.mainloop()  #run ui
        
        reader_thread.join()  #ensure the reader thread completes
        event.wait()  #wait until playback is finished
except KeyboardInterrupt:
    parser.exit('\nInterrupted by user')
except queue.Full:
    parser.exit(1)
except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))