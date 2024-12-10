import argparse
import queue
import sys
import threading
import wave
import tkinter as tk
import numpy as np
import scipy.signal
import sounddevice as sd
import logging
import traceback

# Setup logging to output to the terminal as well
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(message)s', handlers=[
    logging.FileHandler("debug.log"),
    logging.StreamHandler(sys.stdout)
])

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
        self.echo_feedback_slider = tk.Scale(master, from_=0, to=2, orient=tk.HORIZONTAL, length=400, resolution=0.01)
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
        logging.debug(f"Updated Volume: {self.volume_var[0]}")

    def update_echo_delay(self, event=None):
        with self.echo_delay_lock:
            self.delay_var[0] = self.current_echo_delay.get()
        logging.debug(f"Updated Echo Delay: {self.delay_var[0]}")

    def update_echo_feedback(self, event=None):
        with self.echo_feedback_lock:
            self.feedback_var[0] = self.current_echo_feedback.get()
        logging.debug(f"Updated Echo Feedback: {self.feedback_var[0]}")

def create_slider_app(volume_lock, volume_var, echo_delay_lock, delay_var, echo_feedback_lock, feedback_var):
    root = tk.Tk()
    app = SliderApp(root, volume_lock, volume_var, echo_delay_lock, delay_var, echo_feedback_lock, feedback_var)
    return app, root

def apply_effects(audio_data, sample_rate, volume, delay, feedback):
    echo_delay_samples = int(sample_rate * delay)
    feedback = max(0.0, min(feedback, 1.0))  # Ensure feedback is between 0 and 1
    logging.debug(f"Applying echo with delay={delay}s ({echo_delay_samples} samples) and feedback={feedback}, volume={volume}")

    data = audio_data * volume
    if echo_delay_samples > 0:
        buffer = np.zeros_like(data)
        for i in range(echo_delay_samples, len(data)):
            buffer[i] = data[i-echo_delay_samples] * feedback
            data[i] += buffer[i]
            data[i] = np.clip(data[i], -1.0, 1.0)
        logging.debug(f"Data after applying echo: {data[0:min(50, len(data))]}")

    data = np.clip(data, -1.0, 1.0)
    return data.astype(np.float32)


def callback(outdata, frames, time, status, volume_lock, volume, echo_delay_lock, delay_var, echo_feedback_lock, feedback_var):
    assert frames == BLOCK_SIZE

    if status.output_underflow:
        logging.error('Output underflow: increase blocksize?')
        raise sd.CallbackAbort
    assert not status

    try:
        data = q.get_nowait()
    except queue.Empty:
        logging.error('Buffer is empty: increase buffersize or decrease blocksize?')
        raise sd.CallbackAbort

    with volume_lock:
        current_volume = volume[0]
    with echo_delay_lock:
        current_delay = delay_var[0]
    with echo_feedback_lock:
        current_feedback = feedback_var[0]

    logging.debug(f"Callback processing with volume={current_volume}, delay={current_delay}, feedback={current_feedback}")

    try:
        data = apply_effects(data, samplerate, current_volume, current_delay, current_feedback)
    except Exception as e:
        logging.error(f"Error applying effects: {e}")
        logging.error(traceback.format_exc())
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
                wf.rewind()  # Rewind to start if file exhausted
                continue
            yield np.frombuffer(data, dtype=dtype).reshape(-1, channels).astype(np.float32) / 32768.0

def file_reader_thread(wave_gen, timeout):
    try:
        while True:
            data = next(wave_gen, None)
            if data is None:
                break
            q.put(data, timeout=timeout)
    except Exception as e:
        logging.error(f"Exception in file_reader_thread: {e}")
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
parser.add_argument('-b', '--blocksize', type=int, default=4096, help='block size (default: %(default)s)')
parser.add_argument('-q', '--buffersize', type=int, default=60, help='number of blocks used for buffering (default: %(default)s)')
args = parser.parse_args(remaining)
if args.blocksize == 0:
    parser.error('blocksize must not be zero')
if args.buffersize < 1:
    parser.error('buffersize must be at least 1')

BLOCK_SIZE = 8192  # args.blocksize
BUFFER_SIZE = 100  # args.buffersize

q = queue.Queue(maxsize=BUFFER_SIZE)
event = threading.Event()

volume_lock = threading.Lock()
volume = [0.5]
echo_delay_lock = threading.Lock()
delay_var = [0.3]
echo_feedback_lock = threading.Lock()
feedback_var = [0.5]

def main():
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
                                    echo_delay_lock=echo_delay_lock, delay_var=delay_var,
                                    echo_feedback_lock=echo_feedback_lock, feedback_var=feedback_var),
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
        parser.exit('\nInterrupted by user')
    except queue.Full:
        parser.exit(1)
    except Exception as e:
        logging.error('%s: %s', type(e).__name__, e)
        traceback.print_exc()
        parser.exit(type(e).__name__ + ': ' + str(e))

if __name__ == '__main__':
    main()