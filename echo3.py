import os
import wave
import numpy as np
import simpleaudio as sa
import tkinter as tk
from tkinter import filedialog
import threading
import time

def check_file_exists(file_path):
    if os.path.exists(file_path):
        print("File found!")
        return True
    else:
        print("File not found!")
        return False

def process_wave_file(file_path):
    try:
        with wave.open(file_path, "r") as spf:
            n_channels = spf.getnchannels()
            
            signal = spf.readframes(-1)
            signal = np.frombuffer(signal, dtype=np.int16)
            
            fs = spf.getframerate()
            if n_channels == 2:
                signal = signal.reshape(-1, 2)
                print("Stereo files are not supported for plotting. Proceeding with echo processing...")
            else:
                time_axis = np.linspace(0, len(signal) / fs, num=len(signal))
                #plot_waveform(time_axis, signal)
            
            return signal, fs, n_channels, spf.getsampwidth()
    except Exception as error:
        print(f"An error occurred during WAV file processing: {error}")


def add_echo_with_convolution(signal, fs, n_channels, sampwidth, delay=0.3, feedback=0.5):
    try:
        max_amplitude = np.max(np.abs(signal)) if n_channels == 1 else np.max(np.abs(signal[:, 0]))
        normalized_signal = signal / max_amplitude

        delay_samples = int(delay * fs)
        impulse_response = np.zeros(delay_samples + 1)
        impulse_response[0] = 1  # original
        impulse_response[delay_samples] = feedback  # echo

        if n_channels == 2:
            echoed_signal_left = np.convolve(normalized_signal[:, 0], impulse_response, mode='full')[:len(signal)]
            echoed_signal_right = np.convolve(normalized_signal[:, 1], impulse_response, mode='full')[:len(signal)]
            echoed_signal = np.vstack((echoed_signal_left, echoed_signal_right)).T
        else:
            echoed_signal = np.convolve(normalized_signal, impulse_response, mode='full')[:len(signal)]
        
        echoed_signal = echoed_signal * max_amplitude
        echoed_signal = np.clip(echoed_signal, -32768, 32767).astype(np.int16)

        return echoed_signal
        
    except Exception as error:
        print(f"An error occurred during convolution-based echo processing: {error}")
        return None

def play_audio_simpleaudio(signal, fs, n_channels, sampwidth):
    try:
        wave_obj = sa.WaveObject(signal.tobytes(), num_channels=n_channels, bytes_per_sample=sampwidth, sample_rate=fs)
        play_obj = wave_obj.play()
        play_obj.wait_done()  # Wait until the sound has finished playing
    except Exception as error:
        print(f"An error occurred during audio playback: {error}")

class EchoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Echo Effect GUI")
        
        self.delay_var = tk.DoubleVar(value=0.3)
        self.feedback_var = tk.DoubleVar(value=0.5)
        
        self.load_button = tk.Button(root, text="Load WAV File", command=self.load_file)
        self.load_button.pack(pady=10)
        
        self.delay_slider = tk.Scale(root, from_=0, to_=1, orient=tk.HORIZONTAL, label="Delay", resolution=0.01, variable=self.delay_var)
        self.delay_slider.pack(pady=10)
        
        self.feedback_slider = tk.Scale(root, from_=0, to_=1, orient=tk.HORIZONTAL, label="Feedback", resolution=0.01, variable=self.feedback_var)
        self.feedback_slider.pack(pady=10)
        
        self.file_path = None
        self.signal = None
        self.fs = None
        self.n_channels = None
        self.sampwidth = None

        self.play_audio_var = tk.BooleanVar(value=False)
        
        self.play_thread = threading.Thread(target=self.play_continuous)
        self.play_thread.start()
    
    def load_file(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if self.file_path:
            data = process_wave_file(self.file_path)
            if data:  # Ensure data is not None
                self.signal, self.fs, self.n_channels, self.sampwidth = data
                print("WAV file loaded. Playing...")
                self.play_audio_var.set(True)
                self.update_audio()
    
    def update_audio(self):
        if self.file_path and self.signal is not None:
            delay = self.delay_var.get()
            feedback = self.feedback_var.get()
            print(f"Updating echo with delay: {delay}, feedback: {feedback}")
    
    def play_continuous(self):
        while True:
            if self.play_audio_var.get() and self.signal is not None:
                delay = self.delay_var.get()
                feedback = self.feedback_var.get()
                echoed_signal = add_echo_with_convolution(self.signal, self.fs, self.n_channels, self.sampwidth, delay=delay, feedback=feedback)
                if echoed_signal is not None:
                    play_audio_simpleaudio(echoed_signal, self.fs, self.n_channels, self.sampwidth)
            time.sleep(0.1)

if __name__ == "__main__":
    root = tk.Tk()
    app = EchoApp(root)
    root.mainloop()