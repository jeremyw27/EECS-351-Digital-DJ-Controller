import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import pyaudio
import tkinter as tk
from tkinter import Scale
import threading

# function to apply pitch modulation (modified from phase_vocoder.py)
def phase_vocoder(audio_data, sr, mod_freq, mod_depth):
    # STFT
    n_fft = 2048  # length of windowed signal after padding with zeros
    hop_length = 512  # step size between windows
    D = librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = np.abs(D), np.angle(D)

    # apply pitch modulation (vibrato), modulate the phase with a sinusoidal wave
    modulation_frequency = mod_freq  # Hz (vibrato rate)
    modulation_depth = mod_depth  # depth in semitones
    t = np.arange(D.shape[1]) / sr * hop_length
    phase_modulation = modulation_depth * np.sin(2 * np.pi * modulation_frequency * t)

    # adjust phase and reconstruct complex STFT: combine phase and magnitude with newly modulated phase
    phase_modulated = phase + phase_modulation
    D_modulated = magnitude * np.exp(1j * phase_modulated)

    # ISTFT
    y_modulated = librosa.istft(D_modulated, hop_length=hop_length)
    return y_modulated

# GUI class
class PitchModulationApp:
    def __init__(self, root, audio_file):
        self.root = root
        self.audio_file = audio_file
        self.root.title("Audio File Pitch Modulation")

        # modulation frequency knob
        self.freq_label = tk.Label(root, text="Modulation Frequency (Hz)")
        self.freq_label.pack()
        self.freq_knob = Scale(root, from_=1, to=20, resolution=1, orient=tk.HORIZONTAL)
        self.freq_knob.set(1)
        self.freq_knob.pack()

        # modulation depth knob
        self.depth_label = tk.Label(root, text="Modulation Depth (semitones)")
        self.depth_label.pack()
        self.depth_knob = Scale(root, from_=0, to=24, resolution=1, orient=tk.HORIZONTAL)
        self.depth_knob.set(0)
        self.depth_knob.pack()

        # start button
        self.start_button = tk.Button(root, text="Start Modulation", command=self.start_modulation)
        self.start_button.pack(pady=20)

        # stop button
        self.stop_button = tk.Button(root, text="Stop", command=self.stop_modulation)
        self.stop_button.pack()

        self.running = False

    def start_modulation(self):
        self.running = True
        self.audio_thread = threading.Thread(target=self.process_audio_file)
        self.audio_thread.start()

    def stop_modulation(self):
        self.running = False

    def process_audio_file(self):
        # load the audio file
        audio_data, sr = librosa.load(self.audio_file, sr=None)

        # set up audio output stream
        CHUNK = 1024
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paFloat32, channels=1, rate=sr, output=True)

        try:
            for i in range(0, len(audio_data), CHUNK):
                if not self.running:
                    break

                # extract current chunk
                chunk = audio_data[i:i+CHUNK]

                # padding for the last chunk
                if len(chunk) < CHUNK:  
                    chunk = np.pad(chunk, (0, CHUNK - len(chunk)))

                # sliders
                mod_freq = self.freq_knob.get()
                mod_depth = self.depth_knob.get()

                # use phase vocoder function for pitch modulation
                modulated_chunk = phase_vocoder(chunk, sr, mod_freq, mod_depth)

                # play modulated chunk
                stream.write(modulated_chunk.astype(np.float32).tobytes())
        finally:
            # stop stream
            stream.stop_stream()
            stream.close()
            p.terminate()

# run the GUI
if __name__ == "__main__":
    # choose WAV file
    audio_file = "timeless.wav"

    root = tk.Tk()
    app = PitchModulationApp(root, audio_file)
    root.mainloop()
