import os
import sys
import wave
import numpy as np
import matplotlib.pyplot as plt

from pydub import AudioSegment
from pydub.playback import play


def check_file_exists(file_path):
    #Check if the given file exists
    if os.path.exists(file_path):
        print("File found!")
        return True
    else:
        print("File not found!")
        return False


def process_wave_file(file_path):
    #Extract and process audio data from a WAV file
    try:
        with wave.open(file_path, "r") as spf:
            if spf.getnchannels() == 2:
                print("Only mono files are supported.")
                return
            
            # Extract Raw Audio from Wav File
            signal = spf.readframes(-1)
            signal = np.frombuffer(signal, dtype=np.int16)

            # Get sample rate and time axis
            fs = spf.getframerate()
            time_axis = np.linspace(0, len(signal) / fs, num=len(signal))
            
            # Plot the waveform
            plot_waveform(time_axis, signal)
    except Exception as error:
        print(f"An error occurred during WAV file processing: {error}")


def plot_waveform(time_axis, signal):
    #Plot the waveform of the audio signal
    plt.figure(figsize=(10, 4))
    plt.title("Audio Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.plot(time_axis, signal)
    plt.grid()
    plt.show()

def play_audio(file_path, repeat=3):
    #Play the given audio file
    try:
        audio = AudioSegment.from_file(file_path)
        play(audio * repeat)
    except Exception as error:
        print(f"An error occurred during audio playback: {error}")


# Main script
if __name__ == "__main__":
    FILE_PATH = r"C:\Users\tyler\Downloads\You_like_that.wav"

    if check_file_exists(FILE_PATH):
        play_audio(FILE_PATH, repeat=2)
        process_wave_file(FILE_PATH)