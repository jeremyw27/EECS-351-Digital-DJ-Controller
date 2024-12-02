import os
import wave
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
from pydub.playback import play

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
            if spf.getnchannels() == 2:
                print("Only mono files are supported for plotting.")
                return
            
            signal = spf.readframes(-1)
            signal = np.frombuffer(signal, dtype=np.int16)
            
            fs = spf.getframerate()
            time_axis = np.linspace(0, len(signal) / fs, num=len(signal))
            
            plot_waveform(time_axis, signal)
    except Exception as error:
        print(f"An error occurred during WAV file processing: {error}")

def plot_waveform(time_axis, signal):
    plt.figure(figsize=(10, 4))
    plt.title("Audio Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.plot(time_axis, signal)
    plt.grid()
    plt.show()

def add_echo_with_convolution(file_path, delay=0.3, feedback=0.5, output_path="output_with_echo.wav"):
    try:
        with wave.open(file_path, "r") as spf:
            fs = spf.getframerate()
            signal = spf.readframes(-1)
            signal = np.frombuffer(signal, dtype=np.int16)
        
        delay_samples = int(delay * fs)
        impulse_response = np.zeros(delay_samples + 1)
        impulse_response[0] = 1 #original sound
        impulse_response[delay_samples] = feedback
        
        echoed_signal = np.convolve(signal, impulse_response, mode='full')
        
        echoed_signal = np.clip(echoed_signal, -32768, 32767).astype(np.int16)
        
        with wave.open(output_path, "w") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(fs)
            output.writeframes(echoed_signal.tobytes())
        
        print(f"Echo effect applied using convolution and saved to {output_path}")
    except Exception as error:
        print(f"An error occurred during convolution-based echo processing: {error}")

def play_audio(file_path):
    try:
        audio = AudioSegment.from_file(file_path)
        play(audio)
    except Exception as error:
        print(f"An error occurred during audio playback: {error}")

if __name__ == "__main__":
    FILE_PATH = r"C:\Users\allyb\OneDrive\Desktop\eecs351final\bird.wav"
    OUTPUT_PATH = r"C:\Users\allyb\OneDrive\Desktop\eecs351final\bird_with_echo.wav"

    if check_file_exists(FILE_PATH):
        process_wave_file(FILE_PATH)
        
        print("Playing original audio...")
        play_audio(FILE_PATH)
        
        print("Adding echo effect using convolution...")
        add_echo_with_convolution(FILE_PATH, delay=0.3, feedback=0.5, output_path=OUTPUT_PATH)
        
        print("Playing audio with echo...")
        play_audio(OUTPUT_PATH)