import os
import wave
import time
import numpy as np
import matplotlib.pyplot as plt
import PySimpleGUI as simpg
from pydub import AudioSegment
from pydub.playback import play
from scipy.signal import spectrogram
from scipy.signal import butter
from scipy.signal import lfilter
from threading import Thread

file_path = r'Jon_Vinyl_Chrome Hearts_OFFICIAL_VERSION.wav'

def check_file_exists(file_path):
    """Check if file exists."""
    if os.path.exists(file_path):
        print("File found")
        return True
    else:
        print("File not found, please try again")
        return False

def plot_waveforms(time_axis, inputsignal, filteredsignal, title="Original Signal vs. Filtered Signal"):
    """Plot the waveform of the audio signal side by side."""
    plt.figure(figsize=(10, 4))
    plt.rcParams['axes.facecolor'] = '#fff8ff'  # Light gray for the axes background
    plt.rcParams['figure.facecolor'] = '#f0f0f0'  # Light gray for the figure background

    # Plot the original signal
    plt.subplot(1, 2, 1)  # (rows, columns, index)
    plt.title("Original Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.plot(time_axis, inputsignal, label="Original Signal", color='navy')
    plt.rcParams['figure.facecolor'] = '#b3b3b3'  # Light gray for the figure background
    plt.legend()
    plt.grid()

    # Plot the filtered signal
    plt.subplot(1, 2, 2)  # (rows, columns, index)
    plt.title("Filtered Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.plot(time_axis, filteredsignal, label="Filtered Signal", color='purple')
    plt.legend()
    plt.grid()

    plt.suptitle(title)
    plt.subplots_adjust(top=0.90)  # Adjust the title to avoid overlap with subplots
    plt.show()
    plt.close()

def plot_spectrogram(freqdom, timedom, Sxx, title="Spectogram"):
    """Plot the spectogram of the audio signal."""
    plt.figure(figsize=(10, 4))
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.pcolormesh(timedom, freqdom, 10 * np.log10(Sxx), shading='auto')
    plt.colorbar(label = "Intensity (dB)")
    plt.grid()
    plt.show()
    time.sleep(0.5)
    plt.close()

def convert_wav_to_mono(input_file, output_file):
    """Convert a stereo WAV file to mono."""
    try:
        # Load the WAV file
        audio = AudioSegment.from_wav(input_file)

        # Convert to mono (this averages the stereo channels to a single channel)
        mono_audio = audio.set_channels(1)

        # Export the mono audio to a new WAV file
        mono_audio.export(output_file, format="wav")
        print(f"Mono audio saved as {output_file}")

    except Exception as e:
        print(f"An error occurred while converting the WAV file: {e}")


def low_pass(audioinput, freqlimit, fs, filterorder=6):
    nyquist_freq = 0.5*fs
    lowlimit = freqlimit / nyquist_freq
    num, den = butter(filterorder, lowlimit, btype='low', analog=False)
    return lfilter(num, den, audioinput)

def low_pass_filter(audioinput, repeat_no, fs, start, end):
    mod_audio = np.linspace(start, end, repeat_no, endpoint=False)
    looped_array = []

    for freqlimit in mod_audio:
        new_signal = low_pass(audioinput, freqlimit, fs)
        looped_array.append(new_signal)
    
    return np.concatenate(looped_array)


def play_audio_woblocking(filtered_audio):
    """Play the specified audio segment on a loop."""
    try:
       # Play the filtered segment repeated the specified number of times
        print(f"Playing filtered audio segment")
        play(filtered_audio)

    except Exception as error:
        print(f"An error occurred during audio playback: {error}")


def process_wave_file(file_path, start_time, segment_duration, repeat_no, start_cutoff, end_cutoff, fs):
    """Extract and process audio data with progressively increasing filtering."""
    try:
        with wave.open(file_path, "r") as openfile:
            if openfile.getnchannels() == 2:
                print("Only mono files are supported.")
                return

            # Extract Raw Audio from WAV File
            original_signal = openfile.readframes(-1)  # Read all frames
            original_signal = np.frombuffer(original_signal, dtype=np.int16)

            # Create the segment based on user audio specs input
            start_sample = int(start_time * fs)
            end_sample = start_sample + int(segment_duration * fs)

            # Ensure we don't exceed the length of the file
            if end_sample > len(original_signal):
                end_sample = len(original_signal)

            segment = original_signal[start_sample:end_sample]

            # Generate progressively lower cutoff frequencies
            cutoff_freq = np.linspace(end_cutoff, start_cutoff, repeat_no)

            time_axis_segment = np.linspace(start_time, start_time + segment_duration, num=len(segment))

            for cutoff in cutoff_freq:
                filtered_segment = low_pass(segment, cutoff, fs)
                filtered_segment = np.int16(filtered_segment)  # Ensure correct data type

                filtered_audio = AudioSegment(
                    filtered_segment.tobytes(),
                    frame_rate=fs,
                    sample_width=filtered_segment.dtype.itemsize,
                    channels=1
                )
            
            # Play the filtered audio segment (one at a time)
                print(f"Playing segment with cutoff {cutoff} Hz...")
                play_audio_woblocking(filtered_audio)
                # Alternatively, if you want to play the filtered segment directly:
                # play(filtered_audio)

             # Plot original and filtered signal
            plot_waveforms(time_axis_segment, segment, filtered_segment, title="Original vs. Most Filtered Signal")

            f, t, Sxx = spectrogram(filtered_segment, fs)

            # Check for any zero values in Sxx to avoid log(0) errors
            Sxx[Sxx == 0] = np.finfo(float).eps  # Replace zeros with a small epsilon
            plot_spectrogram(f, t, Sxx, title="Spectrogram: Original Signal")

            # Play the combined audio
            # print("Playing progressively filtered audio...")
            # play(combined_audio)

    except Exception as error:
        print(f"An error occurred during WAV file processing: {error}")


# Main script
if __name__ == "__main__":
    if check_file_exists(file_path):
            # Convert the file to mono before processing
            output_file = 'Jon_Vinyl_Chrome Hearts_OFFICIAL_VERSION_mono.wav'
            convert_wav_to_mono(file_path, output_file)

            # Prompt user for segment details
            start_time = int(input("Enter the start time of the segment (in seconds): "))
            segment_duration = int(input("Enter the duration of the segment (in seconds): "))
            repeat_no = int(input("Enter the number of times to loop the segment: "))
            start_cutoff = float(input("Enter the start cutoff frequency (Hz): "))
            end_cutoff = float(input("Enter the end cutoff frequency (Hz): "))

            # Play the specified audio segment
             # Load sample rate from WAV file
            with wave.open(output_file, "r") as openfile:
                fs = openfile.getframerate()

            # play_audio_woblocking(output_file, start_time=start_time, segment_duration=segment_duration, repeat_no=repeat_no)
            process_wave_file(output_file, start_time=start_time, segment_duration=segment_duration, repeat_no=repeat_no, start_cutoff=start_cutoff, end_cutoff=end_cutoff, fs=fs)
