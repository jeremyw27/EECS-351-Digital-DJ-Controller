import os
import wave
import pyaudio
import numpy as np
import PySimpleGUI as simpg
from pydub import AudioSegment
from scipy.signal import butter, lfilter
from threading import Thread, Event

# Low-pass filter function
def low_pass(audioinput, freqlimit, fs, filterorder=6):
    nyquist_freq = 0.5 * fs
    lowlimit = freqlimit / nyquist_freq
    num, den = butter(filterorder, lowlimit, btype="low", analog=False)
    return lfilter(num, den, audioinput)

def convert_wav_to_mono(input_file, output_file):
    """Convert a stereo WAV file to mono."""
    try:
        # Load the WAV file
        audio = AudioSegment.from_wav(input_file)
        
        # Check if the file is stereo and convert to mono
        if audio.channels == 2:
            print("File is stereo. Converting to mono.")
            audio = audio.set_channels(1)  # Convert to mono
        else:
            print("File is already mono. No conversion needed.")
        
        # Export the mono audio to a new WAV file
        audio.export(output_file, format="wav")

    except Exception as error:
        print(f"An error occurred while converting the WAV file: {error}")

def play_looping_audio(file_path, chunk_size, init_stop, cutoff_slider_values, loop_start_ms, loop_end_ms):
    try:
        # Construct the correct path for the converted mono file
        mono_file_path = os.path.splitext(file_path)[0] + "_mono.wav" 

        convert_wav_to_mono(file_path, mono_file_path)
        
        if not os.path.exists(mono_file_path):
            window.write_event_value("-ERROR", f"File does not exist: {mono_file_path}")
            return

        with wave.open(mono_file_path, "rb") as waveform:
            # Validate file properties
            channels = waveform.getnchannels()
            sample_width = waveform.getsampwidth()
            sample_rate = waveform.getframerate()

            if channels != 1:
                window.write_event_value("-ERROR", "Only mono audio files allowed")
                return

            # Initialize audio output
            playAudio = pyaudio.PyAudio()
            AudioStream = playAudio.open(format=playAudio.get_format_from_width(sample_width),
                            channels=channels,
                            rate=sample_rate,
                            output=True)

            # Looping through the specified segment
            while not init_stop.is_set():
                waveform.setpos(int(loop_start_ms * sample_rate / 1000))  # Start from the loop start position in samples

                while not init_stop.is_set():
                    waveform_info = waveform.readframes(chunk_size)
                    if not waveform_info:
                        break

                    # Process audio
                    cutoff = cutoff_slider_values["-CUTOFF-"]
                    audio_data = np.frombuffer(waveform_info, dtype=np.int16)
                    filtered_data = low_pass(audio_data, cutoff, sample_rate)
                    filtered_bytes = np.int16(filtered_data).tobytes()

                    # Write to audio output
                    AudioStream.write(filtered_bytes)

                    # Check if we've reached the loop end position
                    if waveform.tell() >= int(loop_end_ms * sample_rate / 1000):
                        waveform.setpos(int(loop_start_ms * sample_rate / 1000))  # Restart loop segment

            AudioStream.stop_stream()
            AudioStream.close()
            playAudio.terminate()

    except Exception as e:
        simpg.popup_error(f"Error during playback: {e}")

# GUI Layout
layout = [
    [simpg.Text("Input Audio File"), simpg.Input(key="-FILE-", enable_events=True), simpg.FileBrowse(file_types=(("WAV Files", "*.wav"),))],
    [simpg.Text("Cutoff Frequency (Hz)", size=(17, 1)), simpg.Slider(range=(0, 30000), resolution=100, orientation="h", size=(30, 15), default_value=10000, key="-CUTOFF-")],
    [simpg.Text("Chunk Size"), simpg.Input(default_text="1024", key="-CHUNK-", size=(10, 1))],
    [simpg.Text("Loop Start (ms)"), simpg.Input(default_text="0", key="-LOOP-START-", size=(10, 1))],
    [simpg.Text("Loop End (ms)"), simpg.Input(default_text="10000", key="-LOOP-END-", size=(10, 1))],
    [simpg.Button("PLAY", key="-PLAY-"), simpg.Button("STOP", key="-STOP-"), simpg.Exit()],
    [simpg.Text("", size=(50, 1), key="-ERROR-", text_color="red")] 
]

# GUI Window
window = simpg.Window("Real-Time Looping Effect", layout)
play_thread = None
stop_event = Event()

# Slider value storage
cutoff_slider_values = {"-CUTOFF-": 5000}

while True:
    event, values = window.read(timeout=100)  # Update the GUI every 100 ms

    if event in (simpg.WINDOW_CLOSED, "EXIT"):
        stop_event.set()
        if play_thread and play_thread.is_alive():
            play_thread.join()
        break

    # Display errors if any
    if event == "-ERROR":
        window["-ERROR-"].update(values[event])  # Update the error message area

    # Update slider value for real-time processing
    cutoff_slider_values["-CUTOFF-"] = values["-CUTOFF-"]

    if event == "-PLAY-":
        file_path = values["-FILE-"]
        if not file_path:
            simpg.popup_error("Please select an input audio file.")
            continue

        # Validate chunk size
        try:
            chunk_size = int(values["-CHUNK-"])
        except ValueError:
            simpg.popup_error("Invalid chunk size. Please enter a valid integer.")
            continue

        # Get the loop start and end times (in milliseconds)
        try:
            loop_start_ms = int(values["-LOOP-START-"])
            loop_end_ms = int(values["-LOOP-END-"])
        except ValueError:
            simpg.popup_error("Invalid loop start/end time. Please enter valid integers.")
            continue
        
        stop_event.clear()
        play_thread = Thread(target=play_looping_audio, args=(file_path, chunk_size, stop_event, cutoff_slider_values, loop_start_ms, loop_end_ms))
        play_thread.start()

    if event == "-STOP-":
        stop_event.set()
        if play_thread and play_thread.is_alive():
            play_thread.join()
