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

# def convert_wav_to_mono(input_file, output_file):
#     try:
#         audio = AudioSegment.from_wav(input_file)
#         if audio.channels == 2:
#             audio = audio.set_channels(1)  # Convert to mono

#         audio.export(output_file, format="wav")

#     except Exception as error:
#         print(f"An error occurred while converting the WAV file: {error}")

def play_looping_audio(file_path, chunk_size, init_stop, cutoff_slider_values, loop_slider_values):
    try:
        mono_file_path = os.path.splitext(file_path)[0] + "_mono.wav"
        #convert_wav_to_mono(file_path, mono_file_path)
        if not os.path.exists(mono_file_path):
            window.write_event_value("-ERROR", f"File does not exist: {mono_file_path}")
            return

        with wave.open(mono_file_path, "rb") as waveform:
            channels = waveform.getnchannels()
            sample_width = waveform.getsampwidth()
            sample_rate = waveform.getframerate()

            if channels != 1:
                window.write_event_value("-ERROR", "Only mono audio files allowed")
                return

            playAudio = pyaudio.PyAudio()
            AudioStream = playAudio.open(format=playAudio.get_format_from_width(sample_width), channels=channels, 
                                         rate=sample_rate, output=True)

            while not init_stop.is_set():
                # Continuously fetch current slider values
                start_pos = int(loop_slider_values["-LOOP-START-"] * sample_rate / 1000)
                end_pos = int(loop_slider_values["-LOOP-END-"] * sample_rate / 1000)

                # Ensure start is less than end
                if start_pos >= end_pos:
                    continue

                waveform.setpos(start_pos)
                while not init_stop.is_set():
                    current_pos = waveform.tell()
                    if current_pos >= end_pos:
                        waveform.setpos(start_pos)

                    data = waveform.readframes(chunk_size)
                    if not data:
                        break

                    # Apply low-pass filter
                    cutoff = cutoff_slider_values["-CUTOFF-"]
                    audio_data = np.frombuffer(data, dtype=np.int16)
                    filtered_data = low_pass(audio_data, cutoff, sample_rate)
                    filtered_bytes = np.int16(filtered_data).tobytes()
                    AudioStream.write(filtered_bytes)

                    # Dynamically update loop bounds in real time
                    start_pos = int(loop_slider_values["-LOOP-START-"] * sample_rate / 1000)
                    end_pos = int(loop_slider_values["-LOOP-END-"] * sample_rate / 1000)

                    # If bounds change during playback, will update on the next loop
                    if current_pos < start_pos or current_pos >= end_pos:
                        break

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
    [simpg.Text("Loop Start (ms)", size=(15, 1)), simpg.Slider(range=(0, 60000), resolution=100, orientation="h", size=(30, 15), default_value=0, key="-LOOP-START-")],
    [simpg.Text("Loop End (ms)", size=(15, 1)), simpg.Slider(range=(0, 60000), resolution=100, orientation="h", size=(30, 15), default_value=10000, key="-LOOP-END-")],
    [simpg.Button("PLAY", key="-PLAY-"), simpg.Button("STOP", key="-STOP-"), simpg.Exit()],
    [simpg.Text("", size=(50, 1), key="-ERROR-", text_color="red")]
]

# GUI Window
window = simpg.Window("Real-Time Looping Effect", layout)
play_thread = None
stop_event = Event()

cutoff_slider_values = {"-CUTOFF-": 5000}
loop_slider_values = {"-LOOP-START-": 0, "-LOOP-END-": 10000}

while True:
    event, values = window.read(timeout=100)

    if event in (simpg.WINDOW_CLOSED, "EXIT"):
        stop_event.set()
        if play_thread and play_thread.is_alive():
            play_thread.join()
        break

    if event == "-ERROR":
        window["-ERROR-"].update(values[event])

    # Update real-time slider values
    cutoff_slider_values["-CUTOFF-"] = values["-CUTOFF-"]
    loop_slider_values["-LOOP-START-"] = values["-LOOP-START-"]
    loop_slider_values["-LOOP-END-"] = values["-LOOP-END-"]

    if event == "-PLAY-":
        file_path = values["-FILE-"]
        if not file_path:
            simpg.popup_error("Please select an input audio file.")
            continue

        try:
            chunk_size = int(values["-CHUNK-"])
        except ValueError:
            simpg.popup_error("Invalid chunk size. Please enter nonzero integer input.")
            continue

        stop_event.clear()
        play_thread = Thread(target=play_looping_audio, args=(file_path, chunk_size, stop_event, 
                                                              cutoff_slider_values, loop_slider_values))
        play_thread.start()

    if event == "-STOP-":
        stop_event.set()
        if play_thread and play_thread.is_alive():
            play_thread.join()
