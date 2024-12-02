import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf


def phase_vocoder(audio_file, mod_freq, mod_depth):
    # load audio file
    y, sr = librosa.load(audio_file, sr=None) # signal, sample rate

    # STFT
    n_fft = 2048 # length of windowed signal after padding with zeros
    hop_length = 512 # step size between windows
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length) # complex-valued matrix D
    magnitude, phase = np.abs(D), np.angle(D)

    # apply pitch modulation (vibrato), modulate the phase with a sinusoidal wave
    modulation_frequency = mod_freq # Hz (vibrato rate)
    modulation_depth = mod_depth  # depth in semitones
    t = np.arange(D.shape[1]) / sr * hop_length
    phase_modulation = modulation_depth * np.sin(2 * np.pi * modulation_frequency * t)

    # adjust phase and magnitude: combine phase and magnitude with newly modulated phase
    phase_modulated = phase + phase_modulation
    D_modulated = magnitude * np.exp(1j * phase_modulated)

    # ISTFT
    y_modulated = librosa.istft(D_modulated, hop_length=hop_length)

    # saving output to a new WAV file
    sf.write("modulated_output.wav", y_modulated, sr)

    # plotting audio
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(y, sr=sr)
    plt.title("Original Audio")
    plt.xlabel("Time (s)")
    plt.ylabel("Audio Amplitude")

    plt.subplot(2, 1, 2)
    librosa.display.waveshow(y_modulated, sr=sr)
    plt.title("Pitch Modulated Audio")
    plt.xlabel("Time (s)")
    plt.ylabel("Audio Amplitude")

    plt.tight_layout()
    plt.show()

phase_vocoder("c_note.wav", 1, 24)