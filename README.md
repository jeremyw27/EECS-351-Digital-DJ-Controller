# EECS-351-Digital-DJ-Controller Team 14 Final Project
Our Digital DJ Controller project combines digital signal processing (DSP) and various audio libraries in Python to create an interactive, real-time audio experience with effects such as echo, pitch modulation, filtering, and looping. Unlike traditional systems that work with pre-recorded audio, our project emphasizes live audio manipulation, allowing users to alter sound instantaneously. By leveraging DSP techniques, we transform audio signals in real time, giving users creative control over every beat, tone, and loop. With Python libraries such as Pyo, Pydub, and Librosa, we can produce high-quality, low-latency effects that bring a professional DJ experience to any setting. Explore our project details, code, and live demos on our GitHub page for a deep dive into the technical aspects of our digital audio transformations. 

# Description of files:

- echo2.py: This is a working version of the echo function with no real-time manipulation. 
              A path to the wav file that the user wishes to manipulate is required in the script,
             and the user is able to change the echo's delay, feedback, and number of repititions.

- echo3.py: This is a working version of the echo function WITH real-time audio manipulation.
               When run, a GUI appears with a button to allow the user to browse through their files
               to choose the one they want. There are also two sliders for echo delay and feedback, and when 
               changed, the audio changes based on the new parameters. The audio loops while the user
               manipulates the audio.                    

- loopingrealtime.py: This is the looping function WITH real-time audio manipulation. This script processes
                 audio files by applying progressively filtered segments using a low-pass filter
                 and provides waveforms and spectrograms of the original and filtered signals. 
  
- livefiltering.py:

- phase_vocoder.py: This is the function just to apply pitch modulation to an audio file. It takes in a WAV file as input and outputs a WAV file with the applied effect. It also outputs plots of the original signal waveform as well as the pitch modulated effect.

- pitchmod_realtime.py: This is a modified version of the phase_vocoder.py file and includes the GUI. The GUI has sliders to adjust the modulation frequency as well as the modulation depth for pitch modulation. It takes in an audio file as the input and enables real-time pitch modulation once it is run.
