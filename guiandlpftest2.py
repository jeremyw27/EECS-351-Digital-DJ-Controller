from pydub import AudioSegment
import time
from pydub.playback import play
from pydub.effects import low_pass_filter
from pydub.utils import make_chunks
import slider  
import tkinter as tk

# process audio chunk by chunk based on cut off freq
def process_audio_chunk(lpfslider, root, sound_chunks, index=0, filtered_sound=AudioSegment.empty()): 
    if index < len(sound_chunks):
        start = time.process_time()
        
        # Update the GUI to get the current cut off freq
        root.update()
        lpf_value = lpfslider.get_current_value()

        # Apply lpf to current chunk
        sound_chunk = low_pass_filter(sound_chunks[index], lpf_value)
        filtered_sound += sound_chunk

        stop = time.process_time()
        #measure the time it takes to do this processing so we can subtract the that time from 500ms so it calls for the new value every 500ms 
        # which is the length of our chunks, so we can change our value in "real-time"
        elapsed_time = stop - start
        
        # Schedule next chunk processing
        
        delay = int(max(500 - (elapsed_time * 1000), 1))
        root.after(delay, process_audio_chunk, lpfslider, root, sound_chunks, index + 1, filtered_sound)
        
    else:
    
        
        
        play(filtered_sound)

# Function to start audio processing
def process_audio(slider_app, root):
    
    sound = AudioSegment.from_file("18- Angie (I've Been Lost) & Clara (the night is dark).wav", format="wav")
    sound = sound[:4000]  #just using first 4 second for right now

    # Split the audio into chunks of .5 seconds
    sound_chunks = make_chunks(sound, 500)
    print("Number of chunks: " + str(len(sound_chunks)))

    # Start processing the first chunk
    process_audio_chunk(lpfslider, root, sound_chunks)

if __name__ == "__main__":
    # initialize the slider 
    lpfslider, root = slider.create_slider_app()

    
    process_audio(lpfslider, root)

    
    root.mainloop()