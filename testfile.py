from pyo import *
s = Server().boot()
s.start()

# play a sound file (aiff):
# sf = SfPlayer("fein.aiff", speed=1, loop=True).out()

# time.sleep(10)


# granulate audio buffer:
# snd = SndTable("fein.aiff")
# env = HannTable()
# pos = Phasor(freq=snd.getRate()*0.2, mul=snd.getSize())
# dur = Noise(mul=.001, add=.1)
# g = Granulator(snd, env, [1, 1.001], pos, dur, 32, mul=.1).out()


# generate melodies:
wav = SquareTable()
env = CosTable([(0,0), (100,1), (500,.3), (8191,0)])
met = Metro(.125, 12).play()
amp = TrigEnv(met, table=env, dur=1, mul=.1)
pit = TrigXnoiseMidi(met, dist='loopseg', x1=20, scale=1, mrange=(48,84))
out = Osc(table=wav, freq=pit, mul=amp).out()

time.sleep(10)
s.stop()
s.shutdown()


# ideas:
# make a function that takes in the output of the phasor or some other effect
# take that output and put it into an audio buffer (which will be played)