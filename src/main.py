#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 22:33:41 2019

@author: guilherme
"""

#importing libraries
import numpy as np
import soundfile as sf
from include.audio import record_audio
from include.app_decoder import  app_decoder
from include.FSK4 import FSK4_demodulation
from include.FSK2 import FSK2_demodulation

FSK = 4

if FSK == 4:
    wav_name = "audios/recv_audio-4FSK.wav"
elif FSK == 2:
    wav_name = "audios/recv_audio-2FSK.wav"
else:
    quit(str(FSK)+"-FSK isn't valid!")

# record audio
record_audio(filename = wav_name, channels = 1, seconds = 40, fs = 24000)
data, Fs = sf.read(wav_name)



if FSK == 2:
    msg_bits = FSK2_demodulation(data, Fs)
    
    #catch msg size
    msg_size_bits = np.array2string(msg_bits[15:31]).replace("[",'').replace(']','').replace(' ','')
    msg_size = int(msg_size_bits,2)
    
    print("Size of the Message:", msg_size, "bits")
    app_decoder(msg_bits)
    
if FSK == 4:
    msg_bits = FSK4_demodulation(data, Fs, plot = False, n_samples = 0)
    
    #catch msg size
    msg_size_bits = np.array2string(msg_bits[15:31]).replace("[",'').replace(']','').replace(' ','')
    msg_size = int(msg_size_bits,2)
    
    print("Size of the Message:", msg_size, "bits")
    app_decoder(msg_bits)