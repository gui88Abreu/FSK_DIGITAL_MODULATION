#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 22:33:41 2019

@author: guilherme
"""

#importing libraries
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from audio import record_audio

record_audio(filename = "../audios/recv_audio.wav", channels = 1, seconds = 5)
data, Fs = sf.read("../audios/teste_2fsk.wav")

baudRate=20
start_bit=40
start=int((start_bit/baudRate)*Fs)
data=data[start:start+int((10/baudRate)*Fs)]
t=np.arange(0,len(data)/Fs,1/Fs)
plt.figure(figsize=(32,32))
plt.plot(t,data)
plt.title('Sinal recebido')
plt.show()

F1=800
F2=1200

t_wave=np.arange(0,1/baudRate,1/Fs)
wave1=np.cos(2*np.pi*F1*t_wave + np.pi/4)
wave2=np.cos(2*np.pi*F2*t_wave + np.pi/4)

matched1 = np.convolve(data,np.flip(wave1))
plt.figure(figsize=(32,32))
plt.plot(t,matched1[:len(t)],'r')
plt.title('Saida do Filtro Casado1')
plt.show()

matched2 = np.convolve(data,np.flip(wave2))
plt.figure(figsize=(32,32))
plt.plot(t,matched2[:len(t)],'b')
plt.title('Saida do Filtro Casado2')
plt.show()

phase_error = np.pi + np.min(np.arctan2(matched1, matched2))
wave1=np.cos(2*np.pi*F1*t_wave + phase_error)
wave2=np.cos(2*np.pi*F2*t_wave + phase_error)

#Amostragem do filtro casado a cada período de bit
step=int(Fs/baudRate)

y1_samples = matched1[step::step]
y2_samples = matched2[step::step]
t_samples  = np.arange(step/Fs,t[-1]+step/Fs,step/Fs)

plt.figure(figsize=(16,16))
plt.plot(t_samples,y1_samples,'or')
plt.title('Amostragem Filtro Casado2')
plt.show()

plt.figure(figsize=(16,16))
plt.plot(t_samples,y2_samples,'ob')
plt.title('Amostragem Filtro Casado2')
plt.show()