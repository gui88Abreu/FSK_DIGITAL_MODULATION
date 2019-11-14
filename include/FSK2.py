#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 22:33:41 2019

@author: guilherme
"""

#importing libraries
import matplotlib.pyplot as plt
import numpy as np

def FSK2_demodulation(data, Fs, plot = False, n_samples = 0, baudRate = 20):
    
    F1=800
    F2=1200
    t_wave=np.arange(0,1/baudRate,1/Fs)
    wave1=np.cos(2*np.pi*F1*t_wave)
    wave2=np.cos(2*np.pi*F2*t_wave)
    
    end_bit = int((n_samples/baudRate)*Fs)
    
    if end_bit == 0:
        end_bit = len(data)
    
    # get data
    data  = data[:end_bit]
    t     = np.arange(0,len(data)/Fs,1/Fs)
    
    if plot:
        plt.figure(figsize=(32,32))
        plt.plot(t,data)
        plt.title('Sinal recebido')
        plt.show()
    
    # aplly matched filter
    matched1 = np.convolve(data,np.flip(wave1))
    matched2 = np.convolve(data,np.flip(wave2))
    
    # filter sampling of each bit
    step=int(Fs/baudRate)
    
    y1_samples = matched1[step::step]
    y2_samples = matched2[step::step]
    t_samples  = np.arange(step/Fs,t[-1]+step/Fs,step/Fs)
    
    if plot:
        plt.figure(figsize=(16,16))
        plt.plot(t,matched1[:len(t)],'b')
        plt.title('Saida e Amostragem do Filtro Casado1')
        plt.plot(t_samples,y1_samples,'or')
        plt.show()
        
        plt.figure(figsize=(16,16))
        plt.plot(t,matched2[:len(t)],'b')
        plt.title('Saida e Amostragem do Filtro Casado2')
        plt.plot(t_samples,y2_samples,'or')
        plt.show()
    
    Tb = (1/F1)*Fs/2 + 1
    matched1_abs = np.abs(matched1)/matched1.max()
    displ = [(np.append(np.zeros(i), matched1_abs))[:-i] for i in range(1,int(Tb))]
    matched1d = sum(displ)
    y1 = matched1_abs + matched1d
    
    Tb = (1/F2)*Fs/2 + 1
    matched2_abs = np.abs(matched2)/matched2.max()
    displ = [(np.append(np.zeros(i), matched2_abs))[:-i] for i in range(1,int(Tb))]
    matched2d = sum(displ)
    y2 = matched2_abs + matched2d
    
    y1_samples = y1[step::step]
    y2_samples = y2[step::step]
    t_samples  = np.arange(step/Fs,t[-1]+step/Fs,step/Fs)
    
    if plot:
        plt.figure(figsize=(16,16))
        plt.plot(t,y1[:len(t)],'b')
        plt.plot(t_samples,y1_samples,'or')
        plt.title('Deteccao e Amostragem da Envoltoria 1')
        plt.show()
        
        plt.figure(figsize=(16,16))
        plt.plot(t,y2[:len(t)],'b')
        plt.plot(t_samples,y2_samples,'or') 
        plt.title('Deteccao e Amostragem da Envoltoria 2')
        plt.show()
    
    # load header
    header_file = open("header.txt", 'r')
    h = header_file.readline()
    header_file.close()
    h = h.replace('[','').replace(']','')
    header = np.asarray([int(x) for x in h.split(',')])
    
    # make a decision
    output = y2_samples - y1_samples
    output[output > 0] = 1
    output[output < 0] = 0
    
    # get the optimized delta
    delta = np.argmax(np.correlate(output, header))
    
    msg_bits = (output.astype("uint8"))[delta+len(header):]
    
    return msg_bits