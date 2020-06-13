# -*- coding: utf-8 -*-
"""
Created on Fri May 22 18:45:27 2020

@author: Ted
"""

import librosa
from scipy import signal
import numpy as np

winlens = .064; # window length in seconds
fs = 16000; # frequency downsampled to
olap = 0.250; # percentage segment overlap. 0 <= olap <= 1

class Whisper():
    def __init__(self,lpcorder):
        self.lpcorder = lpcorder
    
    def lpc(self,signal, order):
        x = signal
        p = order
        autocorr = np.correlate(x,x,mode='full')
        r = autocorr[len(x)-1:len(x)+p]
    
        a = np.zeros(p+1)
        k = np.zeros(p)
        a[0] = 1
        a[1] = -r[1] / r[0]
        k[0] = a[1]
        E = r[0] + r[1] * a[1]
        for q in range(1,p):
            k[q] = -np.sum(a[0:q+1] * r[q+1:0:-1]) / E
            U = a[0:q+2]
            V = U[::-1]
            a[0:q+2] = U + k[q] * V
            E *= 1-k[q] * k[q]
        return a, k
    
    def lpcerr(self,winseq,order):
        # Function lpcerr() uses an input data segment "winseq", which is an 
        # appropriately windowed using a function such as triang(), and an LPC
        # order "ord".
        #
        # [pred, a] = lpcerr(winseq,ord)
        #
        # a:    LPC coefficients
        # pred: Prediction error sequence from all zero filter given by
        #       coefficients in a.
        #
        a,er = self.lpc(winseq,order); # LPC of individual segment
        pred = signal.lfilter(a, 1, winseq); # First prediction error all zero filter
        return pred, a
    
    def periost3(self,pred):
        # Takes an input vector representing a speech signal segment and determines
        # if the signal has any peridicities
        #
        #   str = periost3(pred)
        #
        # str:  output string, either contains 'periodic' or 'aperiodic'
        # pred: input speech segment vector
        #
         
        # This will suppress warning that findpeaks finds no peaks in a given
        # segments
        #warning('off', 'all');
         
        # Compute normalized autocorrelation
        ac = np.correlate(pred,pred,"full")
         
        # Search for peaks in positive axis, ignoring peak at zero lags
        temp = ac[round(len(ac)/2):]
        pks = signal.find_peaks(temp, height=0.25);
         
        # output aperiodic if no peaks found greater than .25 and periodic if peak 
        # is found
        if (len(pks)==0):
            strprd = 'aperiodic';
        else:
            strprd = 'periodic';
        return strprd
    
    
    def lpcrecon(self,pred, a, strprd):
        # Computes the reconstructed signal with random noise in place of
        # prediction sequence in order to simulate whispered speech over a given
        # segment. Filters simulated noise array using all-pole filter and LPC
        # coefficients for the filter. Noise array power is normalized to power of input
        # sequence.
        #
        #   recon = lpcrecon(pred, a, str)
        #
        # recon:    reconstructed signal segment
        # pred:     prediction error sequence
        # a:        LPC coefficients
        # str:      string variable denoting whether segment periodic or not
        #
        if(strprd=='periodic'):
            wn = np.random.randn(len(pred))
            wn = (wn-np.mean(wn))/(np.std(wn)/np.std(pred))
            recon = signal.lfilter([1.0], a, wn)
        elif(strprd=='aperiodic'):
            recon = signal.lfilter([1.0], a, pred)
            
        return recon
    
    def whisper_main(self,filename): 
        # Get Waveform
        y, fsp = librosa.core.load(filename, sr=fs)
        ylen = len(y)
        # Uncomment the following lines if necessary for processing input signal
         
        # Transpose to row vector if necessary
        # [ro, co] = size(y);
        # if co < 20
        #     y = y';
        # end
         
        # First iteration of loop outside of loop to set up the variables. Windows
        # first segment and then computes LPC coeffs and reconstructs signal
         
        win=list(range(int(winlens*fs))) #samples in first window
        pl=y[win[0]:win[-1]+1]; # first window of sound wave
        lpl=len(pl); # length first window of sound wave
        hanwin = np.hanning(lpl); # hanning window
         
        tri = hanwin*pl; # hanning window applied to sound segment
         
        [pred, a] = self.lpcerr(tri,self.lpcorder); #compute lpc coeffs and prediction sequence
         
        # Determine if segment is periodic or not
        strprd = self.periost3(pred);
            
        recon = self.lpcrecon(pred, a, strprd); #Reconstruction filter with white noise 
         
        # Initialize vector to hold final output signal
        triwin = np.concatenate((recon, np.zeros(ylen-int(winlens*fs)))); 
         
        # Initialize window segments for deconstruction, reconstruction loop
        k1 = int(winlens*olap*fs);  
        k2 = k1 + int(winlens*fs); 
         
        #While loop to compile windowed sound signal
        while k2 < ylen:
            win=list(range(k1,k2)); #samples in current window
            pl=y[win[0]:win[-1]+1]; # current window of sound wave
            
            tri = hanwin*pl; # triangle window applied to sound segment
            
            #compute lpc coeffs and prediction sequence
            [pred, a] = self.lpcerr(tri,self.lpcorder);
            
            # Determine if segment is periodic or not
            strprd = self.periost3(pred);
            
            # Reconstruction filter with either white noise or prediction error
            # depending on if segment periodic
            recon = self.lpcrecon(pred, a, strprd);
            
            # Increment lower and upper window bounds
            k1 = k1 + int(winlens*olap*fs);
            k2 = k2 + int(winlens*olap*fs);
            
            # Reconstruct signal
            triwin[win[0]:win[-1]+1] = triwin[win[0]:win[-1]+1] + recon;
        b,a = signal.butter(7, np.array((500, 6000))/(fs/2), btype='bandpass')
        out = signal.lfilter(b,a,triwin);
        return out

if __name__ == '__main__':
    filename_in, filename_out = "test.wav", "test_out.wav"
    lpcorder = 28
    whisper = Whisper(lpcorder)
    out = whisper.whisper_main(filename_in)
    librosa.output.write_wav(filename_out,out,fs)
