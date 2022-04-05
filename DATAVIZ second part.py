#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['axes.grid'] = True

import seaborn as sns
sns.set_theme()

from IPython.display import Audio
from scipy.io.wavfile import write
from scipy import signal

import librosa
import librosa.display
import random
import swifter

in order to decrease the data size, without too much quality impact, downsampling from 44100Hz to 22050Hz has been implemented
it is an usual approach in the field
# # Downsampling to 22050 Hz
we use here the dataframe built in  "DATAVIZ first part"
# In[2]:


data = pd.read_csv ('dsd100_44100Hz_paths.csv', index_col = 0)

data.head()


# In[3]:


def AUDIO_loading_path (dataframe, track, channel):
    
    signal, __ = librosa.load (dataframe.loc[track, channel], sr = None)
    
    return signal


def AUDIO_loading_array (dataframe, track, channel):
    
    return dataframe.loc[track, channel]


# In[4]:


def AUdio (loading_mode, dataframe, track, channel, fs=44100 , random = True):

    if loading_mode == 'path':

        if random :
            channel = np.random.choice (['mix', 'vocals', 'drums', 'bass', 'other'])
            track = np.random.choice(np.arange(1,101))
        
        else:
            channel = channel
            track = track
            
        y = AUDIO_loading_path (dataframe, track, channel)
        
    if loading_mode == 'array':
       
        if random :
            channel = np.random.choice (['mix', 'vocals', 'drums', 'bass', 'other', 'music'])
            track = np.random.choice(np.arange(1,101))
            
        else:
            channel = channel
            track = track
        
        y = AUDIO_loading_array (dataframe, track, channel)

    print ('track n° {}, channel: {}'.format(track, channel))
    print()
    
    return Audio (y, rate = fs)


# In[5]:


#track to be chosen in [1,101]

track = random.choice(np.arange(1,101))

#channel to be chosen in ['mix', 'vocals', 'drums', 'bass', 'other'])

channel = 'mix'

# loading mode path or array

loading_mode = 'path'

AUdio (loading_mode, dataframe = data, track = track , channel = channel, fs = 44100, random = False)

# here the tracks are uploaded from their .wav file paths

to implement the downsampling, the data is first uploaded from the data_paths
# In[6]:


DATA = data.copy()

for channel in ['mix', 'vocals', 'drums', 'bass', 'other']:  
    
    DATA[channel] = DATA[channel].swifter.apply(lambda x: librosa.load (x, sr=None)[0])
    
DATA.head()

addition of an extra column
# In[7]:


DATA['music'] = DATA.drums + DATA.bass + DATA.other

DATA.head()


# In[8]:


#track to be chosen in [1,100]

#track = 

#channel to be chosen in ['mix', 'vocals', 'drums', 'bass', 'other', 'music'])

channel = 'mix'

# loading mode,  path or array

loading_mode = 'array'

AUdio (loading_mode, dataframe = DATA, track = track, channel = channel, fs = 44100, random = False)

# here the tracks are uploaded from their np.array data


# ### The simplest way to downsample from 44100Hz to 22050Hz is to keep every other point

# In[9]:


data_22050 = DATA.copy()

for channel in ['mix', 'vocals', 'drums', 'bass', 'other', 'music']:

    data_22050 [channel] = data_22050[channel].apply (lambda x  : (x[::2]))
                                                                               
data_22050.head()


# In[10]:


#track to be chosen in [1,100]

#track = 

#channel to be chosen in ['mix', 'vocals', 'drums', 'bass', 'other', 'music'])

#channel = 'mix'

# loading mode,  path or array

loading_mode = 'array'

AUdio (loading_mode, dataframe = data_22050, track = track, channel = channel, fs = 22050, random = False)

By carefully comparing with the 44100Hz signal, one can ear that downsampling to 22050Hz does not affect that much the sound quality

we may say, it is a little bit less pure and crystalline.....not that obvious

Let's have a quick view how the waveform looks like now
# ### Waveform plotting 

# In[11]:


def plot_audio (track , channel, random = True):
    
    if random == True:
        track = np.random.choice (np.arange(1,101) )
        channel = np.random.choice (['mix', 'vocals', 'drums', 'bass', 'other', 'music'])
        
    else:
        track = track
        channel = channel
    
    sig_44100 = AUDIO_loading_array (DATA, track, channel)
    sig_22050 = AUDIO_loading_array (data_22050, track, channel)
   
    
    x = np.arange (len(sig_44100))/44100
    x1 = np.arange (len(sig_22050))/22050
    
    T = len(sig_44100)/44100
    x_ticks = np.arange(0, 30 * (int(T/30)+1) , 30)
    
    t = int (T/60)
    tick_labels =[]
    
    for i in range (t+1):
        for j in range(2):
            a = str(i)
            if j==0:
                b = '00'
            else:
                b = '30'
        
            tick = a + ':' + b
            tick_labels.append(tick)
            
    while len(tick_labels) != len(x_ticks):
        tick_labels.pop(-1)
        
    fig, ax = plt.subplots (figsize = (23,7))

    ax.plot (x, sig_44100, color = 'black')
    ax.plot (x1, sig_22050, color = 'cyan')
    ax.set_xlabel ('Time', family = 'cursive', fontsize = 17)
    ax.set_xticks (x_ticks, tick_labels, family = 'cursive', fontsize = 13)
    ax.set_ylabel ('Amplitude', family = 'cursive', fontsize = 17) 
    ax.set_yticks (np.arange(-1,1.2, 0.2), np.round(np.arange(-1,1.2, 0.2),1) ,family = 'cursive', fontsize = 13)
    ax.legend (['44100Hz', '22050Hz'], fontsize = 13, loc = 'upper right')
    ax.set_title('Track {}   Channel {}'.format(track, channel[0].upper() + channel[1:]), family = 'cursive', fontsize = 19) 
    ax.set_ylim (-1,1)
    ax.set_xlim (-1, np.max (x)+1)       
    plt.show();


# In[12]:


#track to be chosen in [1,100]

#track = 

#channel to be chosen in  ['mix', 'vocals', 'drums', 'bass', 'other', 'music']

#channel = 

plot_audio (track, channel, random = True)


# ## Magnitude Spectra
Let's have a look to the magnitude spectra now
To begin with, let's start with librosa.amplitude_to_db right after the Fourrier Transform
# In[13]:


def mag_spectrum (track, channel, random = True):
    
    if random : 
        track = np.random.choice (np.arange(1,101))
        channel = np.random.choice (['mix', 'vocals', 'drums', 'bass', 'other', 'music'])
        
    else : 
        track = track
        channel = channel
        
        
    y = AUDIO_loading_array (DATA, track, channel)
    
    f,t,S = signal.stft (y, fs = 44100 , nperseg = 1024, noverlap = 768)
    S = np.abs(S)
    S = librosa.amplitude_to_db (S, ref = np.max)
    S = S + np.abs(np.min(S))
    
    z = AUDIO_loading_array (data_22050, track, channel)
    
    f,t,S1 = signal.stft (z, fs = 22050 , nperseg = 1024, noverlap = 768)
    S1 = np.abs(S1)
    S1 = librosa.amplitude_to_db (S1, ref = np.max)
    S1 = S1 + np.abs(np.min(S1))
    
    
    fig, ax = plt.subplots (2,1, figsize = (31,17), sharey = True)
    
    plt.rcParams['axes.grid'] = False

    img = librosa.display.specshow (S [:, :], x_axis='time', y_axis='hz', htk=True,
                         cmap='coolwarm', sr = 44100, hop_length = 256, ax = ax[0])
    
    cbar = fig.colorbar (img, ax = ax[0], format = '%2.0f dB')
    ax[0].set_ylabel ('Hz', fontsize = 23, family = 'cursive')
    ax[0].set_xlabel ('time ', fontsize = 23, family = 'cursive')
    ax[0].tick_params (axis='x', labelsize = 13)
    ax[0].tick_params (axis='y', labelsize = 13)
    ax[0].set_title('Track {}   Channel {}'.format(track, channel[0].upper() + channel[1:]), family = 'cursive', fontsize = 23) 
    cbar.ax.tick_params (labelsize  = 13)
    
    
    img1 = librosa.display.specshow (S1 [:, :], x_axis='time', y_axis='hz', htk=True,
                         cmap='coolwarm', sr = 22050, hop_length = 256, ax = ax[1])
    
    cbar = fig.colorbar (img1, ax = ax[1], format = '%2.0f dB')
    ax[1].set_ylabel ('Hz', fontsize = 23, family = 'cursive')
    ax[1].set_xlabel ('time ', fontsize = 23, family = 'cursive')
    ax[1].tick_params (axis='x', labelsize = 13)
    ax[1].tick_params (axis='y', labelsize = 13)
    cbar.ax.tick_params (labelsize  = 13)
    
    plt.show();


# In[14]:


#track to be chosen in [1,100]

#track = 

#channel to be chosen in ['mix', 'vocals', 'drums', 'bass', 'other', 'music']

#channel = 

mag_spectrum (track, channel, random = True)


# At this point, the 22050Hz data will be saved in .wav files and stored where ever you want
# It will be easier to go on

# In[15]:


for channel in ['mix', 'vocals', 'drums', 'bass', 'other', 'music']:
    
    for j in range (1,101) :
        
        x = data_22050.loc[j,'Name']
        
        if not os.path.exists ('/Users/francoistouchard/Music/DSD100_22050Hz/'+ x):
            
            os.makedirs ('/Users/francoistouchard/Music/DSD100_22050Hz/'+ x)
            
# Directories creation


# In[16]:


for channel in ['mix', 'vocals', 'drums', 'bass', 'other', 'music']:
    
    for j in range (1,101) :
        
        x = data_22050.loc[j,'Name']+ '/' + channel + '.wav'
        
        if not os.path.exists ('/Users/francoistouchard/Music/DSD100_22050Hz/'+ x ):
        
            write ('/Users/francoistouchard/Music/DSD100_22050Hz/' + x, 22050, data_22050.loc[j,channel])     
            
# Files writing


# ### New dataframe with 22050 Hz .wav file paths

# In[17]:


new = data_22050.copy()

for channel in ['mix', 'vocals', 'drums', 'bass', 'other', 'music']:
    
    for j in range (1,101) :
        
        new.loc[j,channel] = '/Users/francoistouchard/Music/DSD100_22050Hz/' + data_22050.loc[j,'Name'] + '/' + channel + '.wav'
        
new.to_csv('DSD100_22050_paths.csv')


# In[18]:


df = pd.read_csv ('DSD100_22050_paths.csv', index_col = 0)

df.head()


# In[19]:


# track to be chosen in [1,100]

#track = 

# channel to be chosen in ['mix', 'vocals', 'drums', 'bass', 'other', 'music'])

# channel =

# loading mode,  path or array

loading_mode = 'path'

AUdio (loading_mode = loading_mode , dataframe = df, track = track , channel = channel , fs = 22050, random = True)

