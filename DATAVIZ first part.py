#!/usr/bin/env python
# coding: utf-8
The DSD100 dataset is available from https://sigsep.github.io/datasets/dsd100.html

first of all, the different subfolders (mixtures, sources, dev/test) have all been merged in one main DSD100 folder to simplify the coming process

then, the accompanying dsd100.xlsx file has been kept, giving interesting informations, such as the style of any track, that will be used to start with the dataviz hereafter
# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
sns.set_theme ()

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from IPython.display import Audio

import glob
import librosa
import swifter

# loading of the necessary packages/libraries for this first part


# In[2]:


VIZ = pd.read_excel ('dsd100.xlsx')

# use of the aforementionned dsd100 excel file

VIZ.info()
VIZ.head()

The style will be a useful information later on

All the tracks are first downloaded using glob and the overall_list is then used to pick up the titles
# In[4]:


overall_list = glob.glob('/Users/francoistouchard/Music/DSD100/*')

Titles = sorted (list(x.split('/')[5] for x in overall_list))
Titles [:13]

A dataframe is then built from Titles
# In[5]:


AUDIO = pd.DataFrame(Titles, columns = ['Name'])

AUDIO .info()
AUDIO .head()

a loop is then used to load the path off all the tracks and their channels
# In[6]:


for channel in ['mixture', 'vocals', 'drums', 'bass', 'other']:    
    
    AUDIO [channel] = glob.glob('/Users/francoistouchard/Music/DSD100/*/' + channel + '.wav')
    
AUDIO 

The name is then split into the numeric index part and Name keeping the group name and the track title
# In[7]:


AUDIO['Index'] = AUDIO.Name.apply (lambda x: x[:3]).astype (int)

AUDIO['Name'] = AUDIO.Name.apply (lambda x: x[6:])

AUDIO = AUDIO.rename({'mixture':'mix'},axis=1)

AUDIO.head()

At this stage, we merge AUDIO with VIZ on the 'Name' column to pick up the style information
We will see later on for the duration
# In[8]:


AUDIO = AUDIO.merge(VIZ, on = 'Name', how = 'left').drop('Duration', 1)

AUDIO.head()

the 'Index' column then comes as index of the sorted dataframe
# In[9]:


AUDIO = AUDIO.set_index ('Index')

AUDIO.head()

The columns are then ordered differently
# In[10]:


column_titles = ['Name', 'Style', 'mix', 'vocals','drums', 'bass', 'other']

AUDIO = AUDIO.reindex(columns = column_titles)

AUDIO.info()
AUDIO.head()

here the dataframe is saved to keep name, style and the path of all the tracks
# In[11]:


AUDIO.to_csv ('dsd100_paths.csv')

Nice to listen a little bit? Choose or let it be random...
# # LISTENING

# In[12]:


def AUDIO_loading (index, source):
    
    signal, __ = librosa.load (AUDIO.loc[index, source], sr = None)
    
    return signal


# In[14]:


def AUdio (track, channel, random = True):
    
    if random :
        channel = np.random.choice (['mix', 'vocals', 'drums', 'bass', 'other'])
        track = np.random.choice(np.arange(1,101))
        
    else:
        channel = channel
        track = track

    y = AUDIO_loading (track, channel)

    print ('track n° {}, channel: {}'.format(track, channel))
    print()
    
    return Audio (y, rate = 44100)


# In[15]:


# track is between 1 and 100, channel chosen among ['mix', 'vocals', 'drums', 'bass', 'other']

AUdio (999999991, 'anyone')

for the dataviz, all the data is loaded in the form of np.array 
# In[16]:


data = AUDIO.copy()

for channel in ['mix', 'vocals', 'drums', 'bass', 'other']:  
    
    data[channel] = data[channel].swifter.apply (lambda x: librosa.load (x, sr = None)[0])
    
data.head()

a music column, sum of all the non vocal channels is added
# In[17]:


data['music'] = data.drums + data.bass + data.other

data.head()

here it is possible, for those who are interested in, to save this dataframe in the parquet mode

it is quite efficient in term of process time and reduced memory of the final file

in the coming work, the data will be dowloaded using the track paths

it's up to everyone to choose according to its own preference
# In[18]:


def save_parquet (original_data, file, save = False):

    if save  == True:
        
        original_data.to_parquet (file)

def load_parquet (file, load = False):
    
    if load == True:
        
        load = pd.read_parquet (file)
        
        return load


# In[17]:


save_parquet (data, 'df_DSD100_44100Hz', False)


# data = load_parquet ('df_DSD100_44100Hz', False)
# 
# data.head()

# # AUDIO signal VIZ'
here all the original sampling frequencies are 44100Hz
# In[19]:


def plot_audio (track, colors = None, audios = None, labels = None, all_channels = True):
    
    mix =  AUDIO_loading (track, 'mix')
    vocals = AUDIO_loading (track, 'vocals')
    drums = AUDIO_loading (track, 'drums')
    bass = AUDIO_loading (track, 'bass')
    other = AUDIO_loading (track,'other')
    
    x = np.arange (len(mix))/44100
    
    T = len(mix)/44100
    x_ticks = np.arange(0, 30 * (int(T/30)+1) , 30)
    
    t = int (T/60)
    tick_labels =[]
    
    for i in range (t+1):
        for j in range(2):
            a = i
            b = j*30
            tick = str(a) +':' + str(b)
            tick_labels.append(tick)
            
    while len(tick_labels) != len(x_ticks):
        tick_labels.pop(-1)
    
    if all_channels == True:
        
        fig, ax = plt.subplots (figsize = (39,7))

        ax.plot (x, mix, color = 'blue')
        ax.plot (x, vocals, color = 'red')
        ax.plot (x, drums, color = 'black')
        ax.plot (x, bass, color = 'green')
        ax.plot (x, other, color = 'magenta')      
        ax.set_xlabel ('Time', fontsize = 23)
        ax.set_xticks (x_ticks, tick_labels, fontsize = 19)
        ax.set_ylabel ('Amplitude', fontsize = 23) 
        ax.set_yticks (np.arange(-1,1.2, 0.2), np.round(np.arange(-1,1.2, 0.2),1) ,fontsize = 19)
        ax.legend (['mix', 'vocals', 'drums', 'bass', 'other'], fontsize = 23, loc = 'upper right')
        ax.set_ylim (-1,1)
        ax.set_xlim (-1, np.max (x)+1)       
        
    else:
        n = len (audios)
        m = 0
    
        for z in range(n):
            m = np.max([m, len(audios[z])])
        
        x = (np.arange(m))/44100
    
        T = m / 44100
        x_ticks = np.arange(0, 30 * (int(T/30)+1) , 30)
    
        t = int (T/60)
        tick_labels =[]
    
        for i in range (t+1):
            for j in range(2):
                a = i
                b = j*30
                tick = str(a) +':' + str(b)
                tick_labels.append(tick)
            
        while len(tick_labels) != len(x_ticks):
            tick_labels.pop(-1)
    
        fig, ax = plt.subplots (figsize = (39,7))
        
        for z in range(n): 
            ax.plot ((np.arange(len(audios[z])))/44100, audios[z], c = colors [z])  
            
        ax.legend (labels, fontsize = 23, loc = 'upper right')
        ax.set_xlabel ('Time', fontsize = 23)
        ax.set_xticks (x_ticks, tick_labels, fontsize = 19)
        ax.set_ylabel ('Amplitude', fontsize = 23) 
        ax.set_yticks (np.arange(-1,1.2, 0.2), np.round(np.arange(-1,1.2, 0.2),1) ,fontsize = 19)
        ax.set_ylim (-1,1)
        ax.set_xlim (-1, np.max (x)+1) 
        plt.show();


# In[20]:


# overlay of all signals for a random value of index

track = np.random.choice(np.arange(1,101))

print ('track: {}'.format(track))
print ()

plot_audio (track)


# In[21]:


# For any other combination....

#source = np.random.choice (['mix', 'vocals', 'drums', 'bass', 'other'])

track = np.random.choice(np.arange(1,101))

print ('Track: {}'.format(track))
print ()


# In[22]:


audio1 = AUDIO_loading (track,'bass') + AUDIO_loading (track,'drums') + AUDIO_loading (track,'other') + AUDIO_loading (track,'vocals')

audio2 = AUDIO_loading (track,'mix')

plot_audio (track, ['black', 'white'], [audio1, audio2], ['music+vocals', 'mix'], all_channels = False)

we can see just above the perfect match between the mixture on one hand and the sum of the music and vocals on the other hand
this can be probably be checked on all the tracks
this gives a good idea of the data qualityTo get an idea of the distribution of the different sources in the mixture, a function just hereafter has been defined

a treshold is proposed to decide wether or not a channel is present at any given time
# In[23]:


def duration (signal, threshold): 
    
    return (np.where (abs(signal) > threshold, 1, 0).sum()) / (60*44100)


# In[24]:


# Random exemple 

threshold = 0

channel = np.random.choice (['mix', 'vocals', 'drums', 'bass', 'other'])
track = np.random.choice(np.arange(1,101))

signal = AUDIO_loading (track, channel)
mix_ = AUDIO_loading (track, 'mix')

print ('Track {} __\n'.format(track))
print ("{} duration (amplitude >{}) : {} min     mix duration (amplitude >{}) : {} min".format(channel, threshold, np.round(duration (signal, threshold), 2),
                                                                                               threshold, np.round(duration(mix_, threshold),2)))

print ("\n{} proportion vs mix : {}% ".format(channel, np.round (100 * duration (signal, threshold)/duration(mix_, threshold), 2)))

# treshold value has been assigned to 0 because the signals are not noisy
# moreover, the objective is to obtain approximate values here and a great accuracy is thus not that important

this has been applied to any track and channel of data to obtain durations and proportions against the mix
# In[25]:


threshold = 0

data['duration'] = data['mix'].swifter.apply (lambda x: len(x)/(60*44100))
                                                                                           
for channel in ['mix', 'vocals', 'drums', 'bass', 'other', 'music']:

    data[channel] = data[channel].swifter.apply (lambda x: duration (x, threshold))


# In[32]:


for channel in ['mix', 'vocals', 'drums', 'bass', 'other', 'music']:
    
    data [channel + ' ratio'] = (data[channel]/data['mix'])

data is saved here in csv format
# In[33]:


#data.to_csv ('DSD100_44100Hz_DATAVIZ.csv')
data = pd.read_csv ('DSD100_44100Hz_DATAVIZ.csv', index_col = 0)
data.round(3)


# In[34]:


data['vocals'].sum()/data['mix'].sum()

estimates vocals being around 74% of the mix duration for the 100 tracks (remembering here that threshold = 0 has been used)
# # DISTRIBUTION ANALYSES

# In[35]:


plt.figure (figsize = (13,17))
sns.countplot (data = data, y = 'Style')
plt.title ('Style countplot for the whole DSD100 dataset', family = 'cursive', fontsize = 17);


# In[36]:


print("the number of unique 'Style' is", len(np.unique(data['Style'][~(data.Style).isna()])))

the 100 tracks are well distributed over the 65 categories
# In[37]:


sns.displot (data = data, x = 'duration', kde=True, rug=True, hatch='/', bins = 20, height = 7)
plt.title ('Tracks duration distribution', family = 'cursive', fontsize = 17);


# In[38]:


fig, ax = plt.subplots (3, 2, figsize = (23,23), constrained_layout=True)

sns.histplot (data['vocals ratio'], ax = ax[0,0], kde=True, hatch='/', bins = 30)
ax[0,0].set_xlabel ('vocals vs mix', family='cursive', fontsize = 25)
ax[0,0].set_ylabel ('count', family='cursive', fontsize = 25)
ax[0,0].set_xlim ([np.min (data['vocals ratio'])-0.002, 1.002])
ax[0,0].tick_params (axis = 'both', labelsize = 15)

sns.histplot (data['drums ratio'], ax = ax[0,1], kde=True, hatch='/', bins = 30)
ax[0,1].set_xlabel ('drums vs mix', family='cursive', fontsize = 25)
ax[0,1].set_ylabel ('count', family='cursive', fontsize = 25)
ax[0,1].set_xlim ([np.min (data['drums ratio'])-0.002, 1.002])
ax[0,1].tick_params (axis = 'both', labelsize = 15)

sns.histplot (data['bass ratio'], ax = ax[1,0], kde=True, hatch='/', bins = 30)
ax[1,0].set_xlabel ('bass vs mix', family='cursive', fontsize = 25)
ax[1,0].set_ylabel ('count', family='cursive', fontsize = 25)
ax[1,0].set_xlim ([np.min (data['bass ratio'])-0.002, 1.002])
ax[1,0].tick_params (axis = 'both', labelsize = 15)

sns.histplot (data['other ratio'], ax = ax[1,1], kde=True, hatch='/', bins = 30)
ax[1,1].set_xlabel('other vs mix', family='cursive', fontsize = 25)
ax[1,1].set_ylabel('count', family='cursive', fontsize = 25)
ax[1,1].set_xlim ([np.min (data['other ratio'])-0.002, 1.002])
ax[1,1].tick_params (axis = 'both', labelsize = 15)

sns.histplot (data['music ratio'], ax = ax[2,0], kde=True, hatch='/', bins = 30)
ax[2,0].set_xlabel ('music vs mix', family='cursive', fontsize = 25)
ax[2,0].set_ylabel ('count', family='cursive', fontsize = 25)
ax[2,0].set_xlim ([np.min (data['music ratio'])-0.002, 1.002])
ax[2,0].tick_params (axis = 'both', labelsize = 15)

sns.histplot (data['mix']/data['duration'], ax = ax[2,1], kde=True, hatch='/', bins = 30)
ax[2,1].set_xlabel ('mix vs duration', family='cursive', fontsize = 25)
ax[2,1].set_ylabel ('count', family='cursive', fontsize = 25)
ax[2,1].set_xlim ([np.min (data['mix']/data['duration'])-0.002, 1.002])
ax[2,1].tick_params (axis = 'both', labelsize = 15)

fig.suptitle ('Channels distribution', family='cursive', fontsize = 25)
plt.show();

to end this first part, a summary table of durations and ratios for all the channels on the whole DSD100 dataset is given hereafter
# In[39]:


data[data.columns[1:]].describe().round(3)

