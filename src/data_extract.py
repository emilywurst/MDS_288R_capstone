#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pathlib import Path
import tempfile
import tarfile
import zstandard  

import os
from glob import iglob

import json
from collections import defaultdict
import numpy as np
import pandas as pd


# In[ ]:


#extract zst files function
def extract_zst(archive: Path, out_path: Path):
    """extract .zst file
    works on Windows, Linux, MacOS, etc.
    
    Parameters
    ----------
    archive: pathlib.Path or str
      .zst file to extract
    out_path: pathlib.Path or str
      directory to extract files and directories to
    """

    archive = Path(archive).expanduser()
    out_path = Path(out_path).expanduser().resolve()
    # need .resolve() in case intermediate relative dir doesn't exist

    dctx = zstandard.ZstdDecompressor()

    with tempfile.TemporaryFile(suffix=".tar") as ofh:
        with archive.open("rb") as ifh:
            dctx.copy_stream(ifh, ofh)
        ofh.seek(0)
        with tarfile.open(fileobj=ofh) as z:
            z.extractall(out_path)


# In[ ]:


extract_zst("C:/Users/ryannewbury/Downloads/acousticbrainz-highlevel-json-20220623.zst",
            "C:/Users/ryannewbury/Downloads/acousticbrainz-highlevel-json-20220623/highlevel")


# In[2]:


#creates directory route for every file in database
rootdir_glob = 'C:/Users/ryannewbury/Downloads/acousticbrainz-highlevel-json-20220623/highlevel/**/*'
# This will return absolute paths
file_list = [f for f in iglob(rootdir_glob, recursive=True) if os.path.isfile(f)]


# In[11]:


#list of directories
file_list[:20]


# In[ ]:


#extraxts data for every file in directories
#cuation this will take a while
dic = defaultdict(list)
for d in file_list:
        f = open(d)
        data = json.load(f)
        #doesn't collect data if no metadata
        if data.get('metadata') == None:
            continue
        #gets mbid from file title
        temp = d.split('\\')[-1].split('-')
        temp.pop(-1);
        id1 = '-'.join(temp)
        #deduplicates mbids by only allowing one id into dictionary
        if id1 in dic['id']:
            continue
        else:
            dic['id'].append(id1)
        #appends new information to dicts
        dic['danceability'].append(data.get('highlevel').get('danceability').get('all').get('danceable'))
        dic['gender_male'].append(data.get('highlevel').get('gender').get('all').get('male'))
        dic['alternative'].append(data.get('highlevel').get('genre_dortmund').get('all').get('alternative'))
        dic['blues'].append(data.get('highlevel').get('genre_dortmund').get('all').get('blues'))
        dic['electronic'].append(data.get('highlevel').get('genre_dortmund').get('all').get('electronic'))
        dic['folkcountry'].append(data.get('highlevel').get('genre_dortmund').get('all').get('folkcountry'))
        dic['funksoulrnb'].append(data.get('highlevel').get('genre_dortmund').get('all').get('funksoulrnb'))
        dic['jazz'].append(data.get('highlevel').get('genre_dortmund').get('all').get('jazz'))
        dic['pop'].append(data.get('highlevel').get('genre_dortmund').get('all').get('pop'))
        dic['raphiphop'].append(data.get('highlevel').get('genre_dortmund').get('all').get('raphiphop'))
        dic['rock'].append(data.get('highlevel').get('genre_dortmund').get('all').get('rock'))
        dic['genre'].append(data.get('highlevel').get('genre_dortmund').get('value'))
        dic['acoustic'].append(data.get('highlevel').get('mood_acoustic').get('all').get('acoustic'))
        dic['aggressive'].append(data.get('highlevel').get('mood_aggressive').get('all').get('aggressive'))
        dic['mood_electronic'].append(data.get('highlevel').get('mood_electronic').get('all').get('electronic'))
        dic['happy'].append(data.get('highlevel').get('mood_happy').get('all').get('happy'))
        dic['party'].append(data.get('highlevel').get('mood_party').get('all').get('party'))
        dic['relaxed'].append(data.get('highlevel').get('mood_relaxed').get('all').get('relaxed'))
        dic['sad'].append(data.get('highlevel').get('mood_sad').get('all').get('sad'))
        dic['mood_mirex_1'].append(data.get('highlevel').get('moods_mirex').get('all').get('Cluster1'))
        dic['mood_mirex_2'].append(data.get('highlevel').get('moods_mirex').get('all').get('Cluster2'))
        dic['mood_mirex_3'].append(data.get('highlevel').get('moods_mirex').get('all').get('Cluster3'))
        dic['mood_mirex_4'].append(data.get('highlevel').get('moods_mirex').get('all').get('Cluster4'))
        dic['mood_mirex_5'].append(data.get('highlevel').get('moods_mirex').get('all').get('Cluster5'))
        dic['timbre_bright'].append(data.get('highlevel').get('timbre').get('all').get('bright'))
        dic['tonal'].append(data.get('highlevel').get('tonal_atonal').get('all').get('tonal'))
        dic['instrumental'].append(data.get('highlevel').get('voice_instrumental').get('all').get('instrumental'))
        dic['bit_rate'].append(data.get('metadata').get('audio_properties').get('bit_rate'))
        dic['codec'].append(data.get('metadata').get('audio_properties').get('codec'))
        dic['length'].append(data.get('metadata').get('audio_properties').get('length'))
        dic['lossless'].append(data.get('metadata').get('audio_properties').get('lossless'))
        dic['replay_gain'].append(data.get('metadata').get('audio_properties').get('replay_gain'))
        dic['true_genre'].append(data.get('metadata').get('tags').get('genre'))

        #same with metadata, but metadata in lists so needs an except if the list is empty
        try:
            dic['artist'].append(data.get('metadata').get('tags').get('artist')[0])
        except:
            dic['artist'].append(data.get('metadata').get('tags').get('artist'))       
        try:
            dic['album'].append(data.get('metadata').get('tags').get('album')[0])
        except:
            dic['album'].append(data.get('metadata').get('tags').get('album'))
        try:
            dic['bpm'].append(data.get('metadata').get('tags').get('bpm')[0])
        except:
            dic['bpm'].append(data.get('metadata').get('tags').get('bpm'))
        try:
            dic['year'].append(int(data.get('metadata').get('tags').get('date')[0].split('-')[0]))
        except:
            dic['year'].append(data.get('metadata').get('tags').get('date'))
        try:
            dic['date'].append(data.get('metadata').get('tags').get('date')[0])     
        except:
            dic['date'].append(data.get('metadata').get('tags').get('date'))
        try:
            dic['label'].append(data.get('metadata').get('tags').get('label')[0])
        except:
            dic['label'].append(data.get('metadata').get('tags').get('label'))
        try:
            dic['song'].append(data.get('metadata').get('tags').get('title')[0])
        except:
            dic['song'].append(data.get('metadata').get('tags').get('title'))
        try:
            dic['artistsort'].append(data.get('metadata').get('tags').get('artistsort')[0])
        except:
            dic['artistsort'].append(data.get('metadata').get('tags').get('artistsort'))


# In[131]:


#creates dataframe from dict created
data = pd.DataFrame(dic)


# In[137]:


len(data)


# In[141]:


#number of rows with a value in year
data['year'].count()


# In[139]:


#remove null years
data = data[data['year'].isnull() == False]


# In[140]:


len(data)


# In[142]:


data.head(10)


# In[144]:


#loads features extracted from acousticbrainz lowlevel features, from 3 different csvs on their website
lowlevel = pd.read_csv("C:\\Users\\ryannewbury\\Downloads\\acousticbrainz-lowlevel-features-20220623\\acousticbrainz-lowlevel-features-20220623-lowlevel.csv")
rhythm = pd.read_csv("C:\\Users\\ryannewbury\\Downloads\\acousticbrainz-lowlevel-features-20220623\\acousticbrainz-lowlevel-features-20220623-rhythm.csv")
tonal = pd.read_csv("C:\\Users\\ryannewbury\\Downloads\\acousticbrainz-lowlevel-features-20220623\\acousticbrainz-lowlevel-features-20220623-tonal.csv")


# In[147]:


lowlevel.head(10)


# In[148]:


rhythm.head(10)


# In[158]:


tonal.head(10)


# In[175]:


#length of the 3 csv files
len(rhythm)


# In[177]:


len(tonal)


# In[179]:


len(lowlevel)


# In[174]:


#number of unique ids in csvs
rhythm['mbid'].nunique()


# In[176]:


tonal['mbid'].nunique()


# In[178]:


lowlevel['mbid'].nunique()


# In[180]:


#dropping duplicates from csvs
rhythm = rhythm.drop_duplicates(subset = ['mbid'], ignore_index = True)


# In[181]:


len(rhythm)


# In[182]:


tonal = tonal.drop_duplicates(subset = ['mbid'], ignore_index = True)


# In[183]:


len(tonal)


# In[184]:


lowlevel = lowlevel.drop_duplicates(subset = ['mbid'], ignore_index = True)


# In[185]:


len(lowlevel)


# In[186]:


#joining all data into one dataframe
data1 = data.set_index('id').join(rhythm.set_index('mbid'),how = 'left',rsuffix = '_2')


# In[189]:


data2 = data1.join(tonal.set_index('mbid'),how = 'left', rsuffix = '_2')


# In[190]:


data3 = data2.join(lowlevel.set_index('mbid'),how = 'left', rsuffix = '_2')


# In[5]:


len(data3)


# In[6]:


data3.head(10)


# In[7]:


data3.isnull().sum()


# In[8]:


data3.to_csv('data.csv')

