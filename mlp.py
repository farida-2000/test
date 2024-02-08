# #convert video to audio
# from moviepy.editor import *
# #
# # def MP4ToMP3(mp4, mp3):
# #     FILETOCONVERT = AudioFileClip(mp4)
# #     FILETOCONVERT.write_audiofile(mp3)
# #     FILETOCONVERT.close()
# #
# # VIDEO_FILE_PATH = "sample-5s.mp4"
# # AUDIO_FILE_PATH = "sample.mp3"
# # MP4ToMP3(VIDEO_FILE_PATH, AUDIO_FILE_PATH)
# #
# from torch.utils.data import Dataset, DataLoader
# import pandas as pd
# from pydub import AudioSegment
# #
# # # Open an mp3 file
# # song = AudioSegment.from_file("sample.mp3",
# #                               format="mp3")
# #
# # # pydub does things in milliseconds
# # ten_seconds = 4 * 1000
# #
# # # song clip of 10 seconds from starting
# # first_10_seconds = song[:ten_seconds]
# #
# # # save file
# # first_10_seconds.export("first_4_seconds.wav",
# #                         format="wav")
# # print("New Audio file is created and saved")
#

#
# # enter the path of your audio file
# sound1 = AudioSegment.from_mp3("first_4_seconds.wav")
# df=pd.read_csv('C1019_VC_3_FullCon_Wk23_Day3_081419.ft_DS2.csv')
# df=df.loc[(df['role'] >= "Participant")]
# count=len(df)
# c=0
# def audio_cut():
# 	for index,row in df.iterrows():
# 		StrtTime=row['st']
# 		EndTime=row['et']
# 		# StrtSec = 1
# 		#
# 		# EndMin = 4
# 		# EndSec = 4
# 		#
# 		# StrtTime = StrtMin*60*1000+StrtSec*1000
# 		# EndTime = StrtMin*60*1000+EndSec*1000
# 		extract = sound1[StrtTime:EndTime]
# 		if c==0:
# 			temp=extract
# 			c=1
# 		temp=temp+extract
# 	temp.export("temp.wav",format="wav")
# Saving file in required location
# merge two audio
# def merge_two_songs(sound1,sound2):
# 	print("Sound Overlay")
# 	sound3 = sound1.append(sound2,crossfade=1500)
# 	sound3.export("merge_sound",format="wav")
#
# merge_two_songs(sound1,sound2)
import pandas as pd
from pydub import AudioSegment
from moviepy.editor import *
import os
def audio_embedding(directory,s):
    # dataset = CustomDataset(directory)
    # data_loader = DataLoader(dataset, batch_size=5, shuffle=True)
    # for files in data_loader:
    print(directory)
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if "Wk04_Day1" in f:
            FILETOCONVERT = AudioFileClip(f)
            FILETOCONVERT.write_audiofile("temp.mp3")
            FILETOCONVERT.close()
            sound1 = AudioSegment.from_mp3("temp.mp3")
            d="/media/data1/farida1/t/craft_csv"
            for name in os.listdir(d):
                f = os.path.join(d, name)
                if s in f:
                        for csvfile in os.listdir(f):
                            address=os.path.join(f, csvfile)
                            print(address)
                            df=pd.read_csv(address)
            df=df.loc[(df['role'] >= "Participant")]
            c=0
            for index,row in df.iterrows():
            	StrtTime=row['st']
            	EndTime=row['et']
            		# StrtSec = 1
            		#
            		# EndMin = 4
            		# EndSec = 4
            		#
            		# StrtTime = StrtMin*60*1000+StrtSec*1000
            		# EndTime = StrtMin*60*1000+EndSec*1000
            	extract = sound1[StrtTime:EndTime]
            	if c==0:
            		temp=extract
            		c=1
            	temp=temp+extract
            location="/media/data1/farida1/t/craft_audio/"
            temp.export(location+s+".wav",format="wav")
    return temp
directory = "/media/data1/I-CONECT/Video_Recordings/"
sub=["C1007","C1018","C1019","C1033","C1034","C1039","C1046","C1054","C1070","C1083","C1103","C1104","C1113","C1124","C1127","C1133","C1156","C1159","C1162","C1166","C1168","C1170","C1179","C1184","C1202","C1229","C1238","C2040","C2046"]
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    for inside in os.listdir(f):
        if "Wk04_Day1" in inside:
            #path = os.path.join(f, inside)
            if filename in sub:
                print(filename)
                f = os.path.join(directory, filename)
                audio_embedding(f,filename)
# import torch
# import random
# v = torch.randn((3, 2, 128))  # visual embeddings
# q = torch.randn(3, 2, 128)  # word embeddings
# joint_rep = torch.cat((v, q))
# print("shape is: ",joint_rep.shape)
# joint_rep = joint_rep.reshape(1, 1536)
# print("shape is: ",joint_rep.shape)
#
# a = random.sample(range(1, 2), 1)
# q_lens = torch.LongTensor(a)
# print(q_lens.shape)

#preprocessing to extract participant role for depression on WOIZ
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    for inside in os.listdir(f):
        if "TRANCRIPT.csv" in inside:
            name=inside.split('_')[0]
            df=df.loc[(df['speaker'] >= "Participant")]
            text=df['value']
            text_file = open(name+".txt", "w")
            text_file.write(text)
            text_file.close()
