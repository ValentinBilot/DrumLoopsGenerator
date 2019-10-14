# @Author: bilotv
# @Date:   2019-03-13T09:59:33+01:00
# @Last modified by:   bilotv
# @Last modified time: 2019-03-13T15:12:38+01:00



import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
filenames = os.listdir("../../AllWaveFiles")
print(filenames)
for f in tqdm(filenames):

    y, sr = librosa.load("../../AllWaveFiles/"+f, sr=48000)

    # FROM Here
    mb  = librosa.filters.mel(48000, 2048, 240)
    S = librosa.stft(y, n_fft=2048, win_length=1024, hop_length=240)

    S_mel = mb.dot(abs(S))

    #S_rec = mb.T.dot(S_mel)

    # TO Here

    # All credits to Antoine CAILLON
    #print(S_mel.shape)
    #break
    np.save("MelAll"+f[-5]+"/"+f, S_mel)
