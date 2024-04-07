'''This is a sample inference script to demonstrate how to run inference on the model for a single .wav file

Authors
* Jia Qi Yip 2024
'''
import torch
from model.mossformer2 import Mossformer2Wrapper

model_configs = ["mossformer2-librimix-2spk", "mossformer2-wsj0mix-3spk", "mossformer2-whamr-2spk"]
myfiles = ['mix2clear.wav', 'mix2hazy.wav', 'mix3clear.wav', 'mix3haze.wav', 'mix4.wav', 'mix5.wav']
for myfile in myfiles:
    for mc in model_configs:
        model = Mossformer2Wrapper.from_pretrained(f'alibabasglab/{mc}')
        # model.inference(f'./test_samples/{mc}/item0_mix.wav',f'./test_samples/{mc}/model_output')
        model.inference(f'./test_samples/my/{myfile}',f'./test_samples/my{myfile.rstrip(".wav")}/{mc}')

        #O:/Programming/ML/audio/MossFormer2/MossFormer2_standalone/test_samples