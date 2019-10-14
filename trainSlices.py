import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
from tensorboardX import SummaryWriter
from tqdm import tqdm
import models
from models import Slices25Encoder, Slices25Decoder
import librosa.output
import librosa.display
import os
import dataLoader
import LossUtil

writer = SummaryWriter("runs/slices_mmd_3_dim")



def recontructFromSlicesWithExp(slices, lenSeq):
    sampled_slices = slices[0,:,0,:,:]
    sampled_output = np.zeros(shape=(240,lenSeq*25))+0.5
    for i in range(lenSeq):
        sampled_output[:,i*25:i*25+25] = sampled_slices[i]
    if sampled_output.min() == sampled_output.max():
        print("problem, min and max are the same : ", sampled_output.max())
        print("sampled output shape: ",sampled_output.shape)
    sampled_output = np.exp(sampled_output)
    sampled_output = sampled_output-sampled_output.min()
    sampled_output = sampled_output/sampled_output.max()
    if sampled_output.max() != 1.0:
        print("normalisation failed, max = ", sampled_output.max())
    # print(sampled_output.max())
    # print(sampled_output.min())
    return sampled_output

# Griffin Lim Algorithm
def GL(S, n=100, mel=False):
    try:
        mb  = librosa.filters.mel(48000, 2048, 240)
        if mel:
            S = mb.T.dot(S)
        S /= np.max(abs(S))
        phase = np.random.randn(*S.shape)

        for i in range(n):
            y = librosa.istft(S*np.exp(1j*phase),
                        win_length=1024,
                        hop_length=240,
                        center=False)
            S_ = librosa.stft(y,
                        n_fft=2048,
                        win_length=1024,
                        hop_length=240,
                        center=False)
            phase = np.angle(S_)
    except:
        y=np.array([0.5]*len(S)*240)
        print("Error while doing GL")
    return y

def F_plot2(data_m, title, col_v=np.zeros(0), row_v=np.zeros(0), labelCol='', labelRow=''):
    #plt.imshow(data_m, origin='lower', aspect='auto', extent=[row_v[0], row_v[-1], col_v[0], col_v[-1]], interpolation='nearest')
    fig= plt.figure()
    plt.imshow(data_m, origin='lower', aspect='auto', interpolation='nearest')
    plt.colorbar()
    plt.set_cmap('magma')
    plt.xlabel("Time")
    plt.ylabel("Mel Frequency")
    plt.title(title)
    #plt.grid(True)
    #time.sleep(1)
    #plt.show()
    return fig

def plot_waveform(sound):
    fig = plt.figure(figsize=(10, 4))
    librosa.display.waveplot(sound, sr=48000)
    return fig

def sample(output):
    output = output.detach().cpu().numpy()
    F_plot2(output[0][0])


lenSeq=1

#note_dimensions have to be 3 for the Slices25Decoder to understand it
note_dimensions=3
slicesEncoder = Slices25Encoder(1024, note_dimensions=note_dimensions)
slicesDecoder = Slices25Decoder(1024, note_dimensions=note_dimensions)

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
print("use_cuda")
#use_cuda = True
print(use_cuda)
device = torch.device("cuda:0" if use_cuda else "cpu")

slicesEncoder.to(device)
slicesDecoder.to(device)



learning_rate_generator = 1e-3
learning_rate_discriminator = 1e-3

reconstruction_criterion = nn.MSELoss()
slices_reg = nn.MSELoss()


slicesEncoder_optimizer = torch.optim.Adam(slicesEncoder.parameters(), lr = learning_rate_generator, weight_decay=1e-5)
slicesDecoder_optimizer = torch.optim.Adam(slicesDecoder.parameters(), lr = learning_rate_discriminator, weight_decay=1e-5)



# Begin training

print_every = 5
sample_every = 20
save_every = 100 #applies to meta_epoch
max_meta_epoch = 10 #will be multiplied by the number of subsets (10)
epoch = 1000
lossTotal=0
lossReconstruction = 0
lossRegularisation = 0
#lossCartogaphie = 0
lossList=[]
reconstruction_to_audio=True


save = True
loadConv = False
come_back_from = 0

if loadConv:
    slicesEncoder_state_dict = torch.load(f"reconstruction_saved_models/slices25Encoder_{come_back_from}.pt", map_location=device)
    slicesDecoder_state_dict = torch.load(f"reconstruction_saved_models/slices25Decoder_{come_back_from}.pt", map_location=device)
    slicesEncoder.load_state_dict(slicesEncoder_state_dict)
    slicesDecoder.load_state_dict(slicesDecoder_state_dict)
    slicesEncoder_optimizer_state_dict = torch.load(f"reconstruction_saved_models/slices25Encoder_optimizer_{come_back_from}.pt", map_location=device)
    slicesDecoder_optimizer_state_dict = torch.load(f"reconstruction_saved_models/slices25Decoder_optimizer_{come_back_from}.pt", map_location=device)
    slicesEncoder_optimizer.load_state_dict(slicesEncoder_optimizer_state_dict)
    slicesDecoder_optimizer.load_state_dict(slicesDecoder_optimizer_state_dict)
    print("networks loaded")

step=0
warmup=10
rootname = "DATA/MelAll0"
filenames = os.listdir(rootname)
dataset = dataLoader.ReconstructionDrumSoundsSlices25Dataset(filenames)
# Create generators
params = {'batch_size': 400,
          'shuffle': True,
          'num_workers': 6}
training_generator = torch.utils.data.DataLoader(dataset, **params)

for epoch in tqdm(range(epoch)):

    lossTotal=0
    lossReconstruction=0
    lossSlices=0
    
    size = 0


    
    for local_batch in training_generator:
   
        size+=len(local_batch)
        step+=1
        slicesEncoder.zero_grad()
        slicesDecoder.zero_grad()

        # Transfer to GPU
        local_batch = local_batch.to(device)

        latent_sequence, means_sequence, std_sequence = slicesEncoder(local_batch, lenSeq=lenSeq)

        #print(latent_sequence.max())
        output = slicesDecoder(latent_sequence, lenSeq=lenSeq)

        #cartographie__loss = 0.01*cartographie_criterion(means,local_labels.to(device))
        #cartographie__loss.backward(retain_graph=True)
        #regularisation_loss = 0.0001*slices_reg(torch.cat((means_sequence.norm(dim=2).unsqueeze(dim=2),std_sequence),2),torch.tensor([[[0.8]+[0.0]*3]*lenSeq]*len(local_batch)).to(device))


        reconstruction_loss = reconstruction_criterion(output, local_batch)
        reconstruction_loss.backward(retain_graph=True)




        #slicesEncoder_optimizer.step()
        #slicesDecoder_optimizer.step()

        if means_sequence.max()>10:
            mean_loss=slices_reg(means_sequence, torch.zeros_like(means_sequence))
            mean_loss.backward(retain_graph=True)
            mean_loss=mean_loss.item()
        else:
            mean_loss=0
        if std_sequence.max()>5:
            std_loss = slices_reg(std_sequence, torch.zeros_like(std_sequence))
            std_loss.backward(retain_graph=True)
            std_loss=std_loss.item()
        else:
            std_loss=0
        #slicesEncoder.zero_grad()
        #slicesDecoder.zero_grad()
        #slicesEncoder_optimizer.zero_grad()
        #slicesDecoder_optimizer.zero_grad()
        regularisation_loss = LossUtil.compute_mmd(latent_sequence.reshape(-1,note_dimensions))*1e-2
        regularisation_loss.backward()


        #print(regularisation_loss)
        

        #print(latent_sequence.shape, means_sequence.shape, std_sequence.shape)

        #print(latent_sequence.shape)

        #loss = regularisation_loss + reconstruction_loss
        #loss.backward()
        slicesEncoder_optimizer.step()
        slicesDecoder_optimizer.step()


        lossReconstruction += reconstruction_loss.item()
        lossSlices += regularisation_loss.item()
        max_mean = means_sequence.max().item()
        max_std = std_sequence.max().item()

        lossTotal = lossReconstruction+lossSlices

    # ================================================================== #
    #                        Tensorboard Logging                         #
    # ================================================================== #

    # 1. Log scalar values (scalar summary)

        writer.add_scalar('Slices_Loss/lossTotal', lossTotal/size, step)
        writer.add_scalar('Slices_Loss/lossReconstruction', lossReconstruction/size, step)
        writer.add_scalar('Slices_Loss/lossRegularisationSlices', lossSlices/size, step)
        writer.add_scalar('Latent_monitor/Max_Mean', max_mean, step)
        writer.add_scalar('Latent_monitor/Max_std', max_std, step)
        writer.add_scalar('Slices_Loss/mean_loss', mean_loss, step)
        writer.add_scalar('Slices_Loss/std_loss', std_loss, step)

    if epoch % sample_every == 0:
        slicesDecoder.train(mode=False)
        # Sample from normal distribution in the latent space
        random_in = torch.randn(1,1,note_dimensions).to(device)
        sampled_slices = slicesDecoder(random_in, lenSeq=1)

        original = recontructFromSlicesWithExp(local_batch.detach().cpu().numpy(),lenSeq)
        sampled = recontructFromSlicesWithExp(sampled_slices.detach().cpu().numpy(),lenSeq)
        reconstruction = recontructFromSlicesWithExp(output.detach().cpu().numpy(),lenSeq)

        #write a .wav file
        spectrogram = sampled
        sound = GL(spectrogram, mel=True)
        sound = sound[880:-1000]
        sound = sound/sound.max()
        librosa.output.write_wav(f'Slices_Sampled_sounds/sound{step}.wav', sound, 48000)

        if reconstruction_to_audio:
            spectrogram = reconstruction
            sound = GL(spectrogram, mel=True)
            sound = sound[880:-1000]
            sound = sound/sound.max()
            librosa.output.write_wav(f'Reconstructions/sound{step}.wav', sound, 48000)

        #Strings to identify the original sound in the latent space
        #means_str = str(means[0].detach().cpu().numpy().round(2))
        #std_str = str(std[0].detach().cpu().numpy().round(2))

        #Log to Tensorboard
        writer.add_figure('Original/Original Spectrogram', F_plot2(original, "Original"), step)
        writer.add_figure('Out/Reconstruction Spectrogram', F_plot2(reconstruction, "Reconstruction"), step)
        writer.add_figure('Out/Sampled Spectrogram', F_plot2(sampled, "Sampled"), step)
        writer.add_figure('Waveform', plot_waveform(sound), step)





    if epoch % save_every == 0  and save:
        torch.save(slicesEncoder.state_dict(), f"reconstruction_saved_models/slices25Encoder_{epoch}.pt")
        torch.save(slicesDecoder.state_dict(), f"reconstruction_saved_models/slices25Decoder_{epoch}.pt")
        torch.save(slicesEncoder_optimizer.state_dict(), f"reconstruction_saved_models/slices25Encoder_optimizer_{epoch}.pt")
        torch.save(slicesDecoder_optimizer.state_dict(), f"reconstruction_saved_models/slices25Decoder_optimizer_{epoch}.pt")




torch.save(slicesEncoder.state_dict(), f"reconstruction_saved_models/slices25Encoder_last.pt")
torch.save(slicesDecoder.state_dict(), f"reconstruction_saved_models/slices25Decoder_last.pt")
torch.save(slicesEncoder_optimizer.state_dict(), f"reconstruction_saved_models/slices25Encoder_optimizer_last.pt")
torch.save(slicesDecoder_optimizer.state_dict(), f"reconstruction_saved_models/slices25Decoder_optimizer_last.pt")
print("End of Training")

