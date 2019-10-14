import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
from tensorboardX import SummaryWriter
from tqdm import tqdm
import models
from models import Slices25Encoder, Slices25Decoder, LatentMaker, MeasureMaker, BeatMaker, NVoicesNoteMaker
import librosa.output
import librosa.display
import LossUtil

writer = SummaryWriter("runs/RNN_pre_mcnn_sqrt3")


def recontructFromSlicesWithExp(slices, lenSeq):
    sampled_slices = slices[0,:,0,:,:]
    sampled_output = np.zeros(shape=(240,lenSeq*25))
    for i in range(lenSeq):
        sampled_output[:,i*25:i*25+25] = sampled_slices[i]
    #sampled_output = sampled_output**2
    sampled_output = np.exp(sampled_output)
    sampled_output = sampled_output-sampled_output.min()
    sampled_output = sampled_output / sampled_output.max()
    return sampled_output


def GL(S, n=20, mel=False):
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

    return y

def F_plot2(data_m, title, col_v=np.zeros(0), row_v=np.zeros(0), labelCol='', labelRow=''):
    #plt.imshow(data_m, origin='lower', aspect='auto', extent=[row_v[0], row_v[-1], col_v[0], col_v[-1]], interpolation='nearest')
    fig=plt.figure()
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
    fig=plt.figure(figsize=(10, 4))
    librosa.display.waveplot(sound, sr=48000)
    return fig

def sample(output):
    output = output.detach().cpu().numpy()
    F_plot2(output[0][0])

import os
import dataLoader
rootname = "DATA/AllMelLoops"
filenames = os.listdir(rootname)
dataset = dataLoader.MansDrumSoundsSlices25Dataset(filenames)
# Create generators
params = {'batch_size': 2,
          'shuffle': True,
          'num_workers': 6}
training_generator = torch.utils.data.DataLoader(dataset, **params)

# for local_batch, lenSeq in training_generator:
#     lenSeq=lenSeq.item()
#     break

latent_dimensions=64
measures_dimensions=16
beat_dimensions=8
#note_dimensions have to be 3 for the Slices25Decoder to understand it
note_dimensions=3
num_voices=3
slicesEncoder = Slices25Encoder(1024, note_dimensions)
latentMaker = LatentMaker(512, latent_dimensions=latent_dimensions, note_dimensions=note_dimensions)
measureMaker = MeasureMaker(128, latent_dimensions=latent_dimensions, measures_dimensions=measures_dimensions)
beatMaker = BeatMaker(64, measures_dimensions=measures_dimensions, beat_dimensions=beat_dimensions, latent_dimensions=latent_dimensions)
nVoicesNoteMaker = NVoicesNoteMaker(32, beat_dimensions=beat_dimensions, note_dimensions=note_dimensions, num_voice=num_voices, latent_dimensions=latent_dimensions)
slicesDecoder = Slices25Decoder(1024, note_dimensions)

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
print("use_cuda")
#use_cuda = True
print(use_cuda)
device = torch.device("cuda:0" if use_cuda else "cpu")

slicesEncoder.to(device)
latentMaker.to(device)
measureMaker.to(device)
beatMaker.to(device)
nVoicesNoteMaker.to(device)
slicesDecoder.to(device)



learning_rate_generator = 1e-3
learning_rate_discriminator = 1e-3

reconstruction_criterion = nn.MSELoss()
mse = nn.MSELoss()
latent_reg = nn.MSELoss()
measure_reg = nn.MSELoss()
beat_reg = nn.MSELoss()
note_reg = nn.MSELoss()

latentMaker_optimizer = torch.optim.Adam(latentMaker.parameters(), lr = learning_rate_discriminator, weight_decay=1e-7)
measureMaker_optimizer = torch.optim.Adam(measureMaker.parameters(), lr = learning_rate_generator, weight_decay=1e-7)
beatMaker_optimizer = torch.optim.Adam(beatMaker.parameters(), lr = learning_rate_generator, weight_decay=1e-7)
nVoicesNoteMaker_optimizer = torch.optim.Adam(nVoicesNoteMaker.parameters(), lr = learning_rate_generator, weight_decay=1e-7)

# Begin training

print_every = 5
sample_every = 5
save_every = 5
max_epochs = 101

lossList=[]
reconstruction_to_audio=True


save = True
loadRNN = False
come_back_from = 0
slicesEncoder.train(mode=False)
slicesDecoder.train(mode=False)
slicesEncoder_state_dict = torch.load(f"reconstruction_saved_models/slices25Encoder_last.pt", map_location=device)
slicesDecoder_state_dict = torch.load(f"reconstruction_saved_models/slices25Decoder_last.pt", map_location=device)
slicesEncoder.load_state_dict(slicesEncoder_state_dict)
slicesDecoder.load_state_dict(slicesDecoder_state_dict)

if loadRNN:
    latentMaker_state_dict = torch.load(f"mans_voices_saved_models/latentMaker{come_back_from}.pt", map_location=device)
    measureMaker_state_dict = torch.load(f"mans_voices_saved_models/measureMaker{come_back_from}.pt", map_location=device)
    beatMaker_state_dict = torch.load(f"mans_voices_saved_models/beatMaker{come_back_from}.pt", map_location=device)
    nVoicesNoteMaker_state_dict = torch.load(f"mans_voices_saved_models/nVoicesNoteMaker{come_back_from}.pt", map_location=device)
    latentMaker.load_state_dict(latentMaker_state_dict)
    measureMaker.load_state_dict(measureMaker_state_dict)
    beatMaker.load_state_dict(beatMaker_state_dict)
    nVoicesNoteMaker.load_state_dict(nVoicesNoteMaker_state_dict)
    latentMaker_optimizer_state_dict = torch.load(f"mans_voices_saved_models/latentMaker_optimizer{come_back_from}.pt", map_location=device)
    measureMaker_optimizer_state_dict = torch.load(f"mans_voices_saved_models/measureMaker_optimizer{come_back_from}.pt", map_location=device)
    beatMaker_optimizer_state_dict = torch.load(f"mans_voices_saved_models/beatMaker_optimizer{come_back_from}.pt", map_location=device)
    nVoicesNoteMaker_optimizer_state_dict = torch.load(f"mans_voices_saved_models/nVoicesNoteMaker_optimizer{come_back_from}.pt", map_location=device)
    latentMaker_optimizer.load_state_dict(latentMaker_optimizer_state_dict)
    measureMaker_optimizer.load_state_dict(measureMaker_optimizer_state_dict)
    beatMaker_optimizer.load_state_dict(beatMaker_optimizer_state_dict)
    nVoicesNoteMaker_optimizer.load_state_dict(nVoicesNoteMaker_optimizer_state_dict)


step = 0
step_init=step
reg_every=45
for epoch in range(come_back_from,max_epochs):


    reg_loss=0
    for local_batch, lenSeq in tqdm(training_generator):
        lossTotal=0
        lossReconstruction=0
        lossLatent=0
        lossMeasures=0
        lossBeat=0
        lossNote=0
        max_latent=0
        #lenSeq=lenSeq.item()
        lenSeq=64
        step+=1

        latentMaker.zero_grad()
        measureMaker.zero_grad()
        beatMaker.zero_grad()
        nVoicesNoteMaker.zero_grad()


        # Transfer to GPU
        local_batch = local_batch.to(device)
        with torch.no_grad():

            latent_sequence, means_sequence, std_sequence = slicesEncoder(local_batch, lenSeq=lenSeq)

        #latent = (batch, latent_dim)
        latent, means_latent, std_latent = latentMaker(latent_sequence)
        if latent.norm(2, dim=1).max().item()>max_latent:
            max_latent = latent.norm(2, dim=1).max().item()


        #latent_measures = (batch, num_measures, measures_dim)
        latent_measures, means_measures, std_measures = measureMaker(latent)

        #go throught each measure to get the beats, starting by the first to have a tensor to store values
        latent_beats, means_beats, std_beats = beatMaker(latent_measures[:,0,:], latent)
        for i in range (1,len(latent_measures[0])):
            new_beats, new_means_beats, new_std_beats = beatMaker(latent_measures[:,i,:], latent)
            latent_beats = torch.cat((latent_beats, new_beats),1)
            means_beats = torch.cat((means_beats, new_means_beats),1)
            std_beats = torch.cat((std_beats, new_std_beats),1)
        #latent_beats = (batch, num_beats*num_measures, beat_dim)

        #go throught each beat to get the notes, starting by the first to have a tensor to store values
        latent_notes, means_notes, std_notes, gate = nVoicesNoteMaker(latent_beats[:,0,:], latent)
        for i in range (1,len(latent_beats[0])):
            new_notes, new_means_notes, new_std_notes, new_gate = nVoicesNoteMaker(latent_beats[:,i,:], latent)
            latent_notes = torch.cat((latent_notes, new_notes),1)
            means_notes = torch.cat((means_notes, new_means_notes),1)
            std_notes = torch.cat((std_notes, new_std_notes),1)
            gate = torch.cat((gate, new_gate),1)
        #latent_notes = (batch, num_note*num_beats*num_measures, num_voices*note_dim)
        #with torch.no_grad():
        output = slicesDecoder(latent_notes[:,:,0:note_dimensions], lenSeq=lenSeq)
        local_gate = gate[:,:,0].unsqueeze(2).unsqueeze(3).unsqueeze(4).expand(-1,-1,-1,240,25)
        output = output*local_gate

        for i in range(1, num_voices):
            preOutput=slicesDecoder(latent_notes[:,:,i*note_dimensions:(i+1)*note_dimensions], lenSeq=lenSeq)
            local_gate = gate[:,:,i].unsqueeze(2).unsqueeze(3).unsqueeze(4).expand(-1,-1,-1,240,25)
            preOutput = preOutput*local_gate
            output = output + preOutput


        #latent_reg_loss = latent_reg(torch.cat((means_latent.norm(dim=1).unsqueeze(dim=1),std_latent),1),torch.tensor([[0.8]+[0.0]*latent_dimensions]*len(local_batch)).to(device)) /1e5
        # latent_reg_loss.backward(retain_graph=True)
        #measure_reg_loss = measure_reg(torch.cat((means_measures.norm(dim=2).unsqueeze(dim=2),std_measures),2),torch.tensor([[[0.8]+[0.0]*measures_dimensions]*4]*len(local_batch)).to(device)) /1e5

        #beats_reg_loss = beat_reg(torch.cat((means_beats.norm(dim=2).unsqueeze(dim=2),std_beats),2),torch.tensor([[[0.8]+[0.0]*beat_dimensions]*16]*len(local_batch)).to(device)) / 1e5

        #notes_reg_loss = note_reg(torch.cat((means_notes.norm(dim=2).unsqueeze(dim=2),std_notes),2),torch.tensor([[[0.8]+[0.0]*note_dimensions*num_voices]*lenSeq]*len(local_batch)).to(device)) /1e5

        # Here is sanity Check, since sometimes mmd loss leads to exploding latent values
        secure_loss_total=0
        if means_latent.max()>5:
            secure_loss = mse(means_latent, torch.zeros_like(means_latent))/1e3
            secure_loss.backward(retain_graph=True)
            secure_loss_total += secure_loss.item()
        if std_latent.max()>5:
            secure_loss = mse(std_latent, torch.zeros_like(std_latent))/1e3
            secure_loss.backward(retain_graph=True)
            secure_loss_total += secure_loss.item()
        if means_measures.max()>5:
            secure_loss = mse(means_measures, torch.zeros_like(means_measures))/1e3
            secure_loss.backward(retain_graph=True)
            secure_loss_total += secure_loss.item()
        if std_measures.max()>5:
            secure_loss = mse(std_meeasures, torch.zeros_like(std_measures))/1e3
            secure_loss.backward(retain_graph=True)
            secure_loss_total += secure_loss.item()
        if means_beats.max()>5:
            secure_loss = mse(means_beats, torch.zeros_like(means_beats))/1e3
            secure_loss.backward(retain_graph=True)
            secure_loss_total += secure_loss.item()
        if std_beats.max()>5:
            secure_loss = mse(std_beats, torch.zeros_like(std_beats))/1e3
            secure_loss.backward(retain_graph=True)
            secure_loss_total += secure_loss.item()
        if means_notes.max()>5:
            secure_loss = mse(means_notes, torch.zeros_like(means_notes))/1e3
            secure_loss.backward(retain_graph=True)
            secure_loss_total += secure_loss.item()
        if std_notes.max()>5:
            secure_loss = mse(std_notes, torch.zeros_like(std_notes))/1e3
            secure_loss.backward(retain_graph=True)
            secure_loss_total += secure_loss.item()




        reconstruction_loss = reconstruction_criterion(output, local_batch)
        reconstruction_loss.backward(retain_graph=True)
        lossReconstruction += reconstruction_loss.item()



        notes_reg_loss = LossUtil.compute_mmd(latent_notes.reshape(-1, note_dimensions)) /1e3
        lossNote = notes_reg_loss.item()
        writer.add_scalar('LossRNN/lossNote', lossNote, step)
        notes_reg_loss.backward(retain_graph=True)

        if (step==(step_init+1)):
            latent_cat=latent
            latent_measures_cat=latent_measures
            latent_beats_cat=latent_beats



        #if (len(local_batch)!=params["batch_size"]):
        if step !=0 and step%reg_every==0:
            latent_reg_loss = LossUtil.compute_mmd(latent_cat) / 1e2
            measure_reg_loss = LossUtil.compute_mmd(latent_measures_cat.reshape(-1, measures_dimensions)) /1e3
            beats_reg_loss = LossUtil.compute_mmd(latent_beats_cat.reshape(-1, beat_dimensions)) /1e3

            lossLatent = latent_reg_loss.item()
            lossMeasures = measure_reg_loss.item()
            lossBeat = beats_reg_loss.item()

            reg_loss += latent_reg_loss + measure_reg_loss + beats_reg_loss

            writer.add_scalar('LossRNN/lossLatent', lossLatent, step)
            writer.add_scalar('LossRNN/lossMeasures', lossMeasures, step)
            writer.add_scalar('LossRNN/lossBeat', lossBeat, step)

            writer.add_scalar('LossRNN/reg_loss', lossLatent+lossMeasures+lossBeat+lossNote, step)
            reg_loss.backward()
            reg_loss=0

            latent_cat=latent
            latent_measures_cat=latent_measures
            latent_beats_cat=latent_beats


            #print("regularisation time!")

        else:
            latent_cat = torch.cat((latent_cat,latent),0)
            latent_measures_cat = torch.cat((latent_measures_cat,latent_measures),0)
            latent_beats_cat = torch.cat((latent_beats_cat,latent_beats),0)


        latentMaker_optimizer.step()
        measureMaker_optimizer.step()
        beatMaker_optimizer.step()
        nVoicesNoteMaker_optimizer.step()

        #lossCartogaphie += cartographie__loss
        lossTotal = lossReconstruction+lossLatent+lossMeasures+lossBeat+lossNote+secure_loss_total


    # ================================================================== #
    #                        Tensorboard Logging                         #
    # ================================================================== #

    # 1. Log scalar values (scalar summary)


        writer.add_scalar('LossRNN/lossTotal', lossTotal, step)
        writer.add_scalar('LossRNN/lossReconstruction', lossReconstruction, step)
        writer.add_scalar('Latent_monitor_rnn/Latent_Max_Mean', means_latent.max().item(), step)
        writer.add_scalar('Latent_monitor_rnn/Latent_Max_Std', std_latent.max().item(), step)
        writer.add_scalar('Latent_monitor_rnn/Measures_Max_Mean', means_measures.max().item(), step)
        writer.add_scalar('Latent_monitor_rnn/Measures_Max_Std', std_measures.max().item(), step)
        writer.add_scalar('Latent_monitor_rnn/Beats_Max_Mean', means_beats.max().item(), step)
        writer.add_scalar('Latent_monitor_rnn/Beats_Max_Std', std_beats.max().item(), step)
        writer.add_scalar('Latent_monitor_rnn/Notes_Max_Mean', means_notes.max().item(), step)
        writer.add_scalar('Latent_monitor_rnn/Notes_Max_Std', std_notes.max().item(), step)
        writer.add_scalar('LossRNN/Sanity_Check', secure_loss_total, step)



    if epoch % sample_every == 0:
        print("It's sample time man! epoch: ",epoch)
        with torch.no_grad():
            # 3. Log training images (image summary)
            #info = { 'images': images.view(-1, 28, 28)[:10].cpu().numpy() }
            measureMaker.train(mode=False)
            beatMaker.train(mode=False)
            nVoicesNoteMaker.train(mode=False)
            slicesDecoder.train(mode=False)
            # Sample from normal distribution in the latent space

            random_starting_point = torch.randn((1,latent_dimensions)).to(device)
            #latent_measures = (batch, num_measures, measures_dim)
            latent_measures, means_measures, std_measures = measureMaker(random_starting_point)
            #go throught each measure to get the beats, starting by the first to have a tensor to store values
            latent_beats, means_beats, std_beats = beatMaker(latent_measures[:,0,:], random_starting_point)
            for i in range (1,len(latent_measures[0])):
                new_beats, new_means_beats, new_std_beats = beatMaker(latent_measures[:,i,:], random_starting_point)
                latent_beats = torch.cat((latent_beats, new_beats),1)
                means_beats = torch.cat((means_beats, new_means_beats),1)
                std_beats = torch.cat((std_beats, new_std_beats),1)
            #latent_beats = (batch, num_beats*num_measures, beat_dim)

            #go throught each beat to get the notes, starting by the first to have a tensor to store values
            latent_notes, means_notes, std_notes, gate = nVoicesNoteMaker(latent_beats[:,0,:], random_starting_point)
            for i in range (1,len(latent_beats[0])):
                new_notes, new_means_notes, new_std_notes, new_gate = nVoicesNoteMaker(latent_beats[:,i,:], random_starting_point)
                latent_notes = torch.cat((latent_notes, new_notes),1)
                means_notes = torch.cat((means_notes, new_means_notes),1)
                std_notes = torch.cat((std_notes, new_std_notes),1)
                gate = torch.cat((gate, new_gate),1)
            #latent_notes = (batch, num_note*num_beats*num_measures, beat_dim)

            sampled_slices = slicesDecoder(latent_notes[:,:,0:note_dimensions], lenSeq=lenSeq)
            local_gate = gate[:,:,0].unsqueeze(2).unsqueeze(3).unsqueeze(4).expand(-1,-1,-1,240,25)
            sampled_slices = sampled_slices*local_gate

            for i in range(1, num_voices):
                preOutput=slicesDecoder(latent_notes[:,:,i*note_dimensions:(i+1)*note_dimensions], lenSeq=lenSeq)
                local_gate = gate[:,:,i].unsqueeze(2).unsqueeze(3).unsqueeze(4).expand(-1,-1,-1,240,25)
                preOutput = preOutput*local_gate
                sampled_slices = sampled_slices + preOutput



            measureMaker.train(mode=True)
            beatMaker.train(mode=True)
            nVoicesNoteMaker.train(mode=True)


            original = recontructFromSlicesWithExp(local_batch.detach().cpu().numpy(),lenSeq)
            sampled = recontructFromSlicesWithExp(sampled_slices.detach().cpu().numpy(),lenSeq)
            reconstruction = recontructFromSlicesWithExp(output.detach().cpu().numpy(),lenSeq)

            #write a .wav file
            spectrogram = sampled
            sound = GL(spectrogram, mel=True)
            sound = sound[880:-1000]
            sound = sound/sound.max()
            librosa.output.write_wav(f'Sampled_voices_sounds_mans/loop{epoch}.wav', sound, 48000)

            if reconstruction_to_audio:
                spectrogram = reconstruction
                sound = GL(spectrogram, mel=True)
                sound = sound[880:-1000]
                sound = sound/sound.max()
                librosa.output.write_wav(f'Reconstructions/loop{epoch}.wav', sound, 48000)



            writer.add_figure('Original/Original Spectrogram', F_plot2(original, "Original"), epoch)
            writer.add_figure('Out/Reconstruction Spectrogram', F_plot2(reconstruction, "Reconstruction"), epoch)
            writer.add_figure('Out/Sampled Spectrogram', F_plot2(sampled, "Sampled"), epoch)
            writer.add_figure('Waveform', plot_waveform(sound), epoch)

    if epoch % save_every == 0  and save:

        torch.save(latentMaker.state_dict(), f"mans_voices_saved_models/latentMaker{epoch}.pt")
        torch.save(measureMaker.state_dict(), f"mans_voices_saved_models/measureMaker{epoch}.pt")
        torch.save(beatMaker.state_dict(), f"mans_voices_saved_models/beatMaker{epoch}.pt")
        torch.save(nVoicesNoteMaker.state_dict(), f"mans_voices_saved_models/nVoicesNoteMaker{epoch}.pt")

        torch.save(latentMaker_optimizer.state_dict(), f"mans_voices_saved_models/latentMaker_optimizer{epoch}.pt")
        torch.save(measureMaker_optimizer.state_dict(), f"mans_voices_saved_models/measureMaker_optimizer{epoch}.pt")
        torch.save(beatMaker_optimizer.state_dict(), f"mans_voices_saved_models/beatMaker_optimizer{epoch}.pt")
        torch.save(nVoicesNoteMaker_optimizer.state_dict(), f"mans_voices_saved_models/nVoicesNoteMaker_optimizer{epoch}.pt")

print("End of Training")
