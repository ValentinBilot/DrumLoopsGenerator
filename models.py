
import torch
import torch.nn as nn
import numpy as np
import sys
import time


class Slices25Encoder(nn.Module):
    def __init__(self, hidden_units, note_dimensions):
        super(Slices25Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, (9,6), stride=(3,1) ,padding=0, dilation=1, groups=1, bias=False)
        self.conv2 = nn.Conv2d(4, 16, (6,6), stride=(3,2) ,padding=0, dilation=1, groups=1, bias=False)
        self.conv3 = nn.Conv2d(16, 64, (5,3), stride=(2,1) ,padding=0, dilation=1, groups=1, bias=False)
        self.conv4 = nn.Conv2d(64, 128, (8,3), stride=(1,1) ,padding=0, dilation=1, groups=1, bias=False)
        self.lin1 = nn.Linear(2048, hidden_units)
        self.lin1etdemi = nn.Linear(hidden_units, hidden_units)
        self.lin1etquart = nn.Linear(hidden_units, hidden_units)
        self.lin2 = nn.Linear(hidden_units, 200)
        self.lastlin = nn.Linear(200,note_dimensions*2)
        self.activation = nn.ReLU()

        self.norm200 = nn.BatchNorm1d(200)
        self.normh = nn.BatchNorm1d(hidden_units)
        self.norm4 = nn.BatchNorm2d(4)
        self.norm16 = nn.BatchNorm2d(16)
        self.norm64 = nn.BatchNorm2d(64)
        self.norm128 = nn.BatchNorm2d(128)
        self.note_dimensions=note_dimensions
        for p in self.parameters():
            try:
                nn.init.xavier_normal_(p)
            except:
                pass




    def forward(self, input_batch, lenSeq):
        output = input_batch.reshape(len(input_batch)*lenSeq,1,240,25)
        output = self.conv1(output)
        output = self.norm4(output)
        output = self.activation(output)
        output = self.conv2(output)
        output = self.norm16(output)
        output = self.activation(output)
        output = self.conv3(output)
        output = self.norm64(output)
        output = self.activation(output)
        output = self.conv4(output)
        output = self.norm128(output)
        output = self.activation(output)
        output = output.reshape(len(input_batch)*lenSeq, 2048)
        output = self.lin1(output)
        #output = self.normh(output)
        output = self.activation(output)
        output = self.lin1etquart(output)
        #output = self.normh(output)
        output = self.activation(output)
        output = self.lin1etdemi(output)
        #output = self.normh(output)
        output = self.activation(output)
        output = self.lin2(output)
        #output = self.norm200(output)
        output = self.activation(output)
        output = self.lastlin(output)
        output = output.reshape(len(input_batch),lenSeq,-1)
        means = output[:,:,0:self.note_dimensions]
        std = output[:,:,self.note_dimensions:self.note_dimensions*2]
        std_exp = std.exp()
        output = means + torch.randn_like(means)*std_exp

        return output, means, std

class Slices25Decoder(nn.Module):
    def __init__(self, hidden_units, note_dimensions):
        super(Slices25Decoder, self).__init__()
        self.lin0 = nn.Linear(note_dimensions,200)
        self.lin1 = nn.Linear(200, hidden_units)
        self.lin2 = nn.Linear(hidden_units, 2048)
        self.lin1etquart = nn.Linear(hidden_units, hidden_units)
        self.lin1etdemi = nn.Linear(hidden_units, hidden_units)
        self.activation = nn.ReLU()
        self.conv1 = nn.Conv2d(128, 64, (3,3), stride=1 ,padding=0, dilation=1, groups=1, bias=False)
        self.conv2 = nn.Conv2d(64, 16, (2,4), stride=(1,2) ,padding=0, dilation=1, groups=1, bias=False)
        self.conv3 = nn.Conv2d(16, 8, (3,4), stride=1 ,padding=0, dilation=1, groups=1, bias=False)
        self.conv4 = nn.Conv2d(8, 8, (6,5), stride=(1,1) ,padding=0, dilation=1, groups=1, bias=False)
        self.conv5 = nn.Conv2d(8, 4, (6,6), stride=(1,2) ,padding=0, dilation=1, groups=1, bias=False)
        self.conv6 = nn.Conv2d(4, 2, (7,3), stride=(1,1) ,padding=0, dilation=1, groups=1, bias=False)
        self.conv7 = nn.Conv2d(2, 1, (9,4), stride=(1,1) ,padding=0, dilation=1, groups=1, bias=False)

        self.norm200 = nn.BatchNorm1d(200)
        self.normh = nn.BatchNorm1d(hidden_units)
        self.norm2048 = nn.BatchNorm1d(2048)
        self.norm128 = nn.BatchNorm2d(128)
        self.norm64 = nn.BatchNorm2d(64)
        self.norm16 = nn.BatchNorm2d(16)
        self.norm8 = nn.BatchNorm2d(8)
        self.norm4 = nn.BatchNorm2d(4)
        self.norm2 = nn.BatchNorm2d(2)

        self.sig = nn.Sigmoid()
        for p in self.parameters():
            try:
                nn.init.xavier_normal_(p)
            except:
                pass


    def forward(self, input_batch, lenSeq):
        output = input_batch.reshape(len(input_batch)*lenSeq,-1)
        output = self.lin0(output)
        #output = self.norm200(output)
        output = self.activation(output)
        output = self.lin1(output)
        #output = self.normh(output)
        output = self.activation(output)
        output = self.lin1etquart(output)
        #output = self.normh(output)
        output = self.activation(output)
        output = self.lin1etdemi(output)
        #output = self.normh(output)
        output = self.activation(output)
        output = self.lin2(output)
        #output = self.norm2048(output)
        output = self.activation(output)
        output = output.reshape(len(input_batch)*lenSeq, 128, 4, 4)
        output = self.norm128(output)
        output = nn.functional.interpolate(output, scale_factor=2, mode='bilinear')
        output = self.conv1(output)
        output = self.norm64(output)
        output = self.activation(output)
        output = nn.functional.interpolate(output, scale_factor=2, mode='bilinear')
        output = self.conv2(output)
        output = self.norm16(output)
        output = self.activation(output)
        output = nn.functional.interpolate(output, scale_factor=2, mode='bilinear')
        output = self.conv3(output)
        output = self.norm8(output)
        output = self.activation(output)
        output = nn.functional.interpolate(output, scale_factor=2, mode='bilinear')
        output = self.conv4(output)
        output = self.norm8(output)
        output = self.activation(output)
        output = nn.functional.interpolate(output, scale_factor=2, mode='bilinear')
        output = self.conv5(output)
        output = self.norm4(output)
        output = self.activation(output)
        output = nn.functional.interpolate(output, scale_factor=2, mode='bilinear')
        output = self.conv6(output)
        output = self.norm2(output)
        output = self.activation(output)
        output = nn.functional.interpolate(output, scale_factor=2, mode='bilinear')
        output = self.conv7(output)
        #output = output.reshape((len(input_batch)*lenSeq,1,1,6000))
        #output = self.sig(output)
        output = output.reshape((len(input_batch),lenSeq,1,240,25))
        return output



class LatentMaker(nn.Module):
    def __init__(self, hidden_units, latent_dimensions, note_dimensions):
        super(LatentMaker, self).__init__()
        self.rnn = nn.LSTM(note_dimensions, hidden_units, 2, batch_first=True, dropout=0.05)
        self.lin = nn.Linear(hidden_units*4, latent_dimensions*2)
        self.latent_dimensions=latent_dimensions

    def forward(self, input_batch):
        output, (h_n,c_n) = self.rnn(input_batch)
        #Take the last hidden_state of the rnn
        #output = output[:,-1,:]
        lenSeq=len(input_batch[0])
        output = output[:,[lenSeq//4,lenSeq//2,3*lenSeq//4,lenSeq-1],:]
        output = output.reshape(len(output),-1)
        output = self.lin(output)
        means = output[:,0:self.latent_dimensions]
        std = output[:,self.latent_dimensions:self.latent_dimensions*2]
        std_exp = std.exp()

        latent = means + torch.randn_like(means)*std

        return latent, means, std

class MeasureMaker(nn.Module):
    def __init__(self, hidden_units, latent_dimensions, measures_dimensions):
        super(MeasureMaker, self).__init__()
        self.rnn = nn.LSTM(latent_dimensions, hidden_units, 2, batch_first=True, dropout=0.05)
        self.lin = nn.Linear(hidden_units, measures_dimensions*2)
        self.latent_dimensions=latent_dimensions
        self.measures_dimensions = measures_dimensions
        self.inToHidden = nn.Linear(latent_dimensions, hidden_units)

    def forward(self, latent_points, num_measures=4):
        #h0 size : (batch, num_layers, hidden_size)
        h0 = self.inToHidden(latent_points).unsqueeze(0).expand(2,-1,-1).contiguous()
        c0 = self.inToHidden(latent_points).unsqueeze(0).expand(2,-1,-1).contiguous()
        output, (h_n,c_n) = self.rnn(latent_points.unsqueeze(1).expand(-1,num_measures,-1),(h0,c0))
        #Take the last hidden_state of the rnn
        #output = output[:,-1,:]
        output = self.lin(output)
        means = output[:,:,0:self.measures_dimensions]
        std = output[:,:,self.measures_dimensions:self.measures_dimensions*2]
        std_exp = std.exp()

        latent = means + torch.randn_like(means)*std

        return latent, means, std

class BeatMaker(nn.Module):
    def __init__(self, hidden_units, measures_dimensions, beat_dimensions, latent_dimensions):
        super(BeatMaker, self).__init__()
        self.rnn = nn.LSTM(measures_dimensions, hidden_units, 2, batch_first=True, dropout=0.05)
        self.lin = nn.Linear(hidden_units, beat_dimensions*2)
        self.beat_dimensions=beat_dimensions
        self.measures_dimensions = measures_dimensions
        self.inToHidden = nn.Linear(latent_dimensions, hidden_units)

    def forward(self, measures_points, latent_points, num_beat=4):
        h0 = self.inToHidden(latent_points).unsqueeze(0).expand(2,-1,-1).contiguous()
        c0 = self.inToHidden(latent_points).unsqueeze(0).expand(2,-1,-1).contiguous()
        output, (h_n,c_n) = self.rnn(measures_points.unsqueeze(1).expand(-1,num_beat,-1),(h0,c0))
        #Take the last hidden_state of the rnn
        #output = output[:,-1,:]
        output = self.lin(output)
        means = output[:,:,0:self.beat_dimensions]
        std = output[:,:,self.beat_dimensions:self.beat_dimensions*2]
        std_exp = std.exp()

        latent = means + torch.randn_like(means)*std

        return latent, means, std

class NoteMaker(nn.Module):
    def __init__(self, hidden_units, beat_dimensions, note_dimensions, latent_dimensions=2):
        super(NoteMaker, self).__init__()
        self.rnn = nn.LSTM(beat_dimensions, hidden_units, 2, batch_first=True, dropout=0.05)
        self.lin = nn.Linear(hidden_units, note_dimensions*2)
        self.linToGate = nn.Linear(hidden_units, 1)
        self.beat_dimensions=beat_dimensions
        self.note_dimensions = note_dimensions
        self.inToHidden = nn.Linear(latent_dimensions, hidden_units)
        self.sig = nn.Sigmoid()


    def forward(self, beat_points, latent_points, num_note=4):
        h0 = self.inToHidden(latent_points).unsqueeze(0).expand(2,-1,-1).contiguous()
        c0 = self.inToHidden(latent_points).unsqueeze(0).expand(2,-1,-1).contiguous()
        output, (h_n,c_n) = self.rnn(beat_points.unsqueeze(1).expand(-1,num_note,-1),(h0,c0))
        #Take the last hidden_state of the rnn
        #output = output[:,-1,:]
        gate = self.linToGate(output)
        gate = self.sig(gate)

        output = self.lin(output)
        means = output[:,:,0:self.note_dimensions]
        std = output[:,:,self.note_dimensions:self.beat_dimensions*2]
        std_exp = std.exp()

        latent = means + torch.randn_like(means)*std

        return latent, means, std, gate

class NVoicesNoteMaker(nn.Module):
    def __init__(self, hidden_units, beat_dimensions, note_dimensions, num_voice, latent_dimensions=2):
        super(NVoicesNoteMaker, self).__init__()
        self.rnn = nn.LSTM(beat_dimensions, hidden_units, 2, batch_first=True, dropout=0.05)
        self.lin = nn.Linear(hidden_units, note_dimensions*2*num_voice)
        self.linToGate = nn.Linear(hidden_units, num_voice)
        self.beat_dimensions=beat_dimensions
        self.note_dimensions = note_dimensions
        self.inToHidden = nn.Linear(latent_dimensions, hidden_units)
        self.sig = nn.Sigmoid()
        self.voices = num_voice


    def forward(self, beat_points, latent_points, num_note=4):
        h0 = self.inToHidden(latent_points).unsqueeze(0).expand(2,-1,-1).contiguous()
        c0 = self.inToHidden(latent_points).unsqueeze(0).expand(2,-1,-1).contiguous()
        output, (h_n,c_n) = self.rnn(beat_points.unsqueeze(1).expand(-1,num_note,-1),(h0,c0))
        #Take the last hidden_state of the rnn
        #output = output[:,-1,:]
        gate = self.linToGate(output)
        gate = self.sig(gate)

        output = self.lin(output)
        means = output[:,:,0:self.note_dimensions*self.voices]
        std = output[:,:,self.note_dimensions*self.voices:self.beat_dimensions*2*self.voices]
        std_exp = std.exp()

        latent = means + torch.randn_like(means)*std

        return latent, means, std, gate


if __name__=="__main__":
    x = torch.randn((1,64,240,25))
    slicesEncoder = Slices25Encoder(512)
    #latentWalker = LatentWalker(512)
    #metaLatentDecoder = MetaLatentDecoder(512)
    slicesDecoder = Slices25Decoder(512)
    latent_sequence, means_sequence, std_sequence = slicesEncoder(x, lenSeq=64)
    output = slicesDecoder(latent_sequence, lenSeq=64)
    #meta_latent_sequence, meta_latent_means, meta_latent_std = latentWalker(latent_sequence)
    #decoded_meta_latent_sequence = metaLatentDecoder(meta_latent_sequence)
    #print(decoded_meta_latent_sequence.shape)
    #decoded_sequence = slicesDecoder(decoded_meta_latent_sequence, lenSeq=20)
