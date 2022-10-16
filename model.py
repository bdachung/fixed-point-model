import torch.nn.functional as F
import math

from torch.nn.init import xavier_normal_, constant_
import torch.nn.functional as F
import torch.nn as nn
import torch

# Transformer positinal coding
class PositionalEncoding(nn.Module):
    def __init__(self, num_cycles, embed_dim, dropout=0.):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(num_cycles, embed_dim) ### HERE. Dang combine cycles with points which should be time
        position = torch.arange(0, num_cycles, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).view(-1, num_cycles, embed_dim)
        self.register_buffer('pe', pe)

    def forward(self, x): # batch, num_cycles, embed_dim
        x = x + self.pe[:x.size(0), :,:] ###HERE
        return self.dropout(x)

# customized coding version 1
class PositionalEncodingSimple(nn.Module):
    def __init__(self, num_cycles, embed_dim, dropout=0.):
        super(PositionalEncodingSimple, self).__init__()
        pe = torch.arange(0.0, num_cycles, dtype=torch.float) / num_cycles # linear array of position
        pe = pe.unsqueeze(0).unsqueeze(2)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x): # batch, num_cycles, embed_dim
        pe = self.pe.repeat(x.size(0),1,1)
        return self.dropout(torch.cat([x, pe], 2))


# customized coding version 2
class PositionalEncodingSimple2(nn.Module):
    def __init__(self, num_cycles, embed_dim, dropout=0.):
        super(PositionalEncodingSimple2, self).__init__()
        #pe = 2**(-num_cycles +1 + torch.arange(0.0, num_cycles, dtype=torch.float))
        pe = torch.tensor([0]*(num_cycles - 9) + [-0.125,0.125,-0.25,0.25,-0.5,0.5, -1, 1,0]) # exponential array of position
        pe = pe.unsqueeze(0).unsqueeze(2)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x): # batch, num_cycles, embed_dim
        pe = self.pe.repeat(x.size(0),1,1)
        return self.dropout(torch.cat([x, pe], 2))

# customized coding version 3: trainable weights
class PositionalEncodingTrainable(nn.Module):
    def __init__(self, num_cycles, embed_dim, dropout=0.):
        super(PositionalEncodingTrainable, self).__init__()
        self.pe = torch.nn.Parameter(torch.empty(1,num_cycles,1))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x): # batch, num_cycles, embed_dim
        pe = self.pe.repeat(x.size(0),1,1)
        return self.dropout(torch.cat([x, pe], 2))


class CycleWiseSelfAttention(nn.Module):
    def __init__(self, num_cycles, input_dim, embed_dim, dropout=0.):
        super(CycleWiseSelfAttention, self).__init__()
        self.num_cycles = num_cycles 
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        self.q_proj_weight = nn.Linear(input_dim, embed_dim)
        self.k_proj_weight = nn.Linear(input_dim, embed_dim)
        self.v_proj_weight = nn.Linear(input_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

        #self._init_parameters()
    
    def forward(self, query, key, value):
        # query.shape = batch_size x num_cycles x embed_dim
        q = F.relu(self.q_proj_weight(query))
        k = F.relu(self.k_proj_weight(key))
        v = F.relu(self.v_proj_weight(value))
        q = q * (float(self.embed_dim) ** -0.5)
        attn_output_weights = F.softmax(q @ k.transpose(-2, -1), dim=-1)
        attn_output_weights = self.dropout(attn_output_weights)
        out = attn_output_weights @ v
        return out, attn_output_weights


class MCTransformer(nn.Module):
    def __init__(self, embedding_mode, out_embed_dim, dropout=0, layer_norm_eps=1e-5, num_cycles=10):
        super(MCTransformer, self).__init__()
        self.out_embed_dim = out_embed_dim

        # Embedding 2
        if embedding_mode == 'ConstSeries':
            self.pos_encoder = PositionalEncodingSimple2(num_cycles, out_embed_dim)
            self.cwa1 = CycleWiseSelfAttention(num_cycles, out_embed_dim, out_embed_dim, dropout)
            self.cwa2 = CycleWiseSelfAttention(num_cycles, out_embed_dim, out_embed_dim, dropout)
        # Embedding Original
        elif embedding_mode == 'Original':
            self.pos_encoder = PositionalEncoding(num_cycles, out_embed_dim)
            self.cwa1 = CycleWiseSelfAttention(num_cycles, out_embed_dim, out_embed_dim, dropout)
            self.cwa2 = CycleWiseSelfAttention(num_cycles, out_embed_dim, out_embed_dim, dropout)
        # Embedding Trainable
        elif embedding_mode == 'Trainable':
            self.pos_encoder = PositionalEncodingTrainable(num_cycles, out_embed_dim)
            self.cwa1 = CycleWiseSelfAttention(num_cycles, out_embed_dim, out_embed_dim, dropout)
            self.cwa2 = CycleWiseSelfAttention(num_cycles, out_embed_dim, out_embed_dim, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(out_embed_dim, layer_norm_eps)
        self.norm2 = nn.LayerNorm(out_embed_dim, layer_norm_eps)

        #self._init_parameters()
        
    def _init_parameters(self):
        xavier_normal_(self.in_proj_weight)
        constant_(self.in_proj_bias, 0.)

    def forward(self, x): # batch, num_cycles, out_embed_dim
        x = self.pos_encoder(x) # batch, num_cycles, out_embed_dim (+ 1)

        x2, attn_weights = self.cwa1(x, x, x)
        x = x + self.dropout1(x2)
        x = self.norm1(x)

#         x2, attn_weights = self.cwa2(x, x, x)
#         x = x + self.dropout2(x2)
#         x = self.norm2(x)

        return x, attn_weights

class FeatureBuilder(nn.Module): #num_labels, num_cycles, in_embed_dim, out_embed_dim, hidden_dim,
    def __init__(self, mode, out_embed_dim, sequence_feature_count=3, num_points=11, global_feature_count=6):
        super(FeatureBuilder, self).__init__()
        self.mode = mode
        if self.mode == 'Conv1D_k3mk3':
            self.conv1 = torch.nn.Conv1d(in_channels=sequence_feature_count, out_channels=16, kernel_size=3, stride=1, padding=0, bias=True)
            self.max1d = torch.nn.MaxPool1d(3, stride=2, padding=0)
            self.conv2 = torch.nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=0, bias=True)
            # output of Conv1D is 32
            self.in_proj = nn.Linear(32 + global_feature_count, out_embed_dim)
        elif self.mode == 'Conv1D_k5k5':
            self.conv1 = torch.nn.Conv1d(in_channels=sequence_feature_count, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True)
            self.conv2 = torch.nn.Conv1d(in_channels=16, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True)
            # output of Conv1D is 48
            self.in_proj = nn.Linear(48 + global_feature_count, out_embed_dim)
        elif self.mode == 'LSTM_uni':
            self.gru = torch.nn.GRU(input_size=sequence_feature_count, hidden_size=30, num_layers=1, batch_first=True, bidirectional=False)
            # output of LSTM is 30
            self.in_proj = nn.Linear(30 + global_feature_count, out_embed_dim)
        elif self.mode == 'LSTM_bi':
            self.gru = torch.nn.GRU(input_size=sequence_feature_count, hidden_size=15, num_layers=1, batch_first=True, bidirectional=True)
            # output of LSTM is 2*15=30
            self.in_proj = nn.Linear(30 + global_feature_count, out_embed_dim)
        elif self.mode == 'Flatten':
            self.in_proj = nn.Linear(num_points * sequence_feature_count + global_feature_count, out_embed_dim)
        

    def forward(self, channels_features, global_features): # batch, num_cycles, num_points, sequence_feature_count=3
                                                            # batch, num_cycles, global_feature_count = 6
        batch_size = channels_features.shape[0]
        num_cycles = channels_features.shape[1]
        num_points = channels_features.shape[2]
        sequence_feature_count = channels_features.shape[3]
        global_feature_count = global_features.shape[-1]
        if self.mode == 'Conv1D_k5k5':
            x = torch.reshape(channels_features, (-1, num_points, sequence_feature_count))
            x = x.transpose(1,2)
            x = self.conv1(x)
            x = self.conv2(x)
            x = torch.reshape(x, [batch_size, num_cycles, -1])
        elif self.mode == 'Conv1D_k3mk3':
            x = torch.reshape(channels_features, (-1, num_points, sequence_feature_count))
            x = x.transpose(1,2)
            x = self.conv1(x)
            x = self.max1d(x)
            x = self.conv2(x)
            x = torch.reshape(x, [batch_size, num_cycles, -1])
        elif self.mode == 'LSTM_uni' or self.mode == 'LSTM_bi':
            x = torch.reshape(channels_features, (-1, num_points, sequence_feature_count))
            _, x = self.gru(x)
            x = x.transpose(0,1)
            x = torch.reshape(x, [batch_size, num_cycles, -1])
        elif self.mode == 'Flatten':
            x = torch.reshape(channels_features, (-1, num_cycles, num_points * sequence_feature_count))
        x = torch.cat([x, global_features], axis=2)
        x = F.relu(self.in_proj(x)) # batch, num_cycles, out_embed_dim
        return x

class Regressor(nn.Module):
    def __init__(self, model_mode, feature_mode, use_AR, num_labels, out_embed_dim, hidden_dim, dropout=0., num_cycles=10):
        super(Regressor, self).__init__()
        self.mode = model_mode 
        self.use_AR = use_AR
        if model_mode == 'bert-newposcode' or model_mode == 'bert-trainposcode':
            self.featurebuild = FeatureBuilder(feature_mode, out_embed_dim - 1)
        else:
            self.featurebuild = FeatureBuilder(feature_mode, out_embed_dim)

        if self.mode == 'bert':
            self.simplebert = MCTransformer('Original', out_embed_dim, dropout) 
        elif self.mode == 'bert-newposcode':
            self.simplebert = MCTransformer('ConstSeries', out_embed_dim, dropout) 
        elif self.mode == 'bert-trainposcode':
            self.simplebert = MCTransformer('Trainable', out_embed_dim, dropout) 
        elif self.mode == 'lstm':
            self.lstm_cycles = nn.GRU(out_embed_dim, out_embed_dim, num_layers=1, batch_first=True)

        self.w1 = nn.Linear(out_embed_dim, hidden_dim, bias=True)
        self.w2 = nn.Linear(hidden_dim, num_labels, bias=True)

        self.highway = nn.Linear(num_cycles, 1, bias=True)

        self.dropout = nn.Dropout(dropout)
        self.normalized = nn.LayerNorm([hidden_dim])
        
        # self._init_parameters()
    
    def _init_parameters(self):
        xavier_normal_(self.w1)
        xavier_normal_(self.w2)
        constant_(self.b1, 0.)
        constant_(self.b2, 0.)


    def forward(self, channels_features, global_features): # batch, num_cycles, num_points, sequence_feature_count=3
                                                            # batch, num_cycles, global_feature_count = 6
                                                            # => batch, num_labels
        if self.use_AR:
            sohs = global_features[:,:,-1]
            linear_part = self.highway(sohs)

        x = self.featurebuild(channels_features, global_features) # batch, num_cycles, out_embed_dim
        
        if self.mode == 'bert' or self.mode == 'bert-newposcode' or self.mode == 'bert-trainposcode':
            x, attn_weights = self.simplebert(x) # batch, num_cycles, out_embed_dim
            # Only use the embedding of the latest cycle for prediction
            x = torch.mean(x, dim=1)
#             x = torch.flatten(x[:,-1], start_dim=1)
        elif self.mode == 'lstm':
            _, x = self.lstm_cycles(x)
            x = torch.squeeze(x.transpose(0,1), 1)
        x = self.dropout(F.relu(self.normalized(self.w1(x))) )# Add normalization as Dang suggests
        x = self.w2(x)

        if self.use_AR:
            return linear_part + x, x
        else:
            return x, x