import torch
from torch import nn
import torchvision
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TileEncoder(nn.Module):
    """
    Tile TileEncoder.
    """
    def __init__(self, encoded_image_size=14):
        super(TileEncoder, self).__init__()
        self.enc_image_size = encoded_image_size
        self.tanh = torch.nn.Tanh()
        self.resnet = torchvision.models.resnet18(num_classes=511)  

        resnet18_pretn = torchvision.models.resnet18(pretrained=True)
        pretn_state_dict = resnet18_pretn.state_dict()
        model_state_dict = self.resnet.state_dict()
        update_state = {k:v for k, v in pretn_state_dict.items() if k not in ["fc.weight", "fc.bias"] and k in model_state_dict}
        model_state_dict.update(update_state)
        self.resnet.load_state_dict(model_state_dict)

        self.fine_tune()

    def forward(self, images, proportions):
        batch_size = images.size(0)
        image_hw = images.size(-1)
        image_ch = images.size(2)
        image_cl = images.size(1)
        images_flat = images.view((-1, image_ch, image_hw, image_hw))
        out = self.resnet(images_flat)  
        tanh_out = self.tanh(out)
        tanh_out = tanh_out.view((batch_size, image_cl, -1))
        proportions = torch.unsqueeze(proportions, dim=2)
        cat_out = torch.cat([tanh_out, proportions], dim=2)
        return cat_out

    def fine_tune(self, fine_tune=True):
        for p in self.resnet.parameters():
            p.requires_grad = fine_tune


class Attention(nn.Module):
    def __init__(self, tile_encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.tile_encoder_att = nn.Linear(tile_encoder_dim, attention_dim)  
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  
        self.full_att = nn.Linear(attention_dim, 1)  
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  

    def forward(self, tile_out, decoder_hidden):
        att1 = self.tile_encoder_att(tile_out)  
        att2 = self.decoder_att(decoder_hidden)  
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  
        alpha = self.softmax(att)  
        attention_weighted_encoding = (tile_out * alpha.unsqueeze(2)).sum(dim=1)  
        return attention_weighted_encoding, alpha

class DecoderWithAttention(nn.Module):
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=512, dropout=0.5):
        super(DecoderWithAttention, self).__init__()
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  

        self.embedding = nn.Embedding(vocab_size, embed_dim)  
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  
        self.init_weights()   

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, tile_encoder_out):
        mean_encoder_out = tile_encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, tile_out, encoded_captions, caption_lengths):
        batch_size = tile_out.size(0)
        encoder_dim = tile_out.size(-1)
        vocab_size = self.vocab_size

        tile_out = tile_out.view(batch_size, -1, encoder_dim)  
        num_pixels = tile_out.size(1)

        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        tile_out = tile_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        embeddings = self.embedding(encoded_captions)  

        h, c = self.init_hidden_state(tile_out)  

        decode_lengths = (caption_lengths - 1).tolist()

        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(tile_out[:batch_size_t],
                                                                h[:batch_size_t])
            
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  
            preds = self.fc(self.dropout(h))  
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind
