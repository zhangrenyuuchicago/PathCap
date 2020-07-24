import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import Encoder, TileEncoder, DecoderWithAttention
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import sys

data_folder = '../data/images/'  
tile_folder = '/mnt/scratch/renyu/gpu-cluster/ML4HC/NODE2/GTEx_Tiles_jpg/'
cluster_folder = '/mnt/scratch/renyu/gpu-cluster/ML4HC/NODE3/AAAI_rerun/mu_0.1_margin_0.001_real/cluster_epoch3/train_index_3/'
data_name = 'GTEx_1_cap_per_img_5_min_word_freq'  

emb_dim = 512  
attention_dim = 512  
decoder_dim = 512  
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
cudnn.benchmark = True  

start_epoch = 0
epochs = 120  
epochs_since_improvement = 0  
batch_size = 32
workers = 10  

encoder_lr = 1e-4  
decoder_lr = 4e-4  
grad_clip = 5.  
alpha_c = 1.  
best_bleu4 = 0.  
print_freq = 50  
fine_tune_encoder = True  
checkpoint = None  

def main():
    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, word_map
    word_map_file = os.path.join(data_folder, 'word_map.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    if checkpoint is None:
        decoder = DecoderWithAttention(attention_dim=attention_dim,
                                       embed_dim=emb_dim,
                                       decoder_dim=decoder_dim,
                                       vocab_size=len(word_map),
                                       dropout=dropout)
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=decoder_lr)
        encoder = Encoder()
        tile_encoder = TileEncoder()

        encoder.fine_tune(fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=encoder_lr) if fine_tune_encoder else None
        tile_encoder.fine_tune(fine_tune_encoder)
        tile_encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, tile_encoder.parameters()),
                                             lr=encoder_lr) if fine_tune_encoder else None


    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        tile_encoder = checkpoint['tile_encoder']
        tile_encoder_optimizer = checkpoint['tile_encoder_optimizer']

        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=encoder_lr)

    decoder = decoder.to(device)
    encoder = encoder.to(device)
    tile_encoder = tile_encoder.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    
    train_transform = transforms.Compose([
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
    
    train_tile_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ColorJitter(brightness=64.0/255, contrast=0.75, saturation=0.25, hue=0.04),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])


    val_transform = transforms.Compose([
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])

    val_tile_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])

    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, 'train', train_transform, train_tile_transform, tile_folder, cluster_folder), 
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, 'val', val_transform, val_tile_transform, tile_folder, cluster_folder), 
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    for epoch in range(start_epoch, epochs):
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)
                adjust_learning_rate(tile_encoder_optimizer, 0.8)

        train(train_loader=train_loader,
              encoder=encoder,
              tile_encoder=tile_encoder,
              decoder=decoder,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              tile_encoder_optimizer=tile_encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch)

        recent_bleu4 = validate(val_loader=val_loader,
                                encoder=encoder,
                                tile_encoder=tile_encoder,
                                decoder=decoder,
                                criterion=criterion)

        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, tile_encoder, decoder, encoder_optimizer, tile_encoder_optimizer, 
                        decoder_optimizer, recent_bleu4, is_best)


def train(train_loader, encoder, tile_encoder, decoder, criterion, encoder_optimizer, tile_encoder_optimizer, decoder_optimizer, epoch):
    """
    Performs one epoch's training.
    """
    decoder.train()  
    encoder.train()
    tile_encoder.train()

    batch_time = AverageMeter()  
    data_time = AverageMeter()  
    losses = AverageMeter()  
    top5accs = AverageMeter()  

    start = time.time()
    for i, (imgs, caps, caplens, tiles, proportions) in enumerate(train_loader):
        data_time.update(time.time() - start)

        imgs = imgs.to(device)
        tiles = tiles.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)
        proportions = proportions.to(device)

        imgs = encoder(imgs)
        tiles = tile_encoder(tiles, proportions)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, tiles, caps, caplens)

        targets = caps_sorted[:, 1:]

        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        scores = scores.data
        targets = targets.data

        loss = criterion(scores, targets)

        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        if tile_encoder_optimizer is not None:
            tile_encoder_optimizer.zero_grad()

        loss.backward()

        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)
            if tile_encoder_optimizer is not None:
                clip_gradient(tile_encoder_optimizer, grad_clip)

        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()
        if tile_encoder_optimizer is not None:
            tile_encoder_optimizer.step()

        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))


def validate(val_loader, encoder, tile_encoder, decoder, criterion):
    """
    Performs one epoch's validation.
    """
    decoder.eval()  
    if encoder is not None:
        encoder.eval()
    if tile_encoder is not None:
        tile_encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  
    hypotheses = list()  

    with torch.no_grad():
        for i, (imgs, caps, caplens, allcaps, tiles, proportions) in enumerate(val_loader):
            imgs = imgs.to(device)
            tiles = tiles.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)
            proportions = proportions.to(device)

            if encoder is not None:
                imgs = encoder(imgs)
            tiles = tile_encoder(tiles, proportions)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, tiles, caps, caplens)

            targets = caps_sorted[:, 1:]

            scores_copy = scores.clone()
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)

            scores = scores.data
            targets = targets.data

            loss = criterion(scores, targets)

            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))

            allcaps = allcaps[sort_ind]  
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps))  
                references.append(img_captions)

            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

        bleu4 = corpus_bleu(references, hypotheses)
        weight3 = (0.333, 0.333, 0.333)
        bleu3 = corpus_bleu(references, hypotheses, weights=weight3)
        weight2 = (0.5, 0.5)
        bleu2 = corpus_bleu(references, hypotheses, weights=weight2)
        weight1 = [1.0]
        bleu1 = corpus_bleu(references, hypotheses, weights=weight1)
        
        print(f'\n * LOSS - {losses.avg:.3f}, TOP-5 ACCURACY - {top5accs.avg:.3f}, BLEU-1 - {bleu1:.3f}, BLEU-2 - {bleu2:.3f}, BLEU-3 - {bleu3:.3f}, BLEU-4 - {bleu4:.3f}\n')

    return bleu4

if __name__ == '__main__':
    main()
