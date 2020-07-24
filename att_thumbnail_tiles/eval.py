import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
from tqdm import tqdm

data_folder = '../data/images/'  
data_name = 'GTEx_1_cap_per_img_5_min_word_freq'  
checkpoint = './BEST_checkpoint_GTEx_1_cap_per_img_5_min_word_freq.pth.tar'  
word_map_file = '../data/images/word_map.json'  
cluster_folder = '/mnt/scratch/renyu/gpu-cluster/ML4HC/NODE3/AAAI_rerun/mu_0.1_margin_0.001_real/cluster_epoch3/train_index_3/'
tile_folder = '/mnt/scratch/renyu/gpu-cluster/ML4HC/NODE2/GTEx_Tiles_jpg/'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
cudnn.benchmark = True  

checkpoint = torch.load(checkpoint)
decoder = checkpoint['decoder']
decoder = decoder.to(device)
decoder.eval()
encoder = checkpoint['encoder']
encoder = encoder.to(device)
encoder.eval()
tile_encoder = checkpoint['tile_encoder']
tile_encoder = tile_encoder.to(device)
tile_encoder.eval()

with open(word_map_file, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}
vocab_size = len(word_map)

test_transform = transforms.Compose([
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
test_tile_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])

def evaluate(beam_size, test_index):
    loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, 'test', test_transform, test_tile_transform, tile_folder=tile_folder, cluster_folder=cluster_folder),
        batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    references = list()
    hypotheses = list()
    references_words = list()
    hypotheses_words = list()
    for i, (image, caps, caplens, allcaps, tiles, slide_id, proportions) in enumerate(
            tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):
        k = beam_size
        fout = open('result/' + slide_id[0] + '.txt', 'w')
        image = image.to(device) 
        tiles = tiles.to(device)
        proportions = proportions.to(device)
        encoder_out = encoder(image)  
        tile_encoder_out = tile_encoder(tiles, proportions)
        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)

        encoder_out = encoder_out.view(1, -1, encoder_dim)  
        num_pixels = encoder_out.size(1)

        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

        k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        h, c = decoder.init_hidden_state(encoder_out)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:

            embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

            awe, alpha = decoder.attention(tile_encoder_out, h)  # (s, encoder_dim), (s, num_pixels) 
            alpha = alpha.data.cpu().tolist()
            fout.write(str(alpha) + '\n')

            gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
            awe = gate * awe

            h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

            scores = decoder.fc(h)  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=1)

            # Add
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words / vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)

            # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != word_map['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            tile_encoder_out = tile_encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > 50:
                break
            step += 1
        
        #print('complete seqs')
        #print(complete_seqs_scores)
        if len(complete_seqs_scores) == 0:
            continue

        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]

        # References
        img_caps = allcaps[0].tolist()
        
        img_captions = list(
            map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                img_caps))  # remove <start> and pads

        references.append(img_captions)

        # Hypotheses
        hypotheses.append([w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])
        #hypotheses.append([w for w in seq if w not in {word_map['<start>'], word_map['<end>']}])

        assert len(references) == len(hypotheses)
        hypotheses_word = [rev_word_map[word] for word in hypotheses[-1]]
        fout.write('hyposthese:\n')
        fout.write(' '.join(hypotheses_word) + '\n')
        references_word = [rev_word_map[word] for word in img_captions[0]]
        fout.write('reference:\n')
        fout.write(' '.join(references_word) + '\n')
        fout.close()

        hypotheses_words.append(' '.join(hypotheses_word))
        references_words.append([' '.join(references_word)])
    
    
    fout_hyp = open(f'gen_hyp/hyp_{test_index}.txt', 'w')
    fout_ref = open(f'gen_hyp/ref_{test_index}.txt', 'w')
    for i in range(len(hypotheses)):
        h = hypotheses_words[i]
        r = references_words[i]
        fout_hyp.write(h + '\n')
        fout_ref.write(r[0] + '\n')
    
    fout_hyp.close()
    fout_ref.close()
    
    # Calculate BLEU-4 scores
    bleu4 = corpus_bleu(references, hypotheses)
    weight3 = (0.333, 0.333, 0.333)
    bleu3 = corpus_bleu(references, hypotheses, weights=weight3)
    weight2 = (0.5, 0.5)
    bleu2 = corpus_bleu(references, hypotheses, weights=weight2)
    weight1 = [1.0]
    bleu1 = corpus_bleu(references, hypotheses, weights=weight1)

    return bleu1, bleu2, bleu3, bleu4


if __name__ == '__main__':
    beam_size = 1 
    for test_index in range(20):
        bleu = evaluate(beam_size, test_index)
        #print("\nBLEU-4 score @ beam size of %d is %.4f." % (beam_size, evaluate(beam_size)))
        print(f"\nBLEU-1: {bleu[0]}, BLEU-2: {bleu[1]}, BLEU-3: {bleu[2]}, BLEU-4: {bleu[3]}")
