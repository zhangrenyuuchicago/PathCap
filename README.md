# PathCap

This is the repo for our paper

Evaluating and interpreting caption prediction forhistopathology images

- AE_triplet_loss/ contains codes for training autoencoder with triplet loss.
- att_thumbnail_tiles/ is model PathCap. It uses the thumbnail to init the LSTM and sampled tiles, one from each cluster, for each step of LSTM. Some codes (the decoder part) are borrowed from https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning
- att_tiles/ is model which uses tiles only. 
- cluster/ contains scripts for doing K-Mean clustering.
- data/ preprocesses the images and captions.

