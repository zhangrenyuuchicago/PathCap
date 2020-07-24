for entry in `ls ../../../../slide/`; do
    python cluster_slide_box_tsne.py ../../../../slide/$entry ../../../cluster/train_index_5/ ../result/
done
