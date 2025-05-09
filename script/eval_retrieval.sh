torchrun --nproc_per_node=2 ../Retrieval.py --config ../configs/Retrieval_flickr.yaml --output_dir ../output/Retrieval_flickr --evaluate --checkpoint ../output/Noqueue-2491/15-04-2025:02-17all/checkpoint_best.pth

 # --checkpoint ../pretrain-weight/MSCOCO/checkpoint_best.pth