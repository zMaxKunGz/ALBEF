torchrun --nproc_per_node=2 ../Retrieval.py --config ../configs/Retrieval_coco.yaml --output_dir ../output/Retrieval_coco --evaluate --checkpoint ../output/Noqueue/05-02-2025:17-38NOUN/checkpoint_best.pth 

 # --checkpoint ../pretrain-weight/MSCOCO/checkpoint_best.pth