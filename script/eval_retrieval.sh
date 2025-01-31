torchrun --nproc_per_node=1 ../Retrieval.py --config ../configs/Retrieval_coco.yaml --output_dir ../output/Retrieval_coco
--evaluate true
--checkpoint ../output/Pretrain/30-01-2025:06-14NOUN/checkpoint_best.pth