torchrun --nproc_per_node=1 ../Retrieval_origin.py --config ../configs/Retrieval_coco.yaml --output_dir ../output/Retrieval_coco --evaluate --checkpoint ../output/Noqueue/ALBEF-Original/checkpoint_best.pth 
# --evaluate
# --checkpoint ../pretrain-weight/ALBEF_4M.pth
# --checkpoint ../pretrain-weight/MSCOCO/checkpoint_best.pth
