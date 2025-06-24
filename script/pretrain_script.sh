torchrun --nnodes=1 --nproc_per_node=4 ../Pretrain.py \
    --config ../configs/Pretrain.yaml \
    --output_dir ../output/Noqueue-42 \
    --pos all
