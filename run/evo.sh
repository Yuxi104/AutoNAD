export CUDA_VISIBLE_DEVICES=0,1

# ======================= For neu =======================
python -m torch.distributed.launch --nproc_per_node=2 --use_env ../evolution.py --data-path /your/path/to/NEU --gp --change_qk --relative_position --dist-eval --cfg ../config/supernet.yaml \
--resume ../output/neu/checkpoint_tiny.pth --min-param-limits 1 --param-limits 10 --output_dir ../output/neu --batch-size 16 --nb-classes 3 --dataset neu

# ======================= For mt =======================
# python -m torch.distributed.launch --nproc_per_node=2 --use_env ../evolution.py --data-path /your/path/to/MT --gp --change_qk --relative_position --dist-eval --cfg ../config/supernet.yaml \
# --resume ../output/mt/checkpoint_tiny.pth --min-param-limits 1 --param-limits 10 --output_dir ../output/mt --batch-size 8 --nb-classes 5 --dataset mt

# ======================= For msd =======================
# python -m torch.distributed.launch --nproc_per_node=2 --use_env ../evolution.py --data-path /your/path/to/MSD --gp --change_qk --relative_position --dist-eval --cfg ../config/supernet.yaml \
# --resume ../output/msd/checkpoint_tiny.pth --min-param-limits 1 --param-limits 10 --output_dir ../output/msd --batch-size 4 --nb-classes 3 --dataset msd