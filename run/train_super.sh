export CUDA_VISIBLE_DEVICES=0,1

# ======================= For neu =======================
python -m torch.distributed.launch --nproc_per_node=2 --use_env ../supernet_train.py --data-path /your/path/to/NEU --gp --change_qk --relative_position --mode super --dist-eval --cfg ../config/supernet.yaml --epochs 800 --warmup-epochs 20 --lr 5e-4 --output ../output/neu --batch-size 8 --nb-classes 3 --dataset neu

# ======================= For mt =======================
# python -m torch.distributed.launch --nproc_per_node=2 --use_env ../supernet_train.py --data-path /your/path/to/MT --gp --change_qk --relative_position --mode super --dist-eval --cfg ../config/supernet.yaml --epochs 3000 --warmup-epochs 20 --lr 5e-4 --output ../output/mt --batch-size 4 --nb-classes 5 --dataset mt

# ======================= For msd =======================
# python -m torch.distributed.launch --nproc_per_node=2 --use_env ../supernet_train.py --data-path /your/path/to/MSD --gp --change_qk --relative_position --mode super --dist-eval --cfg ../config/supernet.yaml --epochs 2000 --warmup-epochs 20 --lr 5e-4 --output ../output/msd --batch-size 4 --nb-classes 3 --dataset msd