# ======================= For neu =======================
python ../retrain.py --output-dir ../output/neu/retrain --dataset neu --lr 1e-4 \
--num-classes 3 \
--epoch 500 \
--device 'cuda:0' \
--cfg ../config/supernet.yaml \
--batch-size 8 \
--data-path /media/data3/lyx/DEFECT-VOC/NEU \
--resume ../output/neu/checkpoint_tiny.pth \
--subnet ../output/neu/evo/best_arch.json \
--aug


# ======================= For mt =======================
# python ../retrain.py --output-dir ../output/mt/retrain --dataset mt --lr 1e-4 \
# --num-classes 5 \
# --epoch 500 \
# --device 'cuda:0' \
# --cfg ../config/supernet.yaml \
# --batch-size 8 \
# --data-path /media/data3/lyx/DEFECT-VOC/MT \
# --resume ../output/mt/checkpoint_tiny.pth \
# --subnet ../output/mt/evo/best_arch.json \
# --aug


# ======================= For msd =======================
# Note: we do not use --aug here. `--aug` represents the use of extra augmentation.
# python ../retrain.py --output-dir ../output/msd/retrain --dataset msd --lr 1e-4 \
# --num-classes 3 \
# --epoch 500 \
# --device 'cuda:0' \
# --cfg ../config/supernet.yaml \
# --batch-size 8 \
# --data-path /media/data3/lyx/DEFECT-VOC/MSD \
# --resume ../output/msd/checkpoint_tiny.pth \
# --subnet ../output/msd/evo/best_arch.json \
