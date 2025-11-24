# ======================= For neu =======================
python ../retrain.py --output-dir ../output/neu/retrain --dataset neu --lr 1e-4 \
--num-classes 3 \
--epoch 500 \
--device 'cuda:0' \
--cfg ../config/supernet.yaml \
--batch-size 8 \
--data-path /media/data3/lyx/DEFECT-VOC/NEU \
--subnet ../output/neu/evo/best_arch.json \
--resume ../output/neu/retrain/neu.pth \
--test


# ======================= For mt =======================
# python ../retrain.py --output-dir ../output/mt/retrain --dataset mt --lr 1e-4 \
# --num-classes 5 \
# --epoch 500 \
# --device 'cuda:0' \
# --cfg ../config/supernet.yaml \
# --batch-size 8 \
# --data-path /media/data3/lyx/DEFECT-VOC/MT \
# --subnet ../output/mt/evo/best_arch.json \
# --resume ../output/mt/retrain/mt.pth \
# --test


# ======================= For msd =======================
# python ../retrain.py --output-dir ../output/msd/retrain --dataset msd --lr 1e-4 \
# --num-classes 3 \
# --epoch 500 \
# --device 'cuda:0' \
# --cfg ../config/supernet.yaml \
# --batch-size 8 \
# --data-path /media/data3/lyx/DEFECT-VOC/MSD \
# --subnet ../output/msd/evo/best_arch.json \
# --resume ../output/msd/retrain/msd.pth \
# --test