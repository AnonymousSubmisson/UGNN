python sat.py \
    --logdir=train \
    --gpu_list=1 \
    --train_steps=10000000 \
    --num_threads=16 \
    --save_data_path= \
    --is_evaluate=True \
    --load_model=trainold/best-45000 \
    --train_data=sattraindata_newgcn \
    --eval_data=../satevaldata_newgcn-40 > curve.40
#train/debug-2201

# --load_model=train/debug-801
