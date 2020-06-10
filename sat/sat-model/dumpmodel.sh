


python sat.py \
    --gpu_list=0 \
    --load_model=$1 \
    --train_steps=0 \
    --num_threads=16 \
    --save_data_path= \
    --train_data=/home/zhangwj/data/sattraindata \
    --eval_data=/home/zhangwj/data/satevaldata_newgcn \
    --is_evaluate=True \
    --logdir=evallog \
    --dump_model=True
#train/debug-2201

# --load_model=train/debug-801
