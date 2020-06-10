


python sat.py \
    --logdir=train \
    --gpu_list=0 \
    --train_steps=10000000 \
    --num_threads=16 \
    --save_data_path= \
    --train_data=sattraindata_newgcn-small \
    --eval_data=satevaldata_newgcn-small \
	--train_length=1000 
	#--load_model=train/debug-39000
	#train/debug-2201

# --load_model=train/debug-801
