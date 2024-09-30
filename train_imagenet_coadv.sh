python train_imagenet_coadv.py \
   --epoch 250 --learning_rate 0.2 --tmp_data_dir ./data/ImageNet_50 --gpu 0,1,2 --batch_size 256  --report_freq 200 --save ./model/coadv \
   --threshold 0.9 --nb_iter 1 --attack_type 'fgsm_back' --back_epoch 50 --back 1 # --resume './weight/imgeval-try-20210423-112010/checkpoint.pth.tar' #--adv_epoch 300  #--cutout
