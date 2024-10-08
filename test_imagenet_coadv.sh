python train_imagenet_coadv.py \
   --epoch 100 --learning_rate 0.2 --tmp_data_dir ./data/ImageNet_10 --gpu 0 --batch_size 32 --report_freq 200 --save ./model/coadv \
   --threshold 0.9 --nb_iter 1 --attack_type 'fgsm_back' --back_epoch 50 --back 1 --resume './model/coadveval-try-20241003-133848/checkpoint.pth.tar' #--adv_epoch 300  #--cutout
