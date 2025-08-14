flux batch -n 1 -c32 train.sh graph-dm-vae-eps --loss_betas "[0.5,0.5]"
flux batch -n 1 -c32 train.sh graph-dm-vae-eps --loss_betas "[0.1,0.1]"
flux batch -n 1 -c32 train.sh graph-dm-vae-eps --loss_betas "[1.0,1.0]"
flux batch -n 1 -c32 train.sh graph-dm-vae-eps --loss_betas "[1.5,1.5]"