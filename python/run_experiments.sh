# flux batch -n 1 -c32 --job-name=VAE_EPS_0.5 train.sh graph-dm-vae-eps --ratio 0.01
# flux batch -n 1 -c32 --job-name=VAE_EPS_0.1 train.sh graph-dm-vae-eps --ratio 0.05
# flux batch -n 1 -c32 --job-name=VAE_EPS_1.0 train.sh graph-dm-vae-eps --ratio 0.1
# flux batch -n 1 -c32 --job-name=VAE_EPS_1.5 train.sh graph-dm-vae-eps --ratio 0.2



# flux batch -n 1 -c32 --job-name=CLS_hb_0.001 train_devmap_cls.sh dmcls-graphhb --learning_rate 0.001 --weight_decay 0.0001 # f4dEkeepE8UF
# flux batch -n 1 -c32 --job-name=CLS_hb_0.01  train_devmap_cls.sh dmcls-graphhb --learning_rate 0.01  --weight_decay 0.001  # f4dEkekkBFrF
# flux batch -n 1 -c32 --job-name=CLS_hb_0.1   train_devmap_cls.sh dmcls-graphhb --learning_rate 0.1   --weight_decay 0.01   # f4dEkerdAQfZ

# flux batch -n 1 -c32 --job-name=CLS_dm_0.001 train_devmap_cls.sh dmcls-graphdm --learning_rate 0.001 --weight_decay 0.0001 # f4dEkexW9ZUs
# flux batch -n 1 -c32 --job-name=CLS_dm_0.01  train_devmap_cls.sh dmcls-graphdm --learning_rate 0.01  --weight_decay 0.001  # f4dEkf4LAjjV
# flux batch -n 1 -c32 --job-name=CLS_dm_0.1   train_devmap_cls.sh dmcls-graphdm --learning_rate 0.1   --weight_decay 0.01   # f4dEkfABfuGT

# flux batch -n 1 -c32 --job-name=VAE_EPS_0.0 train.sh graph-dm-vae-eps --ratio 0.0 # f4dKGbAyYrEs

# flux batch -n 1 -c32 --job-name=CLS_dm_e10000 train_devmap_cls.sh dmcls-graphdm --epochs 10000 # f4dKJ7vDfMno
# flux batch -n 1 -c32 --job-name=CLS_hb_e10000 train_devmap_cls.sh dmcls-graphhb --epochs 10000 # f4dKJ826eWc7

# flux batch -n 1 -c32 --job-name=CLS_dm_e10000 train_devmap_cls.sh dmcls-graphdm --epochs 10000 --train_from_checkpoint true # f4dZ79AwxWFh
# flux batch -n 1 -c32 --job-name=CLS_hb_e10000 train_devmap_cls.sh dmcls-graphhb --epochs 10000 --train_from_checkpoint true # f4dZ79GvscCP

# flux batch -n 1 -c32 --job-name=VAE_EPS_dm train.sh graph-dm-vae-eps # f4df6QEruYPR
# flux batch -n 1 -c32 --job-name=VAE_EPS_hb train.sh graph-hb-vae-eps # f4dfAUrGdUpX

# flux batch -n 1 -c32 --job-name=CLS_dm_e10000 train_devmap_cls.sh dmcls-graphdm --epochs 10000 --train_from_checkpoint true # f4djozrHk6HM
# flux batch -n 1 -c32 --job-name=CLS_hb_e10000 train_devmap_cls.sh dmcls-graphhb --epochs 10000 --train_from_checkpoint true # f4djozxAjF6f

# flux batch -n 1 -c32 --job-name=CLS_default train_vecparams_cls.sh vpcls-graphhb # f4f6JnsfqFUB
# flux batch -n 1 -c32 --job-name=E2E_default train_vecparams_e2e.sh vpe2e # f4f6PagtdVCb
# flux batch -n 1 -c32 --job-name=E2E_default train_vecparams_e2e.sh vpe2e --train_from_checkpoint true # f4fJJT13X9Tm
# flux batch -n 1 -c32 --job-name=E2E_default train_vecparams_e2e.sh vpe2e --train_from_checkpoint true # f4fQx94AUCwH
# flux batch -n 1 -c32 --job-name=E2E_O0 train_vecparams_e2e.sh vpe2e-O0 # f4fbMGG6nrtP
# flux batch -n 1 -c32 --job-name=E2E_O3 train_vecparams_e2e.sh vpe2e-O3 # f4fbMGN2jzGP
# flux batch -n 1 -c32 --job-name=E2E_O0 train_vecparams_e2e.sh vpe2e-O0 --train_from_checkpoint true # f4fjovDQHn9m
# flux batch -n 1 -c32 --job-name=E2E_O3 train_vecparams_e2e.sh vpe2e-O3 --train_from_checkpoint true # f4fjovKJkvFR

# flux batch -n 1 -c32 --job-name=VSE2E_O0 train_vecparams_e2e.sh vpe2e-O0-wgt # f4fwsvA22Ref
# flux batch -n 1 -c32 --job-name=VSE2E_O3 train_vecparams_e2e.sh vpe2e-O3-wgt # f4fwsvFvVZkK

# flux batch -n 1 -c32 --job-name=CLS_hb_e10000 train_devmap_cls.sh dmcls-amd-graphhb --epochs 4000 # f4gHkvYfQ5V1
# flux batch -n 1 -c32 --job-name=CLS_hb_e10000 train_devmap_cls.sh dmcls-graphhb --epochs 4000 # f4gHkveTwGTH
# flux batch -n 1 -c32 --job-name=CLS_hb_e10000 --dependency=afterany:f4g8xQ19mZ9H train_devmap_cls.sh dmcls-amd-graphhb --epochs 10000 --train_from_checkpoint true # f4g2qv2biakw
# flux batch -n 1 -c32 --job-name=CLS_hb_e10000 --dependency=afterany:f4g8xQ19mZ9H train_devmap_cls.sh dmcls-graphhb --epochs 10000 --train_from_checkpoint true # f4dZ79GvscCP

flux batch -n 1 -c32 --job-name=VPCLS_e4000 train_vecparams_cls.sh vpcls-graphhb-O0 --epochs 4000 # f4gJF7PeBjuq
flux batch -n 1 -c32 --job-name=VPCLS_e4000 train_vecparams_cls.sh vpcls-graphhb-O3 --epochs 4000 # f4gJF7VREwbm
