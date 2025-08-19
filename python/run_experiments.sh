flux batch -n 1 -c32 --job-name=VAE_0.5 train.sh graph-dm-vae --ratio 0.01
flux batch -n 1 -c32 --job-name=VAE_0.1 train.sh graph-dm-vae --ratio 0.05
flux batch -n 1 -c32 --job-name=VAE_1.0 train.sh graph-dm-vae --ratio 0.1
flux batch -n 1 -c32 --job-name=VAE_1.5 train.sh graph-dm-vae --ratio 0.2