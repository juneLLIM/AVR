CUDA_VISIBLE_DEVICES=1 nohup python avr_runner.py --config config_files/avr_raf_empty.yml --dataset_dir ~/data/dataset/raf_dataset/EmptyRoom > /dev/null &
CUDA_VISIBLE_DEVICES=1 nohup python avr_runner.py --config config_files/avr_raf_furnished.yml --dataset_dir ~/data/dataset/raf_dataset/FurnishedRoom > /dev/null &
CUDA_VISIBLE_DEVICES=1 nohup python infer.py --ckpt logs/MeshRIR/Meshrir_sliced_0.1s/ckpts/500000.tar &
tensorboard --logdir tensorboard_logs