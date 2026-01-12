CUDA_VISIBLE_DEVICES=1 nohup python avr_runner.py &
CUDA_VISIBLE_DEVICES=1 nohup python infer.py --ckpt logs/MeshRIR/Meshrir_sliced_0.1s/ckpts/500000.tar &
