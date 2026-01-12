
import os
import argparse
import time
from pathlib import Path
import yaml
import torch
from tqdm import tqdm
from avr_runner import AVR_Runner
from utils.logger import plot_and_save_figure, log_inference_figure

class InferRunner(AVR_Runner):
    """
    Runner for inference that loads a specific checkpoint
    """
    def load_specific_checkpoint(self, ckpt_path):
        self.logger.info('Loading specific ckpt %s', ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=self.devices)

        try: 
            self.renderer.load_state_dict(ckpt['audionerf_network_state_dict'])
        except: 
            # If the model was saved with DataParallel, the keys have 'module.' prefix
            # If current model is not DataParallel (or vice versa), we might need to adjust
            self.renderer.module.load_state_dict(ckpt["audionerf_network_state_dict"])
            
        # We don't necessarily need to load optimizer/scheduler/iteration for inference
        # self.current_iteration = ckpt.get('current_iteration', 0)
        self.logger.info("Checkpoint loaded successfully.")

    def inference(self, output_dir):
        self.logger.info("Start inference")
        self.renderer.eval()

        valid_losses = {'spec_loss': 0, 'fft_loss': 0, 'time_loss': 0, 'energy_loss': 0, 'multi_stft_loss': 0}
        valid_metrics = {'Angle': 0, 'Amplitude': 0, 'Envelope': 0, 'T60': 0, 'C50': 0, 'EDT': 0, 'multi_stft': 0}

        os.makedirs(output_dir, exist_ok=True)
        img_test_dir = os.path.join(output_dir, 'img_test')
        os.makedirs(img_test_dir, exist_ok=True)

        total_time = 0
        first_iter_time = 0

        for check_idx, test_batch in enumerate(tqdm(self.test_iter, desc="Running Inference")):
            start_time = time.time()
            with torch.no_grad():
                if self.dataset_type == "RAF":
                    ori_sig, position_rx, position_tx, direction_tx = test_batch
                    pred_sig = self.renderer(position_rx.cuda(), position_tx.cuda(), direction_tx.cuda())
                else:
                    ori_sig, position_rx, position_tx = test_batch
                    pred_sig = self.renderer(position_rx.cuda(), position_tx.cuda())
                                                                
                pred_sig = pred_sig[...,0] + 1j * pred_sig[...,1]
                ori_sig = (ori_sig.cuda()).to(pred_sig.dtype)

                losses, metrics, ori_time, pred_time = self.calculate_metrics(pred_sig, ori_sig, self.fs)

            batch_time = time.time() - start_time
            total_time += batch_time
            if check_idx == 0:
                first_iter_time = batch_time

            for key in valid_losses:
                valid_losses[key] += losses[key].detach()

            for key in valid_metrics:
                valid_metrics[key] += metrics[key]

            # Save visualization for the first few batches
            if check_idx < 15: # Arbitrary limit to avoid saving too many images
                save_path = os.path.join(img_test_dir, f'{str(check_idx).zfill(5)}.png')
                plot_and_save_figure(pred_sig[0,:], ori_sig[0,:], pred_time[0,:], ori_time[0,:], position_rx[0,:], position_tx[0,:], mode_set='test', save_path=save_path)
            
                save_path_energy = os.path.join(img_test_dir, f'energy_{str(check_idx).zfill(5)}.png')      
                log_inference_figure(ori_time.detach().cpu().numpy()[0,:], pred_time.detach().cpu().numpy()[0,:], metrics=metrics, save_dir=save_path_energy)

        num_batches = len(self.test_iter)
        avg_losses = {key: valid_losses[key] / num_batches for key in valid_losses}
        avg_metrics = {key: valid_metrics[key] / num_batches for key in valid_metrics}

        self.logger.info("Inference finished. Metrics:")
        self.logger.info('Angle:{:.3f}, Amplitude:{:.4f}, Envelope:{:.4f}, T60:{:.5f}, C50:{:.5f}, EDT:{:.5f}, multi_stft:{:.4f}'.format( \
        avg_metrics['Angle'], avg_metrics['Amplitude'], avg_metrics['Envelope'], avg_metrics['T60'], avg_metrics['C50'], avg_metrics['EDT'], avg_metrics['multi_stft']))
        
        # Save metrics to a file
        with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
            for k, v in avg_metrics.items():
                f.write(f"{k}: {v}\n")
        
        # Print results
        print("[Inference results]")
        final_metric = avg_metrics
        print("   ".join([f"{k}: {v:.3f}" for k, v in final_metric.items()]))
        print()
        print("[Inference time]")
        print(f"   {len(self.test_set)} samples generated in {total_time * 1000:.2f} ms.")
        print(
            f"   First iteration time:\t{first_iter_time * 1000:.2f} ms.")
        print("[Including first]")
        print(
            f"   Average time per sample:\t{total_time / len(self.test_set) * 1000:.2f} ms.")
        print(
            f"   Average time per iteration:\t{total_time / len(self.test_iter) * 1000:.2f} ms.")
        if len(self.test_iter) > 1:
            time_wo_first = total_time - first_iter_time
            print("[Excluding first]")
            print(
                f"   Average time per sample:\t{time_wo_first / (len(self.test_set) - self.batch_size) * 1000:.2f} ms.")
            print(
                f"   Average time per iteration:\t{time_wo_first / (len(self.test_iter) - 1) * 1000:.2f} ms.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config_files/avr_meshrir.yml', help='config file path')
    parser.add_argument('--ckpt', type=str, required=True, help='checkpoint path')
    parser.add_argument('--dataset_dir', type=str, default='~/data/dataset/MeshRIR/S1-M3969', help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save results. If not provided, uses logdir/inference')

    args = parser.parse_args()
    
    # Load configuration
    if args.ckpt is not None:
        config = Path(args.ckpt).parent / 'avr_conf.yml'
        print(config)
        if config.exists():
            args.config = str(config)
    print(args.config)
    with open(args.config, 'r') as file:
        kwargs = yaml.load(file, Loader=yaml.FullLoader)
    
    # Load dataset directory
    args.dataset_dir = os.path.expanduser(args.dataset_dir)
    
    # Disable auto-loading of checkpoints by AVR_Runner
    if 'train' in kwargs:
        kwargs['train']['load_ckpt'] = False

    # Initialize runner
    # We use mode='test' to avoid creating training tensorboard logs
    runner = InferRunner(mode='test', dataset_dir=args.dataset_dir, batchsize=args.batch_size, **kwargs)
    
    # Load specific checkpoint
    runner.load_specific_checkpoint(args.ckpt)
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Default to a folder inside the experiment log directory
        output_dir = os.path.join(runner.logdir, runner.expname, 'inference')
    
    # Run inference
    runner.inference(output_dir)
