import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import torch
import torch.optim as optim
import torch.nn as nn
import torch.distributed as dist
import argparse
import time
from tensorboardX import SummaryWriter
import config
from tools.trainer import Trainer
from checkpoints import CheckpointIO
import pickle
import random
from torch.optim import lr_scheduler
from timm.scheduler import CosineLRScheduler

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main(): 
	# Arguments
	cfg = config.load_config('cfgs/setting.yaml')
	#is_cuda = (torch.cuda.is_available() )
	
	device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
	print(device)

	##### DDP
	#parser = argparse.ArgumentParser() 
	#parser.add_argument('--local_rank', default=-1, type=int,
	#					help='node rank for distributed training')
	#args = parser.parse_args()

	#dist.init_process_group(backend='nccl')
	#torch.cuda.set_device(args.local_rank)
	#####

	# Set t0
	t0 = time.time()

	# Shorthands
	out_dir = 'out'
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	logfile = open('out/log.txt','a')
	batch_size=cfg['training']['batch_size']

	model = config.get_model(cfg).cuda()
	device_ids = [0,1,2,3]
	if torch.cuda.device_count() > 1:
		print(f"Let's use {torch.cuda.device_count()} GPUs!")
		model = nn.DataParallel(model, device_ids=device_ids)
	optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)

	scheduler = CosineLRScheduler(optimizer,
							   	t_initial=6000,
								cycle_mul=1,
                               lr_min=1e-5,
                                cycle_decay=0.5,
                                warmup_lr_init=1e-5,
                                warmup_t=50,
                                cycle_limit=1,
                                t_in_epochs=True)
	

	trainer = Trainer(model, device, optimizer)

	checkpoint_io = CheckpointIO(out_dir, model=model, optimizer=optimizer)

	try:
		load_dict = checkpoint_io.load('model.pt')
	except FileExistsError:
		load_dict = dict()
	epoch_it = load_dict.get('epoch_it', -1)
	it = load_dict.get('it', -1)
	metric_val_best = np.inf


	logger = SummaryWriter(os.path.join(out_dir, 'logs'))
	logger_val = SummaryWriter(os.path.join(out_dir, 'vallogs'))

	# Shorthands
	nparameters = sum(p.numel() for p in model.parameters())

	logfile.write('Total number of parameters: %d' % nparameters)

	print_every = cfg['training']['print_every']
	checkpoint_every = cfg['training']['checkpoint_every']
	validate_every = cfg['training']['validate_every']

	train_dataset = config.get_dataset('train', cfg)
	val_dataset = config.get_dataset('val', cfg)

	train_loader = torch.utils.data.DataLoader(
		train_dataset, batch_size=batch_size, num_workers=4, shuffle=True 
	)


	val_loader = torch.utils.data.DataLoader(
		val_dataset, batch_size=batch_size, num_workers=4, shuffle=True  
	)
	
	while True:
		epoch_it += 1
		logfile.flush()

		if epoch_it>cfg['training']['epoch']:
			logfile.close()
			break
		for batch in train_loader:
			it += 1

			loss = trainer.train_step(batch)
			logger.add_scalar('total loss', loss, it)
			
			if print_every > 0 and (it % print_every) == 0 and it > 0 :
				logfile.write('[Epoch %02d] it=%03d, loss=%.6f\n'
					  % (epoch_it, it, loss))
				print('[Epoch %02d] it=%03d, loss=%.6f'
					  % (epoch_it, it, loss))

			# Save checkpoint
			if (checkpoint_every > 0 and (it % checkpoint_every) == 0) and it > 0 :
				logfile.write('Saving checkpoint')
				checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
								   loss_val_best=metric_val_best)

			# Run validation
			if validate_every > 0 and (it % validate_every) == 0 and it > 0 :
				metric_val = trainer.evaluate(val_loader)
				# logger_val.add_scalar('total loss', metric_val, epoch_it)
				logfile.write('Validation metric : %.6f\n'
					  % (metric_val))
				if metric_val < metric_val_best:
					metric_val_best = metric_val
					logfile.write('New best model (loss %.6f)\n' % metric_val_best)
					checkpoint_io.save('model_best.pt', epoch_it=epoch_it, it=it,
									   loss_val_best=metric_val_best)
					
		scheduler.step(epoch_it)
		print(optimizer.state_dict()['param_groups'][0]['lr'])

	logger.close()

if __name__ == '__main__':
	set_seed(2024)
	main()