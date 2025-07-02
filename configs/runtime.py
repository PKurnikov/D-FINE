print_freq=1000
output_dir='./logs'
checkpoint_freq=1

sync_bn=True
find_unused_parameters=True #False

scaler=dict(
  type="GradScaler",
  enabled=True
)

use_wand=False
project_name='D-FINE' # for wandb
exp_name='baseline' # wandb experiment name

device='cuda'