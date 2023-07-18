# Getting Started

we visualize our training details via wandb (https://wandb.ai/site).

## visualization

1) you'll need to login
   ```shell 
   $ wandb login
   ```
2) you'll need to copy & paste you API key in terminal
   ```shell
   $ https://wandb.ai/authorize
   ```
   or add the key to the "code/config/config.py" with
   ```shell
   C.wandb_key = ""
   ```

## training

our code is trained using one nvidia A6000, but our code also supports distributed data parallel mode in pytorch. We
set batch_size=8 for all the experiments, with learning rate 7.5e-6 and 700 * 700 resolution.

### checkpoints

we follow [Meta-OoD](https://github.com/robin-chan/meta-ood) and use the deeplabv3+ checkpoint
in [here](https://github.com/NVIDIA/semantic-segmentation/tree/sdcnet). you'll need to put it in "ckpts/pretrained_ckpts" directory, and
**please note that downloading the checkpoint before running the code is necessary for our approach.**

for training, simply execute

```shell 
$ python rpl_corocl.code/main.py 
```

## inference

please download our checkpoint
from [here](https://drive.google.com/drive/folders/1rVaBRdOpS2JkAo-ZRO64jSZU0VbdZsDn?usp=sharing) and specify the
checkpoint path for **rpl_corocl_weight_path** in config.py.

```shell
- Example execution code
$ python gmlv_mae_training.py --data_path '/path/to/data'

- Specific execution code with 2048 unit
$ python gmlv_mae_training.py --batch_size 256 --epochs 300 --input_length 2048 --patch_size 32 --embed_dim 32 --dec_dim 32 --mask_ratio 0.75 --enc_dep 8 --dec_dep 4 --enc_head 8 --dec_head 4 --lr 1e-3 --min_lr 0. --warmup_epochs 40 --weight_decay 0.05 --cpt_dir './gmlv_cpt' --name 'gmlv_test_neg_u' --data_path '/path/to/data' --device 'cuda' --num_workers 10 

```
