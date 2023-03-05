import torch
import shutil
from pathlib import Path
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from data_utils import ABCset, MeasureNumberSet, pack_collate, PitchDurSplitSet, FolkRNNSet, MeasureOffsetSet, read_yaml, MeasureEndSet
from loss import get_nll_loss

import data_utils
import model_zoo

from trainer import Trainer, TrainerMeasure, TrainerPitchDur
import argparse
import wandb
import datetime


def get_argument_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--path', type=str, default='abc_dataset/folk_rnn_abc_key_cleaned/',
                      help='directory path to the dataset')
  parser.add_argument('--yml_path', type=str, default='yamls/measure_note_xl.yaml',
                      help='yaml path to the config')

  parser.add_argument('--batch_size', type=int, default=32)
  parser.add_argument('--num_iter', type=int, default=100000)
  parser.add_argument('--lr', type=float, default=0.0003)
  parser.add_argument('--scheduler_factor', type=float, default=0.3)
  parser.add_argument('--scheduler_patience', type=int, default=3)
  parser.add_argument('--grad_clip', type=float, default=1.0)
  parser.add_argument('--aug_type', type=str, default='stat')

  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument('--num_workers', type=int, default=2)
  parser.add_argument('--device', type=str, default='cuda')

  parser.add_argument('--num_epoch_per_log', type=int, default=2)
  parser.add_argument('--num_iter_per_valid', type=int, default=5000)

  parser.add_argument('--model_type', type=str, default='pitch_dur')
  parser.add_argument('--model_name', type=str, default='pitch_dur')
  parser.add_argument('--save_dir', type=Path, default=Path('experiments/'))

  parser.add_argument('--no_log', action='store_true')

  return parser

def make_experiment_name_with_date(args, net_param):
  current_time_in_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  return f'{current_time_in_str}-{net_param.model_name}_{net_param.note.hidden_size}_{args.batch_size}_{args.num_iter}_{args.lr}'

if __name__ == '__main__':
  args = get_argument_parser().parse_args()
  torch.manual_seed(args.seed)

  config = read_yaml(args.yml_path)
  net_param = config.nn_params
  data_param = config.data_params
  model_name = net_param.model_name
  if hasattr(net_param, 'dataset_name'):
    dataset_name = net_param.dataset_name
  elif model_name == "PitchDurModel":
    dataset_name = "PitchDurSplitSet"
  elif model_name in ["MeasureHierarchyModel", "MeasureNoteModel", "MeasureNotePitchFirstModel"]:
    dataset_name = "MeasureNumberSet"
  elif model_name == "MeasureInfoModel":
    dataset_name = "MeasureOffsetSet"
  elif model_name == "LanguageModel":
    dataset_name = ABCset
  else:
    raise NotImplementedError

  if 'folk_rnn/data_v3' in args.path:
    score_dir = FolkRNNSet(args.path)
    vocab_path = Path(args.path).parent /  f'{args.model_type}_vocab.json'
  else:
    score_dir = Path(args.path)
    vocab_path = Path(args.path) / f'{args.model_type}_vocab.json'

  dataset = getattr(data_utils, dataset_name)(score_dir, vocab_path, key_aug=data_param.key_aug, vocab_name=net_param.vocab_name)
  config = data_utils.get_emb_total_size(config, dataset.vocab)
  net_param = config.nn_params
  model = getattr(model_zoo, model_name)(dataset.vocab.get_size(), net_param)

  print(f'Vocab size: {dataset.vocab.get_size()}')
  print(f'Number of data: {len(dataset)}')


  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
  # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=args.scheduler_factor, patience=args.scheduler_patience, verbose=True)
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.5)
  loss_fn = get_nll_loss

  if isinstance(dataset, MeasureNumberSet) or isinstance(dataset, MeasureEndSet):
    trainset = dataset.get_trainset()
    validset = dataset.get_testset()
  else:
    trainset, validset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.9), len(dataset) - int(len(dataset)*0.9)], generator=torch.Generator().manual_seed(42))

  print(f'Number of train data: {len(trainset)}')
  print(f'Number of valid data: {len(validset)}')

  train_loader = DataLoader(trainset, batch_size=args.batch_size, collate_fn=pack_collate, shuffle=True) #collate_fn=pack_collate)
  valid_loader = DataLoader(validset, batch_size=args.batch_size, collate_fn=pack_collate, shuffle=False) #collate_fn=pack_collate)


  experiment_name = make_experiment_name_with_date(args, net_param)
  save_dir = args.save_dir / experiment_name
  save_dir.mkdir(parents=True, exist_ok=True)
  args.save_dir = save_dir
  shutil.copy(args.yml_path, save_dir)

  if model_name in ["PitchDurModel", "MeasureInfoModel"]:
    trainer = TrainerPitchDur(model, optimizer, scheduler, loss_fn, train_loader, valid_loader, args)
  elif model_name in ["MeasureHierarchyModel", "MeasureNoteModel", "MeasureNotePitchFirstModel"]:
    trainer = TrainerMeasure(model, optimizer, scheduler, loss_fn, train_loader, valid_loader, args)
  else:
    trainer = Trainer(model, optimizer, scheduler, loss_fn, train_loader, valid_loader, args)
  

  if not args.no_log:
    wandb.init(project="irish-maler", entity="maler", config={**vars(args), **config})
    # wandb.config.update({**vars(args), **config})
    wandb.watch(model)

  trainer.train_by_num_iter(args.num_iter)