import torch
import argparse
from pathlib import Path

import model_zoo
import data_utils
import vocab_utils
from decoding import LanguageModelDecoder
from tqdm.auto import tqdm
from pyabc import pyabc


def inference(args):

  path = Path(args.path)
  if path.is_dir():
    yaml_path = list(path.glob('*.yaml'))[0]
    vocab_path = list(path.glob('*vocab.json'))[0]
    checkpoint_list = list(path.glob('*.pt'))
    checkpoint_list.sort(key= lambda x: int(x.stem.split('_')[-2].replace('iter', '')))
    checkpoint_path = checkpoint_list[-1]

    config = data_utils.read_yaml(yaml_path)
    model_name = config.nn_params.model_name
    vocab_name = config.nn_params.vocab_name
    net_param = config.nn_params

    vocab = getattr(vocab_utils, vocab_name)(json_path= vocab_path)
    config = data_utils.get_emb_total_size(config, vocab)
    model = getattr(model_zoo, model_name)(vocab.get_size(), net_param)

    checkpoint = torch.load(checkpoint_path, map_location= 'cpu')

    model.load_state_dict(checkpoint['model'])

  else:
    pass

  model.eval()
  model.to(args.device)
  decoder =  LanguageModelDecoder(vocab, args.save_dir)
  args.save_dir.mkdir(parents=True, exist_ok=True)

  header =  {'key':f'C {args.key_mode}', 'meter':'4/4', 'unit note length':'1/8', 'rhythm':args.rhythm}
  num_generated = 0
  rand_seed = args.seed
  while num_generated < args.num_samples:
    if args.key_mode == 'random':
      if rand_seed % 10 < 7:
        header['key'] = 'C Major'
      elif rand_seed % 3 == 0:
        header['key'] = 'C minor'
      elif rand_seed % 3 < 1:
        header['key'] = 'C Dorian'
      else:
        header['key'] = 'C Mixolydian'

    try:
      out = model.inference(vocab, manual_seed=rand_seed, header = header)
      meta_string = f'X:1\nT:Title\nM:2/2\nL:{header["unit note length"]}\nK:{header["key"]}\n'
      gen_abc = decoder.decode(out, meta_string)
      tune = pyabc.Tune(gen_abc)
      if data_utils.is_good_reel(tune):
        min_pitch, max_pitch = data_utils.get_min_max_pitch(tune)
        # new_key, transpose = get_rand_transpose(header['key'], rand_seed)
        new_key, transpose = get_transpose_by_pitch(header['key'], min_pitch, max_pitch)
        gen_abc = decoder.decode(out, meta_string, transpose=transpose)
        file_name = f'model_{yaml_path.stem}_seed_{rand_seed}_key_{new_key}'
        decoder(gen_abc, file_name, save_image=args.save_image, save_audio=args.save_audio, meta_string=meta_string)
        num_generated += 1
        print(f'generated {num_generated} tunes')
    except Exception as e:
      print(f"decoding failed: {e}")
    rand_seed += 1


def get_transpose_by_pitch(key, min_pitch, max_pitch):
  # key = 'C Major' or 'C minor' or 'C Dorian' or 'C Mixolydian'
  # min_pitch = -5 to 6 (usually)
  # max_pitch = 23 
  if key == 'C Major':
    if min_pitch < -10:
      return 'A Major', 9
    elif max_pitch < 18:
      return 'G Major', 7
    else: # max_pitch >= 17:
      return 'D Major', 2
  elif key == 'C minor':
    if min_pitch < -6:
      return 'A minor', 9
    elif max_pitch < 19:
      return 'E minor', 4
    else:
      return 'G minor', 7
  elif key == 'C Dorian':
    if max_pitch < 16:
      return 'A Dorian', 9
    else:
      return 'D Dorian', 2
  elif key == 'C Mixolydian':
    if min_pitch < -6:
      return 'A Mixolydian', 9
    else:
      return 'D Mixolydian', 2
  else:
    raise ValueError(f'key {key} not recognized')

def get_rand_transpose(key, rand_seed):
  # key = 'C Major' or 'C minor' or 'C Dorian' or 'C Mixolydian'
  if key == 'C Major':
    if rand_seed % 7 < 3:
      return 'G Major', 7
    elif rand_seed % 7 < 6:
      return 'D Major', 2
    else:
      return 'A Major', 9
  elif key == 'C minor':
    if rand_seed % 7 < 4:
      return 'E minor', 4
    elif rand_seed % 7 < 6:
      return 'A minor', 9
    else:
      return 'G minor', 7
  elif key == 'C Dorian':
    if rand_seed % 7 < 6:
      return 'A Dorian', 9
    else:
      return 'D Dorian', 2
  elif key == 'C Mixolydian':
    if rand_seed % 7 < 4:
      return 'D Mixolydian', 2
    else:
      return 'A Mixolydian', 9
  else:
    raise ValueError(f'key {key} not recognized')


def get_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--path', type=Path, default=Path('experiments/pitch_xl_recover'))
  parser.add_argument('--num_samples', type=int, default=10)
  parser.add_argument('--save_dir', type=Path, default=Path('generated'))
  parser.add_argument('--save_audio', action='store_true')
  parser.add_argument('--save_image', action='store_true')
  parser.add_argument('--device', type=str, default='cpu')
  parser.add_argument('--key_mode', type=str, default='random', choices=['random', 'Major', 'minor', 'Dorian', 'Mixolydian'])
  parser.add_argument('--rhythm', type=str, default='reel', choices=['reel', 'jig'])
  parser.add_argument('--seed', type=int, default=4035) # 4035 was the seed for the first-prize winning tune

  return parser


if __name__ == "__main__":
  args = get_parser().parse_args()
  inference(args)