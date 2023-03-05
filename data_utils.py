from collections import defaultdict
from pathlib import Path
import random

import torch
from torch.nn.utils.rnn import pack_sequence
import yaml
from omegaconf import OmegaConf

from pyabc import pyabc
from str_utils import split_note
from vocab_utils import TokenVocab, MusicTokenVocab
import vocab_utils

def convert_token(token):
  if isinstance(token, pyabc.Note):
    return f"{token.midi_pitch}//{token.duration}"
  if isinstance(token, pyabc.Rest):
    return f"{0}//{token.duration}"
  text = token._text
  if '"' in text:
    text = text.replace('"', '')
  if text == 'M':
    return None
  if text == '\n':
    return None
  if text in ['u', 'v', '.']:
    return None
  return text

def is_used_token(token):
  if isinstance(token, pyabc.ChordSymbol):
    return False
  return True

def is_valid_tune(tune):
  header = tune.header
  if 'key' not in header:
    return False
  if 'meter' not in header:
    return False

  for token in tune.tokens:
    if isinstance(token, pyabc.BodyField):
      '''
      중간에 key나 meter가 바뀌는 경우 사용하지 않음
      '''
      return False
    if isinstance(token, pyabc.InlineField):
      '''
      중간에 key나 meter가 바뀌는 경우 사용하지 않음
      '''
      return False
    token_text = convert_token(token)
    if token_text == '|:1':
      return False
    if token_text == ':||:4':
      return False
    # TODO: 파트가 여러개인 경우 처리하는 부분이 필요함
  # for i in range(1,10):
  #   last_token = tune.tokens[-i]
    
  #   if '|' in  last_token._text:
  #     return True
  #   elif isinstance(last_token, pyabc.Note):
  #     return False
  return True


def is_good_reel(tune:pyabc.Tune):
  if tune.is_tune_with_full_measures \
     and is_valid_tune(tune) \
     and check_last_note_is_root(tune) \
     and not tune.repeat_error\
     and max([measure.meas_offset_from_repeat_start for measure in tune.measures]) < 16 \
     and sum([1 for note in tune.notes if note.duration==0.5]) < 9:
    return True
  else:
    return False

def check_last_note_is_root(tune):
  chromatic = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
  note2idx = {note: i for i, note in enumerate(chromatic)}

  for token in reversed(tune.tokens):
    if isinstance(token, pyabc.Note):
      last_token = token
      break
  key = tune.header['key']
  last_pitch = last_token.pitch.abs_value
  last_dur = last_token.duration

  if last_pitch % 12 == note2idx[key.split(' ')[0]] and last_dur > 1:
    return True
  else:
    return False


def get_min_max_pitch(tune: pyabc.Tune):
  notes = tune.notes
  min_pitch = min([note.pitch.abs_value for note in notes])
  max_pitch = max([note.pitch.abs_value for note in notes])
  return min_pitch, max_pitch

def title_to_list_of_str(tune):
    return [vocab for vocab in tune.title.split(' ')]

def read_abc(file_path):
    with open(file_path, 'r') as f:
        fr = f.read()
    tune = pyabc.Tune(abc=fr)
    return tune

def read_tunes(file_path):
    with open(file_path, 'r') as f:
        fr = f.read()
    tunes = pyabc.Tunes(abc=fr)
    return tunes

'''
 def prepare_abc(paths: list):
    delete_list = ['Z:', 'F:', 'W:'] # F: 가 들어간 abc notation이 있는지 확인 -> 일단은 없음
    tune_list = []
    error_list = []
    
    for path in paths:
        f = open(path)
        abc = f.readlines()
        length = len(abc)

        for line in reversed(abc):
            length -= 1
            if line[:2] in delete_list: # 지워야할 헤더 항목과 각 라인의 앞 부분이 일치하면 pop
                abc.pop(length)

        abc = ''.join(abc)
        abc = abc.replace('\\\n', '\n') # escape 문자로 \ 하나를 더 붙인 부분을 그냥 줄바꿈 기호로 치환

        try: # TODO: 같은 tunes에 묶인 tune을 필요시 구별해서 묶어야함
            tunes = pyabc.Tunes(abc=abc)
            for tune in tunes.tunes:
              # tune = pyabc.Tune(abc=abc)
              if 'rhythm' not in tune.header:
                tune.header['rhythm'] = 'Unspecified'
              if 'unit note length' not in tune.header:
                tune.header['rhythm'] = '1/8'
              if is_valid_tune(tune):
                  tune_list.append(tune)
        except:
            error_list.append(path.name)
        
    return tune_list, error_list
'''

def prepare_abc(paths: list):
  tune_list = []
  error_list = []
  for path in paths:
    try:
      # tune = read_abc(path)
      # tune_list.append(tune)
      tunes = read_tunes(path)
      if len(tunes.tunes) == 0:
        error_list.append(path)
      else:
        tune_list += tunes.tunes
    except:
      error_list.append(path)
  return tune_list, error_list

def decode_melody(melody, vocab):
    '''
    melody (torch.Tensor): model's prediction. Assume that melody has shape of [T]
    '''
    list_of_string = [vocab[token] for token in melody.tolist()[1:]]
    abc_decoded = ''.join(list_of_string)
    return abc_decoded

def read_yaml(yml_path):
  with open(yml_path, 'r') as f:
    yaml_obj = yaml.load(f, Loader=yaml.FullLoader)
  config = OmegaConf.create(yaml_obj)
  return config

def get_emb_total_size(config, vocab):
  vocab_size_dict = vocab.get_size()

  emb_param = config.nn_params.emb
  total_size = 0 
  for key in vocab_size_dict.keys():
    size = int(emb_param[key] * emb_param.emb_size)
    total_size += size
    emb_param[key] = size
  emb_param.total_size = total_size
  config.nn_params.emb = emb_param
  return config

def update_config(config, args):
  config = OmegaConf.merge(config, args)
  return config

class ABCset:
  def __init__(self, dir_path, vocab_path=None, num_limit=None, make_vocab=True, key_aug=None, vocab_name='dict'):
    if isinstance(dir_path, str) or isinstance(dir_path, Path):
      self.dir = Path(dir_path)
      self.abc_list = list(self.dir.rglob('*.abc')) + list(self.dir.rglob('*.ABC'))
      self.abc_list.sort()
      if num_limit is not None:
        self.abc_list = self.abc_list[:num_limit]
      self.tune_list, error_list = prepare_abc(self.abc_list)
    elif isinstance(dir_path, FolkRNNSet):
      self.tune_list = dir_path.pyabc_tunes
    elif isinstance(dir_path, list):
      if isinstance(dir_path[0], pyabc.Tune):
        print("Handling dataset input as a tune list")
        self.tune_list = dir_path
      elif isinstance(dir_path[0], Path):
        print("Handling dataset input as a abc path list")
        self.abc_list = dir_path
        self.tune_list, error_list = prepare_abc(self.abc_list)
    else:
      print(f'Error: Unknown input type: {type(dir_path)}')
    self.tune_list = [tune for tune in self.tune_list if self._check_tune_validity(tune)]
    self._prepare_data()
    if make_vocab:
      self._get_vocab(vocab_path, vocab_name)
    self.augmentor = Augmentor(key_aug, self.tune_list)

  def _check_tune_validity(self, tune):
    return True # NotImplemented

  def _prepare_data(self):
    self.data = [self._tune_to_list_of_str(tune) for tune in self.tune_list]
    self.header = [tune.header for tune in self.tune_list]

  def _get_vocab(self, vocab_path):
    entire_char_list = [token for tune in self.data for token in tune]
    self.vocab = ['<pad>', '<start>', '<end>'] + sorted(list(set(entire_char_list)))
    self.tok2idx = {key: i for i, key in enumerate(self.vocab)}
    #self.idx2tok = self.vocab

  def _get_unique_header_tokens(self):
    target_keys = ['key', 'meter', 'unit note length', 'rhythm']
    head_for_keys = ['K: ', 'M: ', 'L: ', 'R: ']
    output = []
    for head in self.header:
      for i, key in enumerate(target_keys):
        if key in head:
          output.append(head_for_keys[i] + head[key])
    return sorted(list(set(output)))

  def _tune_to_list_of_str(self, tune):
    return [convert_token(token) for token in tune.tokens if is_used_token(token) and convert_token(token) is not None]

  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, idx):
    tune = ['<start>'] + self.data[idx] + ['<end>']
    tune_in_idx = [self.tok2idx[token] for token in tune]
    
    tune_tensor = torch.LongTensor(tune_in_idx)
    
    return tune_tensor[:-1], tune_tensor[1:]

  def get_trainset(self):
    return ABCset(self.tune_list)



class ABCsetTitle(ABCset):
  def __init__(self, dir_path):
    super().__init__(dir_path)
    title_in_text = [title_to_list_of_str(tune) for tune in self.tune_list]
    self.title = title_in_text
    self._get_title_vocab()
  
    
  def _get_title_vocab(self):
    entire_title_list = [token for title in self.title for token in title]
    self.title_vocab = sorted(list(set(entire_title_list)))
    self.ttl2idx = {key: i for i, key in enumerate(self.title_vocab)}
      
  
  def __getitem__(self, idx):
    tune = ['<start>'] + self.data[idx] + ['<end>']
    tune_in_idx = [self.tok2idx[token] for token in tune]
    
    title = self.title[idx]
    title_in_idx = [self.ttl2idx[token] for token in title]
    
    tune_tensor = torch.LongTensor(tune_in_idx)
    title_tensor = torch.LongTensor(title_in_idx)
    
    return tune_tensor, title_tensor


def pack_collate(raw_batch:list):
    '''
  This function takes a list of data, and returns two PackedSequences
  
  Argument
    raw_batch: A list of MelodyDataset[idx]. Each item in the list is a tuple of (melody, shifted_melody)
               melody and shifted_melody has a shape of [num_notes (+1 if you don't consider "start" and "end" token as note), 2]
  Returns
    packed_melody (torch.nn.utils.rnn.PackedSequence)
    packed_shifted_melody (torch.nn.utils.rnn.PackedSequence)

  TODO: Complete this function
    '''  
    
    melody = [mel_pair[0] for mel_pair in raw_batch]
    shifted_melody = [mel_pair[1] for mel_pair in raw_batch]
    
    packed_melody = pack_sequence(melody, enforce_sorted=False)
    packed_shifted_melody = pack_sequence(shifted_melody, enforce_sorted=False)
    
    if len(raw_batch[0]) == 2:
      return packed_melody, packed_shifted_melody
    elif len(raw_batch[0]) == 3:
      measure_numbers = [mel_pair[2] for mel_pair in raw_batch]
      packed_measure_numbers = pack_sequence(measure_numbers, enforce_sorted=False)
      return packed_melody, packed_shifted_melody, packed_measure_numbers
    else:
      raise ValueError("Unknown raw_batch format")



class PitchDurSplitSet(ABCset):
  def __init__(self, dir_path, vocab_path=None, num_limit=None, make_vocab=True, key_aug=None, vocab_name='TokenVocab'):
    super().__init__(dir_path, vocab_path, num_limit, make_vocab, key_aug, vocab_name)

  def _get_vocab(self, vocab_path, vocab_name):
    entire_char_list = [splitted for tune in self.data for token in tune for splitted in split_note(token)]
    unique_header_list = self._get_unique_header_tokens()
    unique_char_list = sorted(list(set(entire_char_list)))  + unique_header_list
    # self.vocab = TokenVocab(vocab_path, unique_char_list)
    self.vocab = getattr(vocab_utils, vocab_name)(vocab_path, unique_char_list)

  def __getitem__(self, idx):
    tune = ['<start>'] + self.data[idx] + ['<end>']
    header = self.header[idx]
    tune_in_idx = [self.vocab(token, header) for token in tune]
    tune_tensor = torch.LongTensor(tune_in_idx)
    header_tensor = torch.LongTensor(self.vocab.encode_header(header))

    tune_tensor = torch.cat([tune_tensor, header_tensor.repeat(len(tune_tensor), 1)], dim=-1)

    return tune_tensor[:-1], tune_tensor[1:]


class MeasureOffsetSet(PitchDurSplitSet):
  def __init__(self, dir_path, vocab_path=None, num_limit=None, make_vocab=True, key_aug=None, vocab_name='TokenVocab'):
    super().__init__(dir_path, vocab_path, num_limit, make_vocab, key_aug, vocab_name)

  def _check_tune_validity(self, tune):
    if len(tune.measures) == 0 or tune.measures[-1].number == 0:
      return False
    if not tune.is_ending_with_bar:
      return False
    if tune.is_tune_with_full_measures or tune.is_incomplete_only_one_measure:
      if is_valid_tune(tune):
        return True
    else:
      return False

  def _prepare_data(self):
    data = [ [self._tune_to_list_of_str(tune), tune.header] for tune in self.tune_list]
    self.data = [x[0] for x in data]
    self.header = [x[1] for x in data]
    # self.header = [tune.header for tune in self.tune_list if tune.is_tune_with_full_measures ]

  def get_str_m_offset(self, token):
    if token.measure_offset is not None:
      return 'm_offset:'+str(float(token.measure_offset))
    else:
      return 'm_offset:'+str(token.measure_offset)

  def _tune_to_list_of_str(self, tune):

    converted_tokens = [ [i, convert_token(token)] for i, token in enumerate(tune.tokens) if is_used_token(token)]
    converted_tokens = [token for token in converted_tokens if token[1] is not None]

    measure_infos = [ ['m_idx:'+str(tune.tokens[i].meas_offset_from_repeat_start),  self.get_str_m_offset(tune.tokens[i])]  for i,_ in converted_tokens]
    last_duration = tune.tokens[converted_tokens[-1][0]].duration if hasattr(tune.tokens[converted_tokens[-1][0]], 'duration') else 0
    last_offset = tune.tokens[converted_tokens[-1][0]].measure_offset + last_duration
    measure_infos += [[measure_infos[-1][0], f'm_offset:{last_offset}']] 
    converted_tokens_w_start = ['<start>'] + [token[1] for token in converted_tokens]

    combined = [ [tk] + meas for tk, meas in zip(converted_tokens_w_start, measure_infos)]

    return combined
    # [[token, 'm_idx:'+str(tune.tokens[i+1].meas_offset_from_repeat_start), 'm_offset:'+str(tune.tokens[i].measure_offset)  ]for i, token in enumerate(converted_tokens)]
    # return [ [convert_token(token), 'm_idx:'+str(token.meas_offset_from_repeat_start), 'm_offset:'+str(token.measure_offset)] 
    #           for token in tune.tokens if is_used_token(token) and convert_token(token) is not None]

  def _get_measure_info_tokens(self):
    return sorted(list(set([info for tune in self.data for token in tune for info in token[1:]])))

  def _get_vocab(self, vocab_path, vocab_name):
    entire_char_list = [splitted for tune in self.data for token in tune for splitted in split_note(token[0])]
    unique_header_list = self._get_unique_header_tokens()
    unique_measure_info_list = self._get_measure_info_tokens()
    unique_char_list = sorted(list(set(entire_char_list)))  + unique_header_list + unique_measure_info_list
    # self.vocab = TokenVocab(vocab_path, unique_char_list)
    # self.vocab = MusicTokenVocab(vocab_path, unique_char_list)
    self.vocab = getattr(vocab_utils, vocab_name)(vocab_path, unique_char_list)

  def __getitem__(self, idx):
    tune = self.data[idx] + [['<end>', '<end>', '<end>']]
    header = self.header[idx]
    tune_in_idx = [self.vocab(token, header) for token in tune]

    tune_tensor = torch.LongTensor(tune_in_idx)
    assert tune_tensor.shape[-1] == 4
    header_tensor = torch.LongTensor(self.vocab.encode_header(header))
    tune_tensor = torch.cat([tune_tensor, header_tensor.repeat(len(tune_tensor), 1)], dim=-1)

    return tune_tensor[:-1], tune_tensor[1:]

class MeasureNumberSet(MeasureOffsetSet):
  def __init__(self, dir_path, vocab_path=None, num_limit=None, make_vocab=True, key_aug=None, vocab_name='MusicTokenVocab'):
    super().__init__(dir_path, vocab_path, num_limit, make_vocab, key_aug, vocab_name)
    # self.vocab = getattr(vocab_utils, vocab_name)('cleaned_vocab_1005.json')
    # self.filter_tune_by_vocab_exists()

  def _get_measure_info_tokens(self):
    return sorted(list(set([info for tune in self.data for token in tune for info in token[1:-1]])))

  def _tune_to_list_of_str(self, tune):

    converted_tokens = [ [i, convert_token(token)] for i, token in enumerate(tune.tokens) if is_used_token(token)]
    converted_tokens = [token for token in converted_tokens if token[1] is not None]

    measure_infos = [ ['m_idx:'+str(tune.tokens[i].meas_offset_from_repeat_start), self.get_str_m_offset(tune.tokens[i]), tune.tokens[i].measure_number]  for i,_ in converted_tokens]

    assert '|' in converted_tokens[-1][1], f"Last token should be barline, {converted_tokens[-1]}"
    # last_duration = tune.tokens[converted_tokens[-1][0]].duration if hasattr(tune.tokens[converted_tokens[-1][0]], 'duration') else 0
    # last_offset = tune.tokens[converted_tokens[-1][0]].measure_offset + last_duration
    # measure_infos += [[measure_infos[-1][0], f'm_offset:{last_offset}', measure_infos[-1][2]]] 
    measure_infos += [[f'm_idx:{str(tune.tokens[converted_tokens[-1][0]].meas_offset_from_repeat_start+1)}', 
                       'm_offset:0.0', 
                       measure_infos[-1][2]+1]]
    converted_tokens_w_start = ['<start>'] + [token[1] for token in converted_tokens]

    combined = [ [tk] + meas for tk, meas in zip(converted_tokens_w_start, measure_infos)]

    return combined
    return [ [convert_token(token), 'm_idx:'+str(token.meas_offset_from_repeat_start), 'm_offset:'+str(token.measure_offset), token.measure_number] 
              for token in tune.tokens if is_used_token(token) and convert_token(token) is not None]

  def filter_tune_by_vocab_exists(self):
    new_tunes = []
    new_headers = []
    for tune, header  in zip(self.data, self.header):
      converted_tune = [x[:-1] for x in tune]
      try:
        [self.vocab(token, header) for token in converted_tune]
        new_tunes.append(tune)
        new_headers.append(header)
        # print('tune added')
      except Exception as e:
        # print(e)
        continue
    self.data = new_tunes
    self.header = new_headers

  def filter_token_by_vocab_exists(self):
    new_tunes = []
    new_headers = []
    for tune, header  in zip(self.data, self.header):
      filtered_tune = []
      converted_tune = [x[:-1] for x in tune]
      for token in converted_tune:
        try:
          self.vocab(token, header)
          filtered_tune.append(token)
        except Exception as e:
          continue
      if len(filtered_tune)>0:
        new_tunes.append(filtered_tune)
        new_headers.append(header)
        print('tune added')
    self.data = new_tunes
    self.header = new_headers

  def __getitem__(self, idx):
    # tune = [['<start>','<start>','<start>' ]] + [x[:-1] for x in self.data[idx]] + [['<end>', '<end>', '<end>']]
    tune = [x[:-1] for x in self.data[idx]]  + [['<end>', '<end>', '<end>']]

    '''
    <start> A A B B | C
            0 1 2 3 4 0 
            0 0 0 0 0 1
    '''
    measure_numbers = [x[-1] for x in self.data[idx]]
    header = self.header[idx]
    tune, new_key = self.augmentor(tune, header)
    new_header = header.copy()
    new_header['key'] = new_key

    tune_in_idx = [self.vocab(token, new_header) for token in tune]

    tune_tensor = torch.LongTensor(tune_in_idx)
    header_tensor = torch.LongTensor(self.vocab.encode_header(new_header))
    tune_tensor = torch.cat([tune_tensor, header_tensor.repeat(len(tune_tensor), 1)], dim=-1)
    # if sum([a>=b for a, b in zip(torch.max(tune_tensor, dim=0).values.tolist(), [x for x in self.vocab.get_size().values()])]) != 0:
    #   print (tune_tensor)

    return tune_tensor[:-1], tune_tensor[1:], torch.tensor(measure_numbers, dtype=torch.long)

  def get_trainset(self, ratio=20):
    train_abc_list = [x for x in self.abc_list if not x.stem.isdigit() or int(x.stem) % ratio != 0]
    trainset =  MeasureNumberSet(train_abc_list, None, make_vocab=False, key_aug=self.augmentor.aug_type)
    trainset.vocab = self.vocab
    return trainset

  def get_testset(self, ratio=20):
    test_abc_list = [x for x in self.abc_list if 'the_session' in x.parent.name and int(x.stem) % ratio == 0]
    if len(test_abc_list) == 0:
      test_abc_list = [x for x in self.abc_list if int(x.stem) % 10 == 0]
    testset =  MeasureNumberSet(test_abc_list, None, make_vocab=False, key_aug=None)
    testset.vocab = self.vocab
    return testset

class MeasureEndSet(MeasureNumberSet):
  def __init__(self, dir_path, vocab_path=None, num_limit=None, make_vocab=True, key_aug=None, vocab_name='MusicTokenVocab'):
    super().__init__(dir_path, vocab_path, num_limit, make_vocab, key_aug, vocab_name)

  def __getitem__(self, idx):
    tune = [x[:-1] for x in self.data[idx]]

    header = self.header[idx]
    for token in reversed(tune):
      if '//' in token[0]:
        last_pitch = int(token[0].split('//')[0])
        last_dur = float(token[0].split('//')[0])
        if last_pitch != 0:
          break
    key = header['key']
    if last_pitch % 12 == self.augmentor.note2idx[key.split(' ')[0]] and last_dur > 1:
      tune.append(['<end>', '<end>', '<end>'])
      measure_numbers = [x[-1] for x in self.data[idx]]
    else:
      measure_numbers = [x[-1] for x in self.data[idx][:-1]]

    tune, new_key = self.augmentor(tune, header)
    new_header = header.copy()
    new_header['key'] = new_key

    tune_in_idx = [self.vocab(token, new_header) for token in tune]

    tune_tensor = torch.LongTensor(tune_in_idx)
    header_tensor = torch.LongTensor(self.vocab.encode_header(new_header))
    tune_tensor = torch.cat([tune_tensor, header_tensor.repeat(len(tune_tensor), 1)], dim=-1)
    return tune_tensor[:-1], tune_tensor[1:], torch.tensor(measure_numbers, dtype=torch.long)

  def get_trainset(self):
    train_abc_list = [x for x in self.abc_list if not x.stem.isdigit() or int(x.stem) % 10 != 0]
    trainset =  MeasureEndSet(train_abc_list, None, make_vocab=False, key_aug=self.augmentor.aug_type)
    trainset.vocab = self.vocab
    return trainset

  def get_testset(self):
    test_abc_list = [x for x in self.abc_list if 'the_session' in x.parent.name and int(x.stem) % 10 == 0]
    if len(test_abc_list) == 0:
      test_abc_list = [x for x in self.abc_list if int(x.stem) % 10 == 0]
    testset =  MeasureEndSet(test_abc_list, None, make_vocab=False, key_aug=None)
    testset.vocab = self.vocab
    return testset


class FolkRNNSet:
  def __init__(self, path='abc_dataset/folk_rnn/data_v3', num_limit=None):
    with open(path, 'r') as f:
      self.abcs = f.read()
    
    self.tunes = self.abcs.split('\n\n')
    if isinstance(num_limit, int):
      print('Limiting to {} tunes'.format(num_limit))
      self.tunes = self.tunes[:num_limit]
    
    self.pyabc_tunes = [] 
    self.error_tunes = []
    for tune in self.tunes:
      try:
        self.pyabc_tunes.append(self.get_pyabc_tune(tune))
      except:
        self.error_tunes.append(tune)
    self.get_vocab()

  def get_vocab(self):
    entire_tokens = [token for tune in self.tunes for token in self.get_tokens(tune)]
    self.vocab = ['<pad>', '<start>', '<end>'] + sorted(list(set(entire_tokens)))
    self.tok2idx = {key: i for i, key in enumerate(self.vocab)}

  def get_meta_vocab(self):
    return

  def get_pyabc_tune(self, abc_like_str):
    return pyabc.Tune('X:0\nT:x\nL:1/8\n' + abc_like_str.replace(' ', '').replace('K:','K:C'))

  def get_meta(self, tune):
    meter, key = tune.split('\n')[:2]
    return meter, key

  def get_tokens(self, tune):
    return tune.split('\n')[-1].split(' ')
    
  def __getitem__(self, idx):
    tune = ['<start>'] + self.data[idx] + ['<end>']
    header = self.header[idx]
    tune_in_idx = [self.vocab(token, header) for token in tune]
    tune_tensor = torch.LongTensor(tune_in_idx)
    header_tensor = torch.LongTensor(self.vocab.encode_header(header))
    tune_tensor = torch.cat([tune_tensor, header_tensor.repeat(len(tune_tensor), 1)], dim=-1)

    return tune_tensor[:-1], tune_tensor[1:]



def get_tunes_from_abc_fns(abc_fns):
  tunes = []
  errors = []
  for fn in abc_fns:
    try:
      with open(fn, 'r') as f:
        abc = f.read()
      tunes.append(pyabc.Tunes(abc).tunes)
    except:
      errors.append(fn)
  return tunes, errors


class Augmentor:
  def __init__(self, aug_type, tune_list):
    self.aug_type = aug_type

    # self.chromatic = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
    self.chromatic = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
    self.note2idx = {note: i for i, note in enumerate(self.chromatic)}

    self.key_stats = self.get_key_stats(tune_list)

  def get_key_stats(self, tune_list):
    counter_by_mode = defaultdict(lambda: defaultdict(int))

    for tune in tune_list:
      key = tune.header['key']
      pitch, mode = key.split(' ')
      counter_by_mode[mode][pitch] += 1

    # normalize each mode
    for mode in counter_by_mode:
      total = sum(counter_by_mode[mode].values())
      for pitch in counter_by_mode[mode]:
        counter_by_mode[mode][pitch] /= total
    
    return counter_by_mode

  def get_key_diff(self, key1, key2):
    pitch_name1 = key1.split(' ')[0]
    pitch_name2 = key2.split(' ')[0]

    pitch_idx1 = self.note2idx[pitch_name1]
    pitch_idx2 = self.note2idx[pitch_name2]

    direct = pitch_idx2 - pitch_idx1
    reverse = pitch_idx2 - 12 - pitch_idx1
    higher = pitch_idx2 + 12 - pitch_idx1

    min_distance = min([abs(direct), abs(reverse), abs(higher)])
    if min_distance == abs(direct):
      return direct
    elif min_distance == abs(reverse):
      return reverse
    else:
      return higher

  def change_note_str(self, note_str, key_diff):
    pitch, dur = note_str.split('//')
    pitch = int(pitch)
    if pitch == 0: # note is rest
      return note_str
    pitch += key_diff
    return f'{pitch}//{dur}'

  def change_token(self, token, key_diff):
    main_token = token[0]
    if '//' in main_token:
      return [self.change_note_str(main_token, key_diff)] + token[1:]
    else:
      return token

  
  def get_random_key(self, org_key):
    org_key, mode = org_key.split(' ')

    new_chroma = self.chromatic.copy()
    new_chroma.remove(org_key)

    return random.choice(new_chroma) + ' ' + mode

  def get_random_stat_key(self, org_key):
    mode = org_key.split(' ')[1]
    distribution = self.key_stats[mode]

    new_key = random.choices(list(distribution.keys()), list(distribution.values()))[0]

    return new_key + ' ' + mode


  def __call__(self, str_tokens, header):
    '''
    str_tokens: list of list of str
    
    '''
    org_key = header['key']
    if self.aug_type is None:
      return str_tokens, org_key
    elif self.aug_type == 'c':
      new_key = "C" + " " + org_key.split(' ')[1]
    elif self.aug_type == 'random':
      new_key = self.get_random_key(org_key)
    elif self.aug_type == 'stat':
      new_key = self.get_random_stat_key(org_key)
    elif self.aug_type == "recover":
      if 'transcription' in  header and ' ' in header['transcription'] and header['transcription'].split(' ')[0] in self.chromatic:
        recover_key = header['transcription']
        key_diff_compen = self.chromatic.index(recover_key.split(' ')[0]) - self.chromatic.index(org_key.split(' ')[0])
        new_key = self.get_random_key(recover_key)
      else:
        return str_tokens, org_key
    else:
      print('Invalid aug_type: {}'.format(self.aug_type))
      raise NotImplementedError

    key_diff = self.get_key_diff(org_key, new_key)
    if self.aug_type == "recover":
      if 'key_diff_compen' in locals():
        key_diff += key_diff_compen
      else:
        key_diff = key_diff % 12 # always transpose to higher direction
    if key_diff > 12:
      key_diff -= 12
    if key_diff == 0:
      return str_tokens, new_key
    converted = [self.change_token(token, key_diff) for token in str_tokens]

    return converted, new_key

