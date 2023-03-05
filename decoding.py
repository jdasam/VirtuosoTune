from music21 import environment
from music21 import converter
import muspy
import os, io, sys
from music21 import abcFormat
from pathlib import Path

def noop(x):
  pass

class MuteWarn:
    def __enter__(self):
        self._init_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._init_stdout


abcFormat.translate.environLocal.warn = noop
us=environment.UserSettings()
us['musescoreDirectPNGPath'] = '/usr/bin/mscore'

os.putenv("QT_QPA_PLATFORM", "offscreen")
os.putenv("XDG_RUNTIME_DIR", environment.Environment().getRootTempDir())


def save_score_image_from_abc(abc, file_name):
  assert isinstance(abc, str)
  with MuteWarn():
    convert = converter.parse(abc)
    convert.write('musicxml.png', fp=file_name)

def save_wav_from_abc(abc, file_name, qpm=200):
  assert isinstance(abc, str)
  with MuteWarn():
    music = muspy.read_abc_string(abc)
    music.tempos[0].qpm = qpm
    music.write_audio(file_name, rate=16000)
    # muspy.read_abc_string(abc).write_audio(file_name, rate=16000)

def save_midi_from_abc(abc, file_name):
  assert isinstance(abc, str)
  with MuteWarn():
    muspy.read_abc_string(abc).write_midi(file_name)


def save_abc(abc_str, abc_fn):
  with open(abc_fn, 'w') as f:
    f.write(abc_str)

class Note2ABC:
  def __init__(self) -> None:
    self.abc_vocab = self.get_abc_pitchs_w_sharp()
    self.abc_vocab_flat = self.get_abc_pitchs_w_flat()

  def get_abc_pitchs_w_sharp(self):
    abc_octave = ["C", "^C", "D", "^D", "E", "F", "^F", "G", "^G", "A", "^A", "B"]
    abc_notation = []

    for num in range(4):
      if num == 0:
        octave = [p + ',' for p in abc_octave]
      elif num == 1:
        octave = abc_octave
      elif num == 2:
        octave = [p.lower() for p in abc_octave]
      else:
        octave = [p.lower() + "'" for p in abc_octave]
      abc_notation.extend(octave)
    return abc_notation

  def get_abc_pitchs_w_flat(self):
    abc_octave = ["C","_D", "D", "_E", "E", "F", "_G", "G", "_A", "A", "_B", "B"]
    abc_notation = []
    for num in range(4):
      if num == 0:
        octave = [p + ',' for p in abc_octave]
      elif num == 1:
        octave = abc_octave
      elif num == 2:
        octave = [p.lower() for p in abc_octave]
      else:
        octave = [p.lower() + "'" for p in abc_octave]
      abc_notation.extend(octave)
    return abc_notation

  def pitch2abc(self, midi_pitch: int, is_sharp: bool):
    if is_sharp:
      return self.abc_vocab[midi_pitch-36]
    else:
      return self.abc_vocab_flat[midi_pitch-36]

  def duration2abc(self, duration):
    if duration == 0.5:
      return '/'
    elif duration == 1:
      return ''
    elif duration == 0.75:
      return '3/4'
    elif duration == 0.25:
      return '1/4'
    elif duration == 1.5:
      return '3/2'
    else:
      return str(int(duration))

  def __call__(self, pitch_dur_str: str, is_sharp=True, transpose=0):
    pitch, dur = pitch_dur_str.split('//')
    midi_pitch = int(pitch.replace('pitch',''))
    dur = float(dur.replace('dur',''))

    pitch_str = self.pitch2abc(midi_pitch + transpose, is_sharp)
    dur_str = self.duration2abc(dur)

    return pitch_str + dur_str


class LanguageModelDecoder:
  def __init__(self, vocab, save_dir='./'):
    self.vocab = vocab
    self.converter = Note2ABC()
    self.save_dir = Path(save_dir)

    self.chromatic = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']

    self.key_to_is_sharp = {'C Major': True, 'C minor': False, 'C Dorian': False, 'C Mixolydian': False,
                   'Db Major': False, 'C# minor': True, 'C# Dorian': True, 'Db Mixolydian': False,
                   'D Major': True, 'D minor': False, 'D Dorian': True, 'D Mixolydian': True,
                   'Eb Major': False, 'Eb minor': False, 'Eb Dorian': False, 'Eb Mixolydian': False,
                   'E Major': True, 'E minor': True, 'E Dorian': True, 'E Mixolydian': True,
                   'F Major': False, 'F minor': False, 'F Dorian': False, 'F Mixolydian': False,
                   'Gb Major': False, 'F# minor': True, 'F# Dorian': True, 'F# Mixolydian': True,
                   'G Major': True, 'G minor': False, 'G Dorian': False, 'G Mixolydian': True,
                   'Ab Major': False, 'G# minor': True, 'G# Dorian': True, 'Ab Mixolydian': False,
                   'A Major': True, 'A minor': True, 'A Dorian': True, 'A Mixolydian': True,
                   'Bb Major': False, 'Bb minor': False, 'Bb Dorian': False, 'Bb Mixolydian': False,
                   'B Major': True, 'B minor': True, 'B Dorian': True, 'B Mixolydian': True}

    self.mode2key = {'Major': 0, 'minor': -3, 'Ionian': 0, 'Aeolian': -3,
               'Mixolydian': -1, 'Dorian': -2, 'Phrygian': -4, 'Lydian': 1, 'Locrian': -5}
    self.root2key = {'C#': 7, 'F#': 6, 'B': 5, 'E': 4, 'A': 3, 'D': 2, 'G': 1, 'C': 0,
           'F': -1, 'Bb': -2, 'Eb': -3, 'Ab': -4, 'Db': -5, 'Gb': -6, 'Cb': -7}

    self.sharp_order = ["f", "c", "g", "d", "a", "e", "b"]
    self.flat_order = ["b", "e", "a", "d", "g", "c", "f"]



  def transpose_key(self, key, transpose):
    '''
    key (str): C Major , C minor, C Dorian etc
    '''
    root = key.split(' ')[0]
    mode = key.split(' ')[1]
    root_idx = self.chromatic.index(root)
    new_root_idx = (root_idx + transpose) % 12

    return self.chromatic[new_root_idx] + ' ' + mode


  def decode(self, model_pred, meta_string='X:1\nT:Title\nM:4/4\nL:1/8\nK:C\n', transpose=0):
    '''
    
    transpose (int): amount of transpose in semitones
    '''
    list_of_string = self.vocab.decode(model_pred)
    key = meta_string.split('\nK:')[1].lstrip().replace('\n', '')
    if transpose != 0:
      key = self.transpose_key(key, transpose)
      meta_string = meta_string.split('\nK:')[0] + '\nK: ' + key + '\n'
    is_sharp = self.key_to_is_sharp[key]

    abc_string = [self.converter(x, is_sharp, transpose) if '//' in x else x for x in list_of_string]
    
    abc_decoded = ''.join(abc_string)
    abc_decoded = self.clean_accidentals(abc_decoded, key)
    abc_decoded = meta_string + abc_decoded

    return abc_decoded

  def parse_num_sharps_from_key(self, key):
    # example: key = G Major
    root, mode = key.split(' ')

    # get key signature
    root_calib = self.root2key[root]
    mode_calib = self.mode2key[mode]

    key_sig = root_calib + mode_calib

    return key_sig

  def clean_accidentals(self, abc_note_string, key):
    num_sharps = self.parse_num_sharps_from_key(key)
    if num_sharps == 0:
      return abc_note_string
    if num_sharps > 0:
      glob_accidentals = self.sharp_order[:num_sharps]
    else:
      glob_accidentals = self.flat_order[:abs(num_sharps)]
    
    new_str = ''
    accidentals = glob_accidentals.copy()
    for i in range(len(abc_note_string)-1):
      sliced_str = abc_note_string[i:i+2]
      if sliced_str[0] in ["_", "^"]:
        if sliced_str[1].lower() in accidentals:
          continue
        else:
          new_str += sliced_str[0]
          accidentals.append(sliced_str[1].lower())
      else:
        new_str += sliced_str[0]
      if sliced_str[0] == "|":
        accidentals = glob_accidentals
    new_str += abc_note_string[-1]

    return new_str

  def __call__(self, model_pred_or_abc, file_code='abc_decoded_0', save_image=True, save_audio=True, meta_string='X:1\nT:Title\nM:4/4\nL:1/8\nK:C\n',):
    # list_of_string = [self.vocab[token] for token in model_pred.tolist()[1:]]
    if isinstance(model_pred_or_abc, str):
      abc_decoded = model_pred_or_abc
    else:
      abc_decoded = self.decode(model_pred_or_abc, meta_string)
    if save_image:
      save_score_image_from_abc(abc_decoded, self.save_dir / f'{file_code}.png')
    if save_audio:
      save_wav_from_abc(abc_decoded, self.save_dir / f'{file_code}.wav')
    save_abc(abc_decoded, self.save_dir / f'{file_code}.abc')
    save_midi_from_abc(abc_decoded, self.save_dir / f'{file_code}.mid')
    return abc_decoded

