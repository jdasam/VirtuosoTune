import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import wandb
from data_utils import decode_melody
from wandb import Html
from decoding import LanguageModelDecoder
from pathlib import Path
from collections import defaultdict
import time

class Trainer:
  def __init__(self, model, 
                     optimizer,
                     scheduler,
                     loss_fn, 
                     train_loader, 
                     valid_loader,
                     args,
                     ):
    self.model = model
    self.optimizer = optimizer
    self.scheduler = scheduler
    self.loss_fn = loss_fn
    self.train_loader = train_loader
    self.valid_loader = valid_loader

    save_dir = args.save_dir
    device = args.device
    model_name = args.model_name
    grad_clip = args.grad_clip
    num_epoch_per_log = args.num_epoch_per_log
    num_iter_per_valid = args.num_iter_per_valid


    self.save_dir = Path(save_dir)
    
    self.model.to(device)
    
    self.grad_clip = grad_clip
    self.best_valid_loss = 100
    self.device = device
    
    self.training_loss = []
    self.validation_loss = []
    self.validation_acc = []
    self.model_name = model_name
    self.num_epoch_per_log = num_epoch_per_log
    self.num_iter_per_valid = num_iter_per_valid

    self.make_log = not args.no_log


    if isinstance(self.train_loader.dataset, torch.utils.data.dataset.Subset):
      vocab = self.train_loader.dataset.dataset.vocab
    else:
      vocab = self.train_loader.dataset.vocab
    self.vocab = vocab
    self.vocab.save_json(self.save_dir / 'vocab.json')
    self.abc_decoder = LanguageModelDecoder(vocab, self.save_dir)

  def save_model(self, path):
    torch.save({'model':self.model.state_dict(), 'optim':self.optimizer.state_dict(), 'vocab':self.vocab}, path)
  

  def train_by_num_iter(self, num_iters):
    generator = iter(self.train_loader)
    for i in tqdm(range(num_iters)):
      try:
          # Samples the batch
          batch = next(generator)
      except StopIteration:
          # restart the generator if the previous generator is exhausted.
          generator = iter(self.train_loader)
          batch = next(generator)

      loss_value, loss_dict = self._train_by_single_batch(batch)
      loss_dict = self._rename_dict(loss_dict, 'train')
      if self.make_log:
        wandb.log(loss_dict, step=i)
      self.training_loss.append(loss_value)
      if (i+1) % self.num_iter_per_valid == 0:
        self.model.eval()
        validation_loss, validation_acc, validation_metric_dict = self.validate()
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
          self.scheduler.step(validation_loss)
        validation_metric_dict = self._rename_dict(validation_metric_dict, 'valid')
        if self.make_log:
          wandb.log(validation_metric_dict, step=i)

        self.validation_loss.append(validation_loss)
        self.validation_acc.append(validation_acc)
        
        self.best_valid_loss = min(validation_loss, self.best_valid_loss)

        if (i+1) % (self.num_epoch_per_log * self.num_iter_per_valid) == 0:
          self.inference_and_log(i)
          self.save_model(self.save_dir / f'{self.model_name}_iter{i}_loss{validation_loss:.4f}.pt')

        self.model.train()


  def train_by_num_epoch(self, num_epochs):
    for epoch in tqdm(range(num_epochs)):
      self.model.train()
      for batch in self.train_loader:
        loss_value, loss_dict = self._train_by_single_batch(batch)
        loss_dict = self._rename_dict(loss_dict, 'train')
        wandb.log(loss_dict)
        self.training_loss.append(loss_value)
      self.model.eval()
      validation_loss, validation_acc, validation_metric_dict = self.validate()
      self.scheduler.step(validation_loss)
      validation_metric_dict = self._rename_dict(validation_metric_dict, 'valid')
      wandb.log(validation_metric_dict)
      # wandb.log({"validation_loss": validation_loss,
      #            "validation_acc": validation_acc})
      self.validation_loss.append(validation_loss)
      self.validation_acc.append(validation_acc)
      
      # if validation_acc > self.best_valid_accuracy:
      if validation_loss < self.best_valid_loss:
        print(f"Saving the model with best validation loss: Epoch {epoch+1}, Loss: {validation_loss:.4f} ")
        self.save_model(self.save_dir / f'{self.model_name}_best.pt')
      else:
        self.save_model(self.save_dir / f'{self.model_name}_last.pt')
      self.best_valid_loss = min(validation_loss, self.best_valid_loss)

      if epoch % self.num_epoch_per_log == 0:
        self.inference_and_log(epoch)

      # decoded_output = decode_melody(generated_output, self.vocab)
      # wandb.log({"generated_abc": Html(decoded_output)})

      
  def _train_by_single_batch(self, batch):
    '''
    This method updates self.model's parameter with a given batch
    
    batch (tuple): (batch_of_input_text, batch_of_label)
    
    You have to use variables below:
    
    self.model (Translator/torch.nn.Module): A neural network model
    self.optimizer (torch.optim.adam.Adam): Adam optimizer that optimizes model's parameter
    self.loss_fn (function): function for calculating BCE loss for a given prediction and target
    self.device (str): 'cuda' or 'cpu'

    output: loss (float): Mean binary cross entropy value for every sample in the training batch
    The model's parameters, optimizer's steps has to be updated inside this method
    '''
    start_time = time.time()
    loss, _, loss_dict = self.get_loss_pred_from_single_batch(batch)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
    self.optimizer.step()
    if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
      self.scheduler.step()
    self.optimizer.zero_grad()
    loss_dict['time'] = time.time() - start_time
    loss_dict['lr'] = self.optimizer.param_groups[0]['lr']
    
    return loss.item(), loss_dict

  def get_loss_pred_from_single_batch(self, batch):
    melody, shifted_melody = batch
    pred = self.model(melody.to(self.device))
    loss = self.loss_fn(pred.data, shifted_melody.data)
    loss_dict = {'total': loss.item()}
    return loss, pred, loss_dict

  def get_valid_loss_and_acc_from_batch(self, batch):
    melody, shifted_melody = batch
    loss, pred, loss_dict = self.get_loss_pred_from_single_batch(batch)
    num_tokens = melody.data.shape[0]

    acc = torch.sum(torch.argmax(pred.data, dim=-1) == shifted_melody.to(self.device).data)

    validation_loss = loss.item() * num_tokens
    num_total_tokens = num_tokens
    validation_acc = acc.item()

    return validation_loss, num_total_tokens, validation_acc, loss_dict

    
  def validate(self, external_loader=None):
    '''
    This method calculates accuracy and loss for given data loader.
    It can be used for validation step, or to get test set result
    
    input:
      data_loader: If there is no data_loader given, use self.valid_loader as default.
      
    output: 
      validation_loss (float): Mean Binary Cross Entropy value for every sample in validation set
      validation_accuracy (float): Mean Accuracy value for every sample in validation set
    '''
    
    ### Don't change this part
    if external_loader and isinstance(external_loader, DataLoader):
      loader = external_loader
      print('An arbitrary loader is used instead of Validation loader')
    else:
      loader = self.valid_loader
      
    self.model.eval()
    
    '''
    Write your code from here, using loader, self.model, self.loss_fn.
    '''
    validation_loss = 0
    validation_acc = 0
    num_total_tokens = 0
    validation_metric_dict = defaultdict(float)
    with torch.inference_mode():
      for batch in tqdm(loader, leave=False):
        tmp_validation_loss, tmp_num_total_tokens, tmp_validation_acc, loss_dict = self.get_valid_loss_and_acc_from_batch(batch)
        validation_loss += tmp_validation_loss
        num_total_tokens += tmp_num_total_tokens
        validation_acc += tmp_validation_acc
        for key, value in loss_dict.items():
          validation_metric_dict[key] += value * tmp_num_total_tokens
    
    for key in validation_metric_dict.keys():
      validation_metric_dict[key] /= num_total_tokens
        
    return validation_loss / num_total_tokens, validation_acc / num_total_tokens, validation_metric_dict

  def inference_and_log(self, iter, num_sample=20):
    # generated_output = self.model.inference(self.vocab)
    # self.abc_decoder(generated_output, self.save_dir / f'abc_decoded_{iter}')
    for i in range(num_sample):
      try:
        generated_output = self.model.inference(self.vocab, manual_seed=i)
        save_out = i==0 # only save the first one
        self.abc_decoder(generated_output, self.save_dir / f'abc_decoded_{iter}_seed_{i}', True, save_out)
        if self.make_log and save_out:
          wandb.log({"gen_score": wandb.Image(str(self.save_dir /f'abc_decoded_{iter}_seed_{i}-1.png')), 
                    "gen_audio": wandb.Audio(str(self.save_dir /f'abc_decoded_{iter}_seed_{i}.wav'))}
                    , step=iter)
        print(f"Inference is logged: Iter {iter} / seed {i}")
      except Exception as e:
        print(e)

  def _rename_dict(self, adict, prefix='train'):
    keys = list(adict.keys())
    for key in keys:
      adict[f'{prefix}.{key}'] = adict.pop(key)
    return dict(adict)

class TrainerPitchDur(Trainer):
  def __init__(self, model, 
                     optimizer, 
                     scheduler,
                     loss_fn, 
                     train_loader, 
                     valid_loader,
                     args):
    super().__init__(model, optimizer, scheduler, loss_fn, train_loader, valid_loader, args)
    

  def get_loss_pred_from_single_batch(self, batch):
    melody, shifted_melody = batch
    pred = self.model(melody.to(self.device))

    main_loss = self.loss_fn(pred.data[:, :self.model.vocab_size[0]], shifted_melody.data[:,0])
    is_note = shifted_melody.data[:,1] > 2 # is not pad, start, or end tokens
    dur_loss = self.loss_fn(pred.data[is_note, self.model.vocab_size[0]:], shifted_melody.data[is_note,1])
    loss = main_loss + dur_loss

    loss_dict = {'main': main_loss.item(), 'dur': dur_loss.item(), 'total':loss.item()}
    return loss, pred, loss_dict


  def get_valid_loss_and_acc_from_batch(self, batch):
    melody, shifted_melody = batch
    loss, pred, loss_dict = self.get_loss_pred_from_single_batch(batch)
    num_tokens = melody.data.shape[0]


    is_note = shifted_melody.data[:,1] > 2 # is not pad, start end tokens
    main_acc = torch.sum(torch.argmax(pred.data[:, :self.model.vocab_size[0]], dim=-1) == shifted_melody.to(self.device).data[:,0])
    dur_acc = torch.sum(torch.argmax(pred.data[is_note, self.model.vocab_size[0]:], dim=-1) == shifted_melody.to(self.device).data[is_note,1])

    acc = main_acc  + dur_acc * is_note.sum() / num_tokens

    validation_loss = loss.item() * num_tokens
    num_total_tokens = num_tokens
    validation_acc = acc.item()

    loss_dict['main_acc'] = main_acc.item()
    loss_dict['dur_acc'] = (dur_acc * is_note.sum() / num_tokens).item()

    return validation_loss, num_total_tokens, validation_acc, loss_dict

class TrainerMeasure(TrainerPitchDur):
  def __init__(self, model, 
                     optimizer, 
                     scheduler,
                     loss_fn, 
                     train_loader, 
                     valid_loader,
                     args):
    super().__init__(model, optimizer, scheduler, loss_fn, train_loader, valid_loader, args)

  def get_loss_pred_from_single_batch(self, batch):
    melody, shifted_melody, measure_numbers = batch
    pred = self.model(melody.to(self.device), measure_numbers)

    main_loss = self.loss_fn(pred.data[:, :self.model.vocab_size[0]], shifted_melody.data[:,0])
    is_note = shifted_melody.data[:,1] > 2 # is not pad, start end tokens
    dur_loss = self.loss_fn(pred.data[is_note, self.model.vocab_size[0]:], shifted_melody.data[is_note,1])
    loss = main_loss + dur_loss
    loss_dict = {'main': main_loss.item(), 'dur': dur_loss.item(), 'total':loss.item()}

    return loss, pred, loss_dict

  def get_valid_loss_and_acc_from_batch(self, batch):
    melody, shifted_melody, _ = batch
    loss, pred, loss_dict = self.get_loss_pred_from_single_batch(batch)
    num_tokens = melody.data.shape[0]

    is_note = shifted_melody.data[:,1] > 2 # is not pad, start end tokens
    main_acc = torch.sum(torch.argmax(pred.data[:, :self.model.vocab_size[0]], dim=-1) == shifted_melody.to(self.device).data[:,0])
    dur_acc = torch.sum(torch.argmax(pred.data[is_note, self.model.vocab_size[0]:], dim=-1) == shifted_melody.to(self.device).data[is_note,1])

    acc = main_acc  + dur_acc * is_note.sum() / num_tokens

    validation_loss = loss.item() * num_tokens
    num_total_tokens = num_tokens
    validation_acc = acc.item()

    loss_dict['main_acc'] = main_acc.item() / num_tokens
    loss_dict['dur_acc'] = dur_acc.item() / is_note.sum().item()

    return validation_loss, num_total_tokens, validation_acc, loss_dict