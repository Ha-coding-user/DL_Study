"""
    Attention 101 > BahdanauAttention

        - this code is for educational purpose.
        - the code is written for easy understanding not for optimized code.

    Author : Sangkeun Jung (hugmanskj@gmail.com)
    All rights reserved. (2021)
"""


# In this code, we will implement
#   - Badanau attention mechanism which is the first attention algorithm.
#   - Note that Badanau attention mechanism is one of the additive attention mechanism.

import torch
import torch.nn as nn
import torch.nn.functional as F


## ------------------------------------------------------------------------ ##
## Important! - Bahdanau Attention                                          ##
## ------------------------------------------------------------------------ ##

class BahdanauAttention(nn.Module):
    """
    Attention > Additive Attention > Bahdanau approach 

    Inputs:
        query_vector  : [hidden_size]
        multiple_items: [batch_size, num_of_items, hidden_size]
    Returns:
        blendded_vector:    [batch_size, item_vector hidden_size]
        attention_scores:   [batch_size, num_of_items]
    """
    def __init__(self, item_dim, query_dim, attention_dim):
        super(BahdanauAttention, self).__init__()
        self.item_dim = item_dim                # dim. of multiple item vector
        self.query_dim = query_dim              # dim. of query vector
        self.attention_dim = attention_dim      # dim. of projected item or query vector

        # W is used for project query to the attention dimension
        # U is used  for project each item to the attention dimension
        self.W = nn.Linear(self.query_dim, self.attention_dim, bias=False)
        self.U = nn.Linear(self.item_dim, self.attention_dim, bias=False)

        # v is used for calculating attention score which is scalar value
        self.v = nn.Parameter(torch.randn(1, attention_dim, dtype=torch.float))

    def _calculate_reacitvity(self, query_vector, multiple_items):
        B, N, H = multiple_items.shape

        # linear projection 마지막 dimension에 적용이 됨
        query_vector = query_vector.unsqueeze(1)
        projected_q = self.W(query_vector)          # [B, 1, Q] ---> [B, 1, D]
        projected_item = self.U(multiple_items)     # [B, N, H] ---> [B, N, D]

        # note that broadcasting is performed when adding different shape
        added_items = projected_q + projected_item  # [B, 1, D] + [B, N, D] ---> [B, N, D]
        tanh_items = torch.tanh(added_items)        # [B, N, D]

        v_t = self.v.transpose(1, 0)
        batch_v = v_t.expand(B, self.attention_dim, 1)      # [B, D, 1]
        reactivity_scores = torch.bmm(tanh_items, batch_v)  # [B, N, D] x [B, D, 1] ---> [B, N, 1]
        reactivity_scores = reactivity_scores.squeeze(-1)   # [B, N, 1] ---> [B, N]
        return reactivity_scores    # [B, N]

    def forward(self, query_vector, multiple_items, mask):
        """"
        Inputs:
            query_vector:   [query_vector hidden_size]
            multiple_items: [batch_size, num_of_items, item_vector hidden_size]
            mask:           [batch_size, num_of_items, num_of_items] 1 for valid item, 0 for invalid item
        Returns:
            blendded_vector:    [batch_size, item_vector hidden_size]
            attention_scores:   [batch_size, num_of_items]
        """

        assert mask is not None, 'mask is required'

        # B : batch_size, N : number of multiple items, H : hidden size of item
        B, N, H = multiple_items.size()

        # Three Steps
        # 1) [reactivity]   try to check the reactivity with (itme_t and query_vector) N times
        # 2) [masking]      try to penalize invalid items such as <pad>
        # 3) [attention]    try to get proper attention scores (=probability form) over the reactivity scores
        # 4) [blend]        try to blend multiple items with attention scores

        # Step-1) reactivity
        # -> query vector와 각각 item들이 어떤 반응성을 일으키는지
        reactivity_scores = self._calculate_reacitvity(query_vector, multiple_items)

        # Step-2) masking
        # The mask marks valid position so we invert it using 'mask & 0'
        # detail : check the masked_fill_() of pytorch
        # mask가 0인 지점의 값을 변경
        reactivity_scores.data.masked_fill_(mask == 0, -float('inf'))

        # Step-3) attention_score
        attention_scores = F.softmax(reactivity_scores, dim=1) 
        # Step-4) blend multiple items
        # merge by weighted sum
        attention_scores = attention_scores.unsqueeze(1)    # [B, 1, #_of_items]

        # [B, 1, #_of_items] * [B, #_of_items, dim_of_item] --> [B, 1, dim_of_item]
        blendded_vector = torch.matmul(attention_scores, multiple_items)
        blendded_vector = blendded_vector.squeeze(1)    # [B, dim_of_item]

        return blendded_vector, attention_scores
    
## ------------------------------------------------------------------------ ##
## Important! - Dot Attention                                               ##
## ------------------------------------------------------------------------ ##
class DotAttention(nn.Module):
    """
    Attention > Additive Attention > Dot approach

    Inputs:
        query_vector    : [batch_size, hidden_size]
        multiple_items  : [batch_size, num_of_items, hidden_size]
    Returns:
        blendded_voector: [batch_size, item_vector hidden_size]
        attention_scores: [batch_size, num_of_items]
    """
    
    def __init__(self, item_dim, query_dim, attention_dim):
        super(DotAttention, self).__init__()
        self.item_dim = item_dim
        self.query_dim = query_dim
        self.attention_dim = attention_dim

        assert query_dim == item_dim, "Dot attention require dim. of query and dim. of item is same"

    def _calculate_reactivity(self, query_vector, multiple_items):
        # 'dot' method try to get scalar value by dot operation
        # see : [1, H] x [H, 1] = [1, 1] => [1] scalar value
        query_vector = query_vector.unsqueeze(-1)   # [B, H] --> [B, H, 1]

        # [B, N, H] x [B, H, 1] --> [B, N, 1]
        reactivity_scores = torch.bmm(multiple_items, query_vector) # [B, N, 1]
        reactivity_scores = reactivity_scores.squeeze(-1)   # [B, N]
        return reactivity_scores
    
    def forward(self, query_vector, multiple_items, mask):
        """
        Inputs:
            query_vector    : [query_vector hidden_size]
            multiple_items  : [batch_size, num_of_items, item_vector hidden_size]
            mask            : [batch_size, num_of_items, num_of_items]
        Returns:
            blendded_vector : [batch_size, item_vector hidden_size]
            attention_scores: [batch_size, num_of_items]
        """
        assert mask is not None, 'mask is required'

        # B : batch_size, N : number of multiple items, H : hidden size of item
        B, N, H = multiple_items.size()

        # Three Steps
        # 1) [reactivity]   try to check the reactivity with (item_t and query_vector) N times
        # 2) [masking]      try to penalize invalid items such as <pad>
        # 3) [attention]    try to get proper attention scores (=probability form) over the reactivity scores
        # 4) [blend]        try to blend multiple items with attention scores

        # Step-1) reactivity
        reactivity_scores = self._calculate_reactivity

        # Step-2) masking
        # The mask marks valid positions so we invert it usin 'mask & 0'
        # detail : check the masked_fill_() of pytorch
        reactivity_scores.masked_fill_(mask==0, -float('inf'))

        # Step-3) attention score
        attention_scores = F.softmax(reactivity_scores, dim=1)

        # Step-4) blend multiple items
        # merge by weighted sum
        attention_scores = attention_scores.unsqueeze(1)    # [B, 1, #_of_items]

        # [B, 1, #_of_items] * [B, #_of_items, dim_of_items] --> [B, 1, dim_of_items]
        blendded_vector = torch.matmul(attention_scores, multiple_items)    # [B, dim_of_items]
        blendded_vector = blendded_vector.squeeze(1)

        return blendded_vector, attention_scores


## ------------------------------------------------------------------------ ##
## Training and Testing with toy dataset                                    ##
## ------------------------------------------------------------------------ ##

import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import numpy as np

# data load 함수
def load_data(fn):
    data = []
    with open(fn, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()    # '\n' 제거

            seq_str, query, y = line.split('\t')
            seqs = seq_str.split(',')
            data.append((seqs, query, y))

    return data

# Dataset 생성 class
class NumberDataset(Dataset):
    """Dataset"""

    def __init__(self, fn, input_vocab, output_vocab, max_seq_length):
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab
        self.max_seq_length = max_seq_length

        # load
        self.data = load_data(fn)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        seq, q, y = self.data[idx]

        # [input]
        seq_ids = [self.input_vocab[t] for t in seq]

        # <pad> processing
        pad_id = self.input_vocab['<pad>']
        num_to_fill = self.max_seq_length - len(seq)
        seq_ids = seq_ids + [0]*num_to_fill

        # mask processing (1 for valid, 0 for valid)
        weights = [1]*len(seq) + [0]*num_to_fill

        # ex) 
        # seq_ids : 6, 3, 5, 2, 4, _, _, _
        # weights : 1, 1, 1, 1, 1, 0, 0, 0

        # [ query ]
        # NOTE : we assume that query vocab space is same as input vocab space

        q_id = self.input_vocab[q]

        # [output]
        y_id = self.output_vocab[y]

        item = [
            # input
            np.array(seq_ids),
            q_id,
            np.array(weights),

            # output
            y_id
        ]

        return item

# Dataset을 모듈로
class NumberDataModule(pl.LightningDataModule):
    def __init__(self,
                 max_seq_length: int=12,
                 batch_size: int=32):
        super().__init__()
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length

        input_vocab, output_vocab = self.make_vocab('./data/numbers/train.txt')
        self.input_vocab_size = len(input_vocab)
        self.output_vcoab_size = len(output_vocab)
        self.padding_idx = input_vocab['<pad>']

        root = './data/numbers/'
        self.all_train_dataset = NumberDataset(root+'train.txt', input_vocab, output_vocab, max_seq_length)
        self.test_dataset = NumberDataset(root+'test.txt', input_vocab, output_vocab, max_seq_length)

        self.input_r_vocab = {v:k for k, v in input_vocab.items()}
        self.output_r_vocab = {v:k for k, v in output_vocab.items()}

        # random split train / valid for early stopping
        N = len(self.all_train_dataset)
        tr = int(N*0.8)
        va = N - tr
        self.train_dataset, self.valid_dataset = torch.utils.data.random_split(self.all_train_dataset, [tr, va])

    def make_vocab(self, fn):
        input_tokens = []
        output_tokens = []
        data = load_data(fn)

        for seqs, query, y in data:
            for token in seqs:
                input_tokens.append(token)
            output_tokens.append(y)

        input_tokens = list(set(input_tokens))
        output_tokens = list(set(output_tokens))

        input_tokens.sort()
        output_tokens.sort()

        # [input vocab]
        # add <pad> symbol to input tokens as a first item
        input_tokens = ['<pad>'] + input_tokens
        input_vocab = {str(token):index for index, token in enumerate(input_tokens)}

        # [output vocab]
        output_vocab = {str(token):index for index, token in enumerate(output_tokens)}

        return input_vocab, output_vocab
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True)
    
    def valid_dataloader(self):
        return DataLoader(self.valid_dataset,
                          batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size)

from torchmetrics import functional as FM

## ------------------------------------------------------------------------ ##
## Important! - Attention                                                   ##
## ------------------------------------------------------------------------ ##

class Attention_Number_Finder(pl.LightningModule):
    def __init__(self,
                 # networking setting
                 input_vocab_size,
                 output_vocab_size,
                 d_model,   # dim. in attention mechanism
                 padding_idx,
                 # optimizer setting
                 learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()

        # note
        # - the dimension for query and multi-itmes do not need to be same
        # - fir simplicity, we make all the dimensions as same.

        # symbol_number_character to vector_number
        self.digit_emb = nn.Embedding(self.hparams.input_vocab_size,
                                      self.hparams.d_models,
                                      padding_idx=self.hparams.padding_idx)
        
        # sequence encoder using RNN
        self.encoder = nn.LSTM(d_model,
                               int(self.hparams.d_model/2), # since bidirectional LSTM
                               num_layers=2,
                               bidirectional=True,
                               batch_first=True)
        
        # attention mechanism - additive attention
        self.att = BahdanauAttention(item_dim=self.hparmas.d_model,
                                     query_dim=self.hparams.d_model,
                                     attention_dim=self.hparams.d_model)
        
        # attention mechanism - DotProduction attention
        """
        self.att = DotAttention(item_dim=self.hparams.d_model,
                                query_dim=self.hparams.d_model,
                                attention_dim=self.hparams.d_model)
        """
        
        # [to output]
        self.to_output = nn.Linear(self.hparams.d_model, self.hparams.output_vocab_size)

        # loss
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, seq_ids, q_id, weight):
        # ---------------------ENCODING with ATTENTION------------------------- #
        # [Digit Character Embedding]
        # seq_ids : [B, max_seq_len]
        seq_embs = self.digit_emb(seq_ids.long()) # [B, max_seq_len, emb_dim]

        # [Sequence of Numbers Encoding]
        seq_encs, _ = self.encoder(seq_embs)    # [B, max_seq_len, enc_dim*2] since we have 2 layers

        # with query (context)
        query = self.digit_emb(q_id)    # [B, query_dim]

        # dynamic encoding-summarization (blending)
        multiple_items = seq_encs

        blendded_vector, attention_score = self.att(query, multiple_items, mask=weight)
        # blendded_vector : [B, query_len, dim_of_sequence_enc]
        # attention_scores : [B, query_len, key_len]
        # mask -> padding이 들어간 부분은 attention 계산에서 제외해야 하므로

        # To Output
        logits = self.to_output(blendded_vector)
        return logits, attention_score
    
    def training_step(self, batch, batch_idx):
        seq_ids, q_id, weights, y_id = batch
        logits, _ = self(seq_ids, q_id, weights)    # [B, output_vocab]
        loss = self.criterion(logits, y_id.long())
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # all logs are automatically stored for tensorboard
        return loss
    
    def validation_step(self, batch, batch_idx):
        seq_ids, q_id, weights, y_id = batch

        logits, _ = self(seq_ids, q_id, weights)    # [B, output_vocab]
        loss = self.criterion(logits, y_id.long())

        # get predicted result
        prob = F.softmax(logits, dim=-1)
        acc = FM.accuracy(prob, y_id)
        metrics = {'val_acc': acc, 'val_loss': loss}
        self.long_dict(metrics)
        return metrics
    
    def validation_step_end(self, val_step_output):
        val_acc = val_step_output['val_acc'].cpu()
        val_loss = val_step_output['val_loss'].cpu()

        self.log('validation_acc', val_acc, prog_bar=True)
        self.log('validation_loss', val_loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        seq_ids, q_id, weights, y_id = batch

        logits, _ = self(seq_ids, q_id, weights)
        loss = self.criterion(logits, y_id)

        # get predicted result
        prob = F.softmax(logits, dim=-1)
        acc = FM.accuracy(prob, y_id)
        metrics = {'test_acc': acc, 'test_loss': loss}
        self.log_dict(metrics, on_epoch=True)

        return metrics
    
    def configure_optimizer(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('ATTENTION')
        parser.add_argument('--learning_rate', type=float, default=0.0001)

        return parent_parser
    
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

plt.rcParams['figure.figsize'] = [10, 8]

def check_attention(model, ex, input_vocab, output_vocab):
    seq_ids, q_id, weights, y_id = ex
    seq_ids = seq_ids.to(model.device)
    q_id = q_id.to(model.device)
    weights = weights.to(model.device)

    import os
    os.makedirs('./output_figs/Bahdanau', exist_ok=True)

    # prediction
    with torch.no_grad():
        logits, att_scores = model(seq_ids, q_id, weights)

        prob = F.softmax(logits)
        y_id_pred = prob.argmax(dim=-1)

        for idx, (a_seq_ids, a_q_id, a_weights, a_y_id, a_y_id_pred, a_att_scores) in enumerate(zip(seq_ids, q_id, weights, y_id, y_id_pred, att_scores)):
            N = a_weights.sum().item()

            input_sym = [input_vocab[i.item()] for i in a_seq_ids[:N]]
            q_sym = input_vocab[a_q_id.item()]

            ref_y_sym = output_vocab[a_y_id_pred.item()]
            pred_y_sym = output_vocab[a_y_id_pred.item()]

            scores = a_att_scores.cpu().detach().numpy()[0][:N].tolist()

            ## heatmap
            data = {'scores': []}
            for word, score in zip(input_sym, scores):
                data['scores'].append(score)
                df = pd.DataFrame(data)
            
            df.index = input_sym

            plt.figure()
            # sns.set(rc={'figure.figsize': (2,8)})
            sns.heatmap(df, cmap='RdYlGn_r')
            plt.title(f'Finding the first larger value than query={q_sym}, ref={ref_y_sym}, pred={pred_y_sym}', fontsize=10)
            plt.savefig(os.path.join('./output_figs/Bahdanau', f'{idx}.png'))

from argparse import ArgumentParser
from pytorch_lightning.callbacks import EarlyStopping

def cli_main():
    pl.seed_everything(1234)

    # ----------------
    # args
    # ----------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=200, type=int)
    parser.add_argument('--d_model', default=512, type=int)     # dim for attention model

    parser = pl.Trainer.add_argparse_args(parser)
    parser = Attention_Number_Finder.add_model_specific_args(parser)
    args = parser.parse_args()

    # ----------------
    # data
    # ----------------
    dm = NumberDataModule.from_argparse_args(args)
    iter(dm.train_dataloader()).next()

    # ----------------
    # model
    # ----------------
    model = Attention_Number_Finder(dm.input_vocab_size,
                                    dm.output_vocab_size,
                                    args.d_model,
                                    dm.padding_idx,
                                    args.learning_rate)
    
    # ----------------
    # training
    # ----------------
    trainer = pl.Trainer(
        max_epochs=10,
        callbacks=[EarlyStopping(monitor='val_loss')],
        gqus = 1
    )
    trainer.fit(model, datamodule=dm)

    # ----------------
    # testing
    # ----------------
    result = trainer.test(model, test_dataloaders=dm.test_dataloader())
    print(result)

    # ----------------
    # Check the attention scores to attend on multiple items
    # ----------------

    ex_batch = iter(dm.test_dataloader()).next()
    check_attention(model, ex_batch, dm.input_r_vocab, dm.output_r_vocab)

if __name__ == '__main__':
    cli_main()