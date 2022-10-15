import sys
import re
import pdb
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
from torch.autograd import Variable
import torch.nn.functional as F

from data import STOP_TOKENS
from domain import get_domain
from models import modules

class SelectModel(nn.Module):
    def __init__(self, word_dict, output_length, args, device, pretrained_embeds):
        super(SelectModel, self).__init__()

        domain = get_domain(args.domain)

        self.word_dict = word_dict
        self.args = args
        self.device = device
        if torch.cuda.is_available():
            self.cuda_usage = True
        self.num_ent = domain.num_ent()

        if self.args.trainable_embeds:
            if self.cuda_usage:
                self.pretrained_embeds = torch.nn.Embedding.from_pretrained(pretrained_embeds, freeze=False)
            else:
                self.pretrained_embeds = torch.nn.Embedding.from_pretrained(torch.from_numpy(pretrained_embeds),
                                                                            freeze=False)
        else:
            self.pretrained_embeds = pretrained_embeds

        # context encoder
        if args.rel_ctx_encoder:
            self.ctx_encoder = modules.RelationalContextEncoder(domain.num_ent(), domain.dim_ent(), args.rel_hidden,
            args.nembed_ctx, args.dropout, args.init_range, device)
        else:
            self.ctx_encoder = modules.MlpContextEncoder(domain.input_length(),
            args.nembed_ctx, args.dropout, args.init_range, device)

        self.dropout = nn.Dropout(args.dropout)

        # a bidirectional selection RNN
        # it will go through input words and generate by the reader hidden states
        # to produce a hidden representation
        self.sel_rnn = nn.GRU(
            input_size=args.nembed_word,
            hidden_size=args.nhid_lang,
            bias=True,
            bidirectional=True)

        self.sel_encoder = nn.Sequential(
            torch.nn.Linear(2 * args.nhid_lang + args.nembed_ctx, args.nhid_sel),
            nn.Tanh()
        )

        # selection decoder
        self.sel_decoder = nn.Linear(args.nhid_sel, self.num_ent)

        if self.args.context_only:
            self.sel_encoder = nn.Sequential(
            torch.nn.Linear(args.nembed_ctx, args.nhid_sel),
            nn.Tanh()
            )
            self.sel_decoder = nn.Linear(args.nhid_sel, self.num_ent)

        self.init_weights()

    def zero_hid(self, bsz, nhid=None, copies=None):
        """A helper function to create an zero hidden state."""
        nhid = self.args.nhid_lang if nhid is None else nhid
        copies = 1 if copies is None else copies
        hid = torch.zeros(copies, bsz, nhid)
        hid = hid.to(self.device)
        return hid

    def init_weights(self):
        """Initializes params uniformly."""

        modules.init_rnn(self.sel_rnn, self.args.init_range)#, bidirectional=True)

        modules.init_cont([self.sel_encoder], self.args.init_range)
        modules.init_cont([self.sel_decoder], self.args.init_range)

    def forward_selection(self, inpt, lang_h, ctx_h):
        """Forwards selection pass."""
        # run a birnn over the concatenation of the input embeddings and
        # language model hidden states
        if self.args.context_only:
            out = ctx_h

            out = self.dropout(out)
            out = self.sel_encoder(out)
            out = self.sel_decoder.forward(out)
            return out.squeeze(0)

        # inpt is a tensor of integers and  is of size (140, 16)

        if self.args.trainable_embeds:
            inpt.to(torch.int64)
            inpt_emb = self.pretrained_embeds(inpt)
        else:
            inpt_emb = self.pretrained_embeds[inpt] # shape (140,16,300)

        if self.cuda_usage:
            h_1 = self.dropout(inpt_emb)
            h_1 = h_1.to('cuda')
        elif self.args.trainable_embeds:
            h_1 = self.dropout(inpt_emb)
        else:
            h_1 = self.dropout(torch.from_numpy(inpt_emb))

        # runs selection rnn over the hidden state h_1
        h_0 = self.zero_hid(bsz=h_1.size(1), nhid=self.args.nhid_lang, copies=2) # shape (2,16,128)

        self.sel_rnn.flatten_parameters()

        if self.device == 'cuda':
            h_0 = h_0.to('cuda')

        out, _ = self.sel_rnn(h_1, h_0)

        out = torch.cat([out[-1,:,:], ctx_h.squeeze(0)], 1)

        out = self.dropout(out)
        out = self.sel_encoder(out)

        out = self.sel_decoder.forward(out)
        return out

    def forward_context(self, ctx):
        """Run context encoder."""
        return self.ctx_encoder(ctx)
