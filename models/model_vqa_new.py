'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

from functools import partial
from models.vit import VisionTransformer, interpolate_pos_embed
from models.xbert import BertConfig, BertModel

import torch
import torch.nn.functional as F
from torch import nn

import numpy as np
import random
import json

class ALBEF(nn.Module):
    def __init__(self,                 
                 text_encoder = None,
                 tokenizer = None,
                 config = None
                 ):
        super().__init__()
        
        self.tokenizer = tokenizer
     
        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))   
        
        bert_config = BertConfig.from_json_file(config['bert_config'])
        self.text_encoder = BertModel.from_pretrained('./bert-model', config=bert_config) 
        text_width = self.text_encoder.config.hidden_size
        self.answer_list = json.load(open(config['answer_list'],'r'))
        self.answer_map = { value:idx for idx, value in enumerate(self.answer_list) }
        self.classifier = nn.Sequential(
          nn.Linear(text_width, text_width),
          nn.ReLU(),
          nn.Linear(text_width, len(self.answer_list))
        )

    def forward(self, image, question, answer=None, train=True, k=0, weights=None):
        
        image_embeds = self.visual_encoder(image) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)

        with torch.no_grad():
            output = self.text_encoder(question.input_ids, 
                                            attention_mask = question.attention_mask,
                                            encoder_hidden_states = image_embeds,
                                            encoder_attention_mask = image_atts,      
                                            return_dict = True,
                                           )
        
        logits = self.classifier(output.last_hidden_state[:, 0, :])
        
        if train:
            batch_size = logits.size(0)
            num_classes = logits.size(1)
            target = torch.zeros(batch_size, num_classes).to(logits.device)

            assert len(answer) == len(weights) == sum(k), "Mismatch in lengths of answer, weights, and k"
            
            idx = 0
            for i in range(batch_size):
                for _ in range(k[i]):
                    ans = answer[idx]
                    score = weights[idx]
                    if ans in self.answer_map:
                        target[i, self.answer_map[ans]] += score
                    idx += 1
            
            # Compute log-softmax and KL divergence loss
            log_probs = F.log_softmax(logits, dim=-1)
            loss = F.kl_div(log_probs, target, reduction='batchmean')

            return loss
        else:
            # Inference mode: predict top-1 answer
            probs = F.softmax(logits, dim=-1)  # [batch_size, num_classes]
            topk_probs, topk_ids = torch.topk(probs, k, dim=-1)  # each of shape [batch_size, k]
    
            return topk_ids, topk_probs
