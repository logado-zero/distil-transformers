"""
Author: Subho Mukherjee (submukhe@microsoft.com)
Code for XtremeDistil for distilling massive multi-lingual models.
"""

from preprocessing import convert_to_unicode

from transformers import BertConfig, BertModel, BertForSequenceClassification
from huggingface_utils import get_output_state_indices

import csv
import logging
import numpy as np
import os
from tqdm import tqdm
import random
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger('xtremedistil')

# set seeds for random number generator for reproducibility
GLOBAL_SEED = int(os.getenv("PYTHONHASHSEED"))
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
torch.cuda.manual_seed(GLOBAL_SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = False


def gelu(x):
  """Gaussian Error Linear Unit.
  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    x: float Tensor to perform activation.
  Returns:
    `x` with the GELU activation applied.
  """
  cdf = 0.5 * (1.0 + torch.tanh(
      (np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3)))))
  return x * cdf

def get_H_t(X):
    ans = X[-1, :, :, :]
    return ans

def SparseCategoricalCrossentropy_torch(true,pred):
    return torch.nn.NLLLoss(torch.log(torch.nn.functional.softmax(pred)), true)


class construct_transformer_teacher_model(torch.nn.Module):
    def __init__(self, args, ModelTeacher, teacher_config):
        super(construct_transformer_teacher_model, self).__init__()
        self.ModelTeacher = ModelTeacher
        self.teacher_config = teacher_config
        self.args = args

        classes = len(args["label_list"])

        self.encoder = ModelTeacher.from_pretrained(args["pt_teacher_checkpoint"], config=teacher_config, from_pt=True, name="tf_model")
        self.dropout = torch.nn.Dropout(p=teacher_config.hidden_dropout_prob)
        self.linear = torch.nn.Linear(768, classes) 
        torch.nn.init.trunc_normal_(self.linear.weight, std= teacher_config.initializer_range)

    def forward(self, input_ids, attention_mask, token_type_ids):

        encode = self.encoder(input_ids, token_type_ids=token_type_ids,  attention_mask=attention_mask)

        output_hidden_state_indx, output_attention_state_indx =  get_output_state_indices(self.ModelTeacher)

        

        embedding = []
        if self.args["distil_multi_hidden_states"]:
            if self.args["do_NER"]:
                #add hidden states
                for i in range(1, self.args["num_hidden_layers"]+2):
                    embedding.append(encode[output_hidden_state_indx][-i])
            else:
                for i in range(1, self.args["num_hidden_layers"]+2):
                    embedding.append(encode[output_hidden_state_indx][-i][:,0])
            if self.args["distil_attention"]:
                #add attention states
                for i in range(1, self.args["num_hidden_layers"]+1):
                    embedding.append(encode[output_attention_state_indx][-i])
        else:
            if self.args["do_NER"]:
                embedding.append(encode[0])
            else:
                embedding.append(encode[0][:,0])

        logger.info(self.teacher_config)

        dropout = self.dropout(embedding[0])
        output = self.linear(dropout)
        
        
        return output 

class construct_transformer_teacher_model(torch.nn.Module):
    def __init__(self, args, ModelTeacher, teacher_config):
        super(construct_transformer_teacher_model, self).__init__()
        self.ModelTeacher = ModelTeacher
        self.teacher_config = teacher_config
        self.args = args

        classes = len(args["label_list"])

        self.encoder = ModelTeacher.from_pretrained(args["pt_teacher_checkpoint"], config=teacher_config, from_pt=True, name="tf_model")

        
    def forward(self, input_ids, attention_mask, token_type_ids):

        encode = self.encoder(input_ids, token_type_ids=token_type_ids,  attention_mask=attention_mask)

        output_hidden_state_indx, output_attention_state_indx =  get_output_state_indices(self.ModelTeacher)

        

        embedding = []
        if self.args["distil_multi_hidden_states"]:
            if self.args["do_NER"]:
                #add hidden states
                for i in range(1, self.args["num_hidden_layers"]+2):
                    embedding.append(encode[output_hidden_state_indx][-i])
            else:
                for i in range(1, self.args["num_hidden_layers"]+2):
                    embedding.append(encode[output_hidden_state_indx][-i][:,0])
            if self.args["distil_attention"]:
                #add attention states
                for i in range(1, self.args["num_hidden_layers"]+1):
                    embedding.append(encode[output_attention_state_indx][-i])
        else:
            if self.args["do_NER"]:
                embedding.append(encode[0])
            else:
                embedding.append(encode[0][:,0])

        logger.info(self.teacher_config)

        dropout = self.dropout(embedding[0])
        output = self.linear(dropout)
        
        
        return output, embedding

def compile_model(model, args, stage):

    #construct student models for different stages
    if stage == 1 or stage == 2:
        if args["distil_attention"] and args["distil_multi_hidden_states"]:
            num_loss = (2 * args["num_hidden_layers"] + 1)
        elif args["distil_multi_hidden_states"]:
            num_loss = (args["num_hidden_layers"] + 1)
        else:
            num_loss = 1
        loss_dict = {
            "loss_name": torch.nn.MSELoss(),
            "num": num_loss
        }
    else:
        loss_dict = {
            "loss_name": SparseCategoricalCrossentropy_torch(),
            "num": 1
        }
        
    return loss_dict


def train_model(model, train_dataset, dev_dataset, optimizer, loss_dict, batch_size=4, epochs =100):
    training_generator = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_generator = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        idx = 0
        pr = tqdm(training_generator, total=len(training_generator), leave=False)
        for batch in pr:
            idx += 1
            outputs = model(batch[0])
            loss = 0
            for i in range(loss_dict["num"]):
                loss += loss_dict["loss_name"](outputs, batch[1])

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss = loss.data.cpu().tolist()
            epoch_loss += loss
            pr.set_description("train loss: {}".format(epoch_loss / idx))
            torch.cuda.empty_cache()
        logging.info("\nepoch {}, average train epoch loss={:.5}\n".format(epoch, epoch_loss / idx))
    

    return model