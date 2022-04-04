"""
Author: Subho Mukherjee (submukhe@microsoft.com)
Code for XtremeDistil for distilling massive multi-lingual models.
"""

from sklearn.decomposition import PCA
from transformers import *

import logging
import numpy as np

logger = logging.getLogger('xtremedistil')

# HuggingFace Transformers has a unified API
# for 10 transformer architectures and 30 pretrained weights.
#          Model          | Tokenizer          | Pretrained model config
MODELS = [(TFBertModel, BertTokenizerFast, BertConfig),
		  (TFElectraModel, ElectraTokenizerFast, ElectraConfig),
          (TFOpenAIGPTModel, OpenAIGPTTokenizerFast, OpenAIGPTConfig),
          (TFGPT2Model, GPT2TokenizerFast, GPT2Config),
          (TFCTRLModel, CTRLTokenizer, CTRLConfig),
          (TFTransfoXLModel,  TransfoXLTokenizer, TransfoXLConfig),
          (TFXLNetModel, XLNetTokenizer, XLNetConfig),
          (TFXLMModel, XLMTokenizer, XLMConfig),
          (TFDistilBertModel, DistilBertTokenizerFast, DistilBertConfig),
          (TFRobertaModel, RobertaTokenizerFast, RobertaConfig),
          (TFXLMRobertaModel, XLMRobertaTokenizer, XLMRobertaConfig),
         ]