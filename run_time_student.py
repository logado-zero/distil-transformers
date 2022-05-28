from evaluation import ner_evaluate
from huggingface_utils import MODELS, get_special_tokens_from_teacher
from preprocessing import generate_sequence_data, get_labels, Dataset
from transformers import *

import argparse
import json
import logging
import models
import numpy as np
import os
import random
import sys
import torch
from torchsummary import summary

#logging
logger = logging.getLogger('xtremedistil')
logging.basicConfig(level = logging.INFO)

#set random seeds
GLOBAL_SEED = int(os.getenv("PYTHONHASHSEED"))
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
torch.cuda.manual_seed(GLOBAL_SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = False
logger.info ("Global seed {}".format(GLOBAL_SEED))

if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    parser = argparse.ArgumentParser()

    #required arguments
    parser.add_argument("--pred_file", nargs="?", help="file for prediction")
    parser.add_argument("--batch_size", nargs="?", type=int, default=256, help="predict batch size")
    parser.add_argument("--model_dir", required=True, help="path of model directory")
    parser.add_argument("--do_eval", action="store_true", default=False, help="whether to evaluate model performance on test data")
    parser.add_argument("--do_predict", action="store_true", default=False, help="whether to make model predictions")

    #mixed precision
    parser.add_argument("--opt_policy", nargs="?", default=False, help="mixed precision policy")

    args = vars(parser.parse_args())
    logger.info(args)

    logger.info ("Directory of script ".format(os.path.dirname(os.path.abspath(__file__))))
    
    
    if not args["do_eval"] and not args["do_predict"]:
        logger.info ("Select one of do_eval or do_predict flags")
        sys.exit(1)

    #load xtreme distil config
    distil_args = json.load(open(os.path.join(args["model_dir"], "xtremedistil-config.json"), 'r'))
    label_list = distil_args["label_list"]

    #get pre-trained model, tokenizer and config
    for indx, model in enumerate(MODELS):
        if model[0].__name__ == distil_args["pt_teacher"]:
            ModelTeacher, Tokenizer, TeacherConfig = MODELS[indx]

    #get pre-trained tokenizer and special tokens
    pt_tokenizer = Tokenizer.from_pretrained(distil_args["pt_teacher_checkpoint"])
    special_tokens = get_special_tokens_from_teacher(Tokenizer, pt_tokenizer)

    #generate sequence data for fine-tuning pre-trained teacher
    if args["do_predict"]:
        X, _ = generate_sequence_data(distil_args["seq_len"], args["pred_file"], pt_tokenizer, label_list=label_list, unlabeled=True, special_tokens=special_tokens, do_pairwise=distil_args["do_pairwise"], do_NER=distil_args["do_NER"])
        logger.info("X Shape {}".format(X["input_ids"].shape))

    if args["do_eval"]:
        X_test, y_test = generate_sequence_data(distil_args["seq_len"], os.path.join(args["pred_file"], "test.tsv"), pt_tokenizer, label_list=label_list, special_tokens=special_tokens, do_pairwise=distil_args["do_pairwise"], do_NER=distil_args["do_NER"])
        logger.info("X Shape {}".format(X_test["input_ids"].shape))

    #initialize word embedding
    word_emb = None
    if distil_args["compress_word_embedding"]:
        if distil_args["freeze_word_embedding"]:
            word_emb = np.load(open(os.path.join(args["model_dir"], "word_embedding.npy"), "rb"))
        else:
            word_emb = np.zeros((pt_tokenizer.vocab_size, distil_args["hidden_size"]))

    #generate Dataset
    test_dataset = Dataset(X_test,y_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.construct_transformer_student_model(distil_args, word_emb=word_emb)
    loss_dict = models.compile_model(model, distil_args, stage=2)
    optimizer = torch.optim.Adam(model.parameters(),lr=3e-5, eps=1e-08)

    logger.info(summary(model,input_size=(384,),depth=1,batch_dim=1, dtypes=[torch.IntTensor]))
    model.load_state_dict(torch.load(os.path.join(args["model_dir"], "xtremedistil.h5"),map_location=torch.device(device)))

    cur_eval = ner_evaluate(model, test_dataset , label_list, special_tokens, distil_args["seq_len"], batch_size=args["batch_size"], device =device,\
                            name_model = 'student',stage =2, check_time_process = True)