from evaluation import train_model, ner_evaluate
from huggingface_utils import MODELS, get_special_tokens_from_teacher, get_output_state_indices, get_word_embedding
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
    parser.add_argument("--task", required=True, help="name of the task")
    parser.add_argument("--model_dir", required=True, help="path of model directory")
    parser.add_argument("--seq_len", required=True, type=int, help="sequence length")
    parser.add_argument("--transfer_file", required=True, help="transfer data for distillation")
    parser.add_argument("--teacher_model_dir", required=True, help="path of model directory")

    #task
    parser.add_argument("--do_NER", action="store_true", default=False, help="whether to perform NER")
    parser.add_argument("--do_pairwise", action="store_true", default=False, help="whether to perform pairwise instance classification tasks")

    #transformer student model parameters (optional)
    parser.add_argument("--hidden_size", nargs="?", type=int, default=384, help="hidden state dimension of the student model")
    parser.add_argument("--num_attention_heads", nargs="?", type=int, default=6, help="number of attention heads")
    parser.add_argument("--num_hidden_layers", nargs="?", type=int, default=6, help="number of layers in the student model")
    #optional student model checkpoint to load from
    parser.add_argument("--pt_student_checkpoint", nargs="?", default=False, help="student model checkpoint to initialize the distilled model with pre-trained weights.")

    #distillation features
    parser.add_argument("--distil_attention", action="store_true", default=False, help="whether to distil teacher attention")
    parser.add_argument("--distil_multi_hidden_states", action="store_true", default=False, help="whether to distil multiple hidden layers from teacher")
    parser.add_argument("--compress_word_embedding", action="store_true", default=False, help="whether to compress word embedding matrix")
    parser.add_argument("--freeze_word_embedding", action="store_true", default=False, help="whether to use pre-trained word embedding with froze parameters")

    #teacher model parameters (optional)
    parser.add_argument("--pt_teacher", nargs="?", default="TFBertModel",help="Pre-trained teacher model to distil")
    parser.add_argument("--pt_teacher_checkpoint", nargs="?", default="bert-base-multilingual-cased", help="teacher model checkpoint to load to pre-trained weights")

    #mixed precision
    parser.add_argument("--opt_policy", nargs="?", default=False, help="mixed precision policy")

    #batch sizes and epochs (optional)
    parser.add_argument("--student_distil_batch_size", nargs="?", type=int, default=128, help="batch size for distillation")
    parser.add_argument("--student_ft_batch_size", nargs="?", type=int, default=32, help="batch size for fine-tuning student model")
    parser.add_argument("--teacher_batch_size", nargs="?", type=int, default=128, help="batch size for distillation")
    parser.add_argument("--ft_epochs", nargs="?", type=int, default=100, help="epochs for fine-tuning")
    parser.add_argument("--distil_epochs", nargs="?", type=int, default=500, help="epochs for distillation")
    parser.add_argument("--distil_chunk_size", nargs="?", type=int, default=100000, help="transfer data partition size (reduce if OOM)")
    parser.add_argument("--patience", nargs="?", type=int, default=5, help="number of iterations for early stopping.")

    args = vars(parser.parse_args())
    logger.info(args)
    logger.info ("Directory of script ".format(os.path.dirname(os.path.abspath(__file__))))
    
    #get pre-trained model, tokenizer and config
    for indx, model in enumerate(MODELS):
        if model[0].__name__ == args["pt_teacher"]:
            ModelTeacher, Tokenizer, TeacherConfig = MODELS[indx]

    #get pre-trained tokenizer and special tokens
    pt_tokenizer = Tokenizer.from_pretrained(args["pt_teacher_checkpoint"])

    special_tokens = get_special_tokens_from_teacher(Tokenizer, pt_tokenizer)
    output_hidden_state_indx, output_attention_state_indx =  get_output_state_indices(ModelTeacher)

    teacher_config = TeacherConfig.from_pretrained(args["pt_teacher_checkpoint"], output_hidden_states=args["distil_multi_hidden_states"], output_attentions=args["distil_attention"])

    if args["pt_student_checkpoint"]:
        student_config = BertConfig.from_pretrained(args["pt_student_checkpoint"], output_hidden_states=args["distil_multi_hidden_states"], output_attentions=args["distil_attention"])
        args["hidden_size"] = student_config.hidden_size
        args["num_hidden_layers"] = student_config.num_hidden_layers
        args["num_attention_heads"] = student_config.num_attention_heads

    if args["distil_attention"]:
        args["distil_multi_hidden_states"] = True

    args["teacher_hidden_size"] = teacher_config.hidden_size

    if args["freeze_word_embedding"]:
        args["compress_word_embedding"] = True

    #get labels for NER
    label_list=None
    if args["do_NER"]:
        label_list = get_labels(os.path.join(args["task"], "labels.tsv"), special_tokens)

    #generate sequence data for fine-tuning pre-trained teacher
    X_train, y_train = generate_sequence_data(args["seq_len"], os.path.join(args["task"], "train.tsv"), pt_tokenizer, label_list=label_list, special_tokens=special_tokens, do_pairwise=args["do_pairwise"], do_NER=args["do_NER"])

    X_test, y_test = generate_sequence_data(args["seq_len"], os.path.join(args["task"], "test.tsv"), pt_tokenizer, label_list=label_list, special_tokens=special_tokens, do_pairwise=args["do_pairwise"], do_NER=args["do_NER"])

    X_dev, y_dev = generate_sequence_data(args["seq_len"], os.path.join(args["task"], "dev.tsv"), pt_tokenizer, label_list=label_list, special_tokens=special_tokens, do_pairwise=args["do_pairwise"], do_NER=args["do_NER"])

    X_unlabeled, _ = generate_sequence_data(args["seq_len"], args["transfer_file"], pt_tokenizer, label_list=label_list, unlabeled=True, special_tokens=special_tokens, do_pairwise=args["do_pairwise"], do_NER=args["do_NER"])

    #generate Dataset
    train_dataset = Dataset(X_train,y_train)
    test_dataset = Dataset(X_test,y_test)
    dev_dataset = Dataset(X_dev,y_dev)

    if not args["do_NER"]:
        label_list = [str(elem) for elem in set(y_train)]

    args["label_list"] = label_list

    #logging teacher data shapes
    logger.info("X Train Shape {} {}".format(X_train["input_ids"].shape, y_train.shape))
    logger.info("X Dev Shape {} {}".format(X_dev["input_ids"].shape, y_dev.shape))
    logger.info("X Test Shape {} {}".format(X_test["input_ids"].shape, y_test.shape))
    logger.info("X Unlabeled Transfer Shape {}".format(X_unlabeled["input_ids"].shape))

    for i in range(3):
        logger.info ("Example {}".format(i))
        logger.info ("Input sequence: {}".format(pt_tokenizer.convert_ids_to_tokens(X_train["input_ids"][i])))
        logger.info ("Input ids: {}".format(X_train["input_ids"][i]))
        logger.info ("Attention mask: {}".format(X_train["attention_mask"][i]))
        logger.info ("Token type ids: {}".format(X_train["token_type_ids"][i]))
        if args["do_NER"]:
            logger.info ("Label sequence: {}".format(' '.join([label_list[v] for v in y_train[i]])))
        else:
            logger.info ("Label: {}".format(y_train[i]))

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    teacher_model = models.construct_transformer_teacher_model(args, ModelTeacher, teacher_config)
    teacher_model.to(device)

    logger.info(summary(teacher_model,input_size=(768,),depth=1,batch_dim=1, dtypes=[torch.IntTensor]))
    loss_dict = models.compile_model(teacher_model, args, stage=3)
    optimizer = torch.optim.Adam(teacher_model.parameters(),lr=3e-5, eps=1e-08)
    
    model_file = os.path.join(args["teacher_model_dir"], 'teacher_weights.pth')

    if os.path.exists(model_file):
        logger.info ("Loadings weights for fine-tuned model from {}".format(model_file))
        teacher_model.load_state_dict(torch.load(model_file))
    else:
        teacher_model = train_model(teacher_model, train_dataset, dev_dataset, optimizer = optimizer, loss_dict =loss_dict,
                    batch_size= args["teacher_batch_size"], epochs=args["ft_epochs"], device=device, path_save =  os.path.join(args["teacher_model_dir"], 'teacher_weights_best.pth'))
        # callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=args["patience"], restore_best_weights=True)]
        torch.save(teacher_model.state_dict(), model_file)


   
    ner_evaluate(teacher_model, test_dataset , label_list, special_tokens, args["seq_len"], batch_size=args["teacher_batch_size"], device =device)
    word_emb = None
    if args["compress_word_embedding"]:
        word_emb = get_word_embedding(teacher_model._modules.get('encoder'), pt_tokenizer, args["hidden_size"])


    model_1 = models.construct_transformer_student_model(args, stage=1, word_emb=word_emb)
    model_2 = models.construct_transformer_student_model(args, stage=2, word_emb=word_emb)
    
    shared_layers = set()
    for name, param in model_1.named_parameters():
        if param.requires_grad:
            shared_layers.add(name.split(".")[0])
    shared_layers = list(shared_layers)
    logger.info ("Shared layers {}".format(shared_layers))