from evaluation import train_model, ner_evaluate, load_history_file, save_history_file, train_model_student
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
from tqdm import tqdm



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
        
        logger.info ("Label sequence: {}".format(' '.join([label_list[v] for v in y_train[i]])))
        

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    teacher_model = models.construct_transformer_teacher_model(args, ModelTeacher, teacher_config)
    teacher_model.to(device)

    logger.info(summary(teacher_model,input_size=(768,),depth=1,batch_dim=1, dtypes=[torch.IntTensor]))
    loss_dict = models.compile_model(teacher_model, args, stage=3)
    optimizer = torch.optim.Adam(teacher_model.parameters(),lr=3e-5, eps=1e-08)
    
    model_file = os.path.join(args["teacher_model_dir"], 'teacher_weights.pth')

    if os.path.exists(model_file):
        logger.info ("Loadings weights for fine-tuned model from {}".format(model_file))
        teacher_model.load_state_dict(torch.load(model_file,map_location=torch.device(device)))
    else:
        teacher_model, _ = train_model(teacher_model, train_dataset, dev_dataset, optimizer = optimizer, loss_dict =loss_dict,
                    batch_size= args["teacher_batch_size"], epochs=args["ft_epochs"], device=device, path_save =  os.path.join(args["teacher_model_dir"], 'teacher_weights_best.pth'))
        # callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=args["patience"], restore_best_weights=True)]
        torch.save(teacher_model.state_dict(), model_file)


   
    ner_evaluate(teacher_model, test_dataset , label_list, special_tokens, args["seq_len"], batch_size=args["teacher_batch_size"], device =device)
    word_emb = None
    if args["compress_word_embedding"]:
        word_emb = get_word_embedding(teacher_model._modules.get('encoder'), pt_tokenizer, args["hidden_size"])


    model_1 = models.construct_transformer_student_model(args, word_emb=word_emb)
    loss_dict1 = models.compile_model(model_1, args, stage=1)
    optimizer1 = torch.optim.Adam(model_1.parameters(),lr=3e-5, eps=1e-08)
    
    shared_layers = set()
    for name, param in model_1.named_parameters():
        if param.requires_grad:
            shared_layers.add(name.split(".")[0])
    shared_layers = list(shared_layers)
    logger.info ("Shared layers {}".format(shared_layers))

    best_model = None
    best_eval = 0
    min_loss = np.inf
    min_ckpt = None

    for stage in range(1, 2*len(shared_layers)+4):

        logger.info ("*** Starting stage {}".format(stage))
        patience_counter = 0

        #stage = 1, optimize representation loss (transfer set) with end-to-end training
        #stage = 2, copy model from stage = 1, and optimize logit loss (transfer set) with all but last layer frozen
        #stage = [3, 4, .., 2+num_shared_layers], optimize logit loss (transfer set) with gradual unfreezing
        #stage == 3+num_shared_layers, optimize CE loss (labeled data) with all but last layer frozen
        #stage = [4+num_shared_layers, ...], optimize CE loss (labeled data)with gradual unfreezing
        if stage == 2:
            #copy weights from model_stage_1
            logger.info ("Copying weights from model stage 1")
            for name, param in model_1.named_parameters():
                if name.split(".")[0] in shared_layers:
                    param.requires_grad = False
            loss_dict1 = models.compile_model(model_1, args, stage=2)
            optimizer1 = torch.optim.Adam(model_1.parameters(),lr=3e-5, eps=1e-08)
            #resetting min loss
            min_loss = np.inf

        elif stage > 2 and stage < 3+len(shared_layers):
            logger.info ("Unfreezing layer {}".format(shared_layers[stage-3]))
            for name, param in model_1.named_parameters():
                if name.split(".")[0] == shared_layers[stage-3]:
                    param.requires_grad = True
            loss_dict1 = models.compile_model(model_1, args, stage=3)
            optimizer1 = torch.optim.Adam(model_1.parameters(),lr=3e-5, eps=1e-08)

        elif stage == 3+len(shared_layers):
            for name, param in model_1.named_parameters():
                if name.split(".")[0] in shared_layers:
                    param.requires_grad = False
            loss_dict1 = models.compile_model(model_1, args, stage=3)
            optimizer1 = torch.optim.Adam(model_1.parameters(),lr=3e-5, eps=1e-08)
            #resetting min loss
            min_loss = np.inf

        elif stage > 3+len(shared_layers):
            logger.info ("Unfreezing layer {}".format(shared_layers[stage-4-len(shared_layers)]))
            for name, param in model_1.named_parameters():
                if name.split(".")[0] == shared_layers[stage-4-len(shared_layers)]:
                    param.requires_grad = True
            loss_dict1 = models.compile_model(model_1, args, stage=3)
            optimizer1 = torch.optim.Adam(model_1.parameters(),lr=3e-5, eps=1e-08)

        start_teacher = 0

        while start_teacher < len(X_unlabeled["input_ids"]) and stage < 3+len(shared_layers):

            end_teacher = min(start_teacher + args["distil_chunk_size"], len(X_unlabeled["input_ids"]))
            logger.info("Teacher indices from {} to {}".format(start_teacher, end_teacher))

            #get teacher logits
            input_data_chunk = {"input_ids": X_unlabeled["input_ids"][start_teacher:end_teacher], "attention_mask": X_unlabeled["attention_mask"][start_teacher:end_teacher], "token_type_ids": X_unlabeled["token_type_ids"][start_teacher:end_teacher]}
            unlabel_dataset = Dataset(input_data_chunk,None)
            
            model_file = os.path.join(args["model_dir"], "model_stage_{}_indx_{}.pth".format(stage, start_teacher))
            history_file = os.path.join(args["model_dir"], "model_stage_{}_indx_{}_history.txt".format(stage, start_teacher))
            
            if stage >= 1 and stage < 3+len(shared_layers):
                if os.path.exists(model_file):
                    logger.info ("Loadings weights for stage {} from {}".format(stage, model_file))
                    model_1.load_state_dict(torch.load(model_file))
                    history = load_history_file(history_file)
                else:
                    logger.info(summary(model_1,input_size=(384,),depth=1,batch_dim=1, dtypes=[torch.IntTensor]))

                    model_1, val_loss = train_model_student(teacher_model, model_1, unlabel_dataset, dev_dataset, batch_size= args["student_distil_batch_size"], optimizer = optimizer1, loss_dict =loss_dict1,\
                                                epochs=args["distil_epochs"], device=device, path_save=  model_file, opt_policy= args["opt_policy"], stage= stage)
                    
                    history = {"val_loss":val_loss}
                    save_history_file(history_file,history)
                    
                val_loss = history['val_loss']
                if  val_loss < min_loss:
                    min_loss = val_loss
                    min_ckpt = model_1.state_dict()
                    logger.info ("Checkpointing model weights with minimum validation loss {}".format(min_loss))
                    patience_counter = 0
                else:
                    patience_counter += 1
                    logger.info ("Resetting model to best weights found so far corresponding to val_loss {}".format(min_loss))
                    model_1.load_state_dict(min_ckpt)
                    if patience_counter == args["patience"]:
                        logger.info("Early stopping")
                        break
       
        if stage > 1 and stage < 3+len(shared_layers):
            
            cur_eval = ner_evaluate(model_1, test_dataset , label_list, special_tokens, args["seq_len"], batch_size=args["student_distil_batch_size"], device =device)
            
            if cur_eval >= best_eval:
                best_eval = cur_eval
                best_model_weights = model_1.state_dict()

        if stage >= 3+len(shared_layers):
            model_file = os.path.join(args["model_dir"], "model-stage-{}.pth".format(stage))
            history_file = os.path.join(args["model_dir"], "model-stage-{}-history.txt".format(stage))
            if os.path.exists(model_file):
                logger.info ("Loadings weights for stage 3 from {}".format(model_file))
                model_1.load_state_dict(torch.load(model_file))
            else:
                logger.info(summary(model_1,input_size=(384,),depth=1,batch_dim=1, dtypes=[torch.IntTensor]))

                model_1, val_loss  = train_model(model_1, train_dataset, dev_dataset, optimizer = optimizer1, loss_dict =loss_dict1,
                                batch_size= args["student_distil_batch_size"], epochs=args["ft_epochs"], device=device, path_save =  model_file)    
                history = {"val_loss":val_loss}
                save_history_file(history_file,history)

              
            cur_eval = ner_evaluate(model_1, test_dataset, label_list, special_tokens, args["seq_len"], batch_size=args["student_distil_batch_size"], device =device)
           

            if cur_eval >= best_eval:
                best_eval = cur_eval
                best_model_weights = model_1.state_dict()

    model_1.load_state_dict(best_model_weights)
    logger.info ("Best eval score {}".format(best_eval))

    #save xtremedistil training config and final model weights
    json.dump(args, open(os.path.join(args["model_dir"], "xtremedistil-config.json"), 'w'))
    torch.save(model_1.state_dict(), os.path.join(args["model_dir"], "xtremedistil.h5"))
    
    word_embeddings = model_1._modules.get('encoder').embeddings.word_embeddings.weight.cpu()
    if type(word_embeddings) != np.ndarray:
        word_embeddings = word_embeddings.numpy()
    np.save(open(os.path.join(args["model_dir"], "word_embedding.npy"), "wb"), word_embeddings)
    logger.info ("Model and config saved to {}".format(args["model_dir"]))