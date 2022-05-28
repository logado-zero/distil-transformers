"""
Author: Subho Mukherjee (submukhe@microsoft.com)
Code for XtremeDistil for distilling massive multi-lingual models.
"""

from sklearn.decomposition import dict_learning_online
import conlleval
import logging
import time
import numpy as np
from tqdm import tqdm 
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

logger = logging.getLogger('xtremedistil')

def validation(model, device, valid_loader, loss_function, stage = None):

    model.eval()
    loss_total = 0

    # Test validation data
    with torch.no_grad():
        pr = tqdm(valid_loader, total=len(valid_loader), leave=False,desc="Validate ....")
        for batch in pr:
            
            input_ids, attention_mask, token_type_ids = batch[0]["input_ids"].type(torch.LongTensor), \
                                            batch[0]["attention_mask"].type(torch.LongTensor),batch[0]["token_type_ids"].type(torch.LongTensor)
            true = batch[1].type(torch.LongTensor)
            
            input_ids, attention_mask, token_type_ids = input_ids.to(device), attention_mask.to(device), token_type_ids.to(device)
            true = true.to(device)

            if stage == None:
                outputs,_ = model(input_ids, attention_mask, token_type_ids)
            else: outputs = model(input_ids, attention_mask, token_type_ids, stage=stage)
            
            loss = 0
            if loss_function["num"] == 1:
              loss += loss_function["loss_name"](outputs, true)
            else:
              for i in range(loss_function["num"]):
                
                loss += loss_function["loss_name"](outputs[i], true[i])
            loss_total += loss

    return loss_total / len(valid_loader)

def validation_student(teacher_model, student_model, device, valid_loader, loss_function, stage= 3):
    teacher_model.eval()
    student_model.eval()
    loss_total = 0

    # Test validation data
    with torch.no_grad():
        pr = tqdm(valid_loader, total=len(valid_loader), leave=False,desc="Validate ....")
        for batch in pr:
            
            input_ids, attention_mask, token_type_ids = batch[0]["input_ids"].type(torch.LongTensor), \
                                            batch[0]["attention_mask"].type(torch.LongTensor),batch[0]["token_type_ids"].type(torch.LongTensor)
            true = batch[1].type(torch.LongTensor)
            
            input_ids, attention_mask, token_type_ids = input_ids.to(device), attention_mask.to(device), token_type_ids.to(device)
            if stage == 1:
                _, output_teacher = teacher_model.forward(input_ids, attention_mask, token_type_ids)
            else:  output_teacher, _ = teacher_model.forward(input_ids, attention_mask, token_type_ids)

            outputs = student_model(input_ids, attention_mask, token_type_ids, stage=stage)
            loss = 0
            if loss_function["num"] == 1:
                    loss += loss_function["loss_name"](outputs, torch.argmax(output_teacher, dim =2))
            else:
                for i in range(loss_function["num"]):
                    
                    loss += loss_function["loss_name"](outputs[i], output_teacher[i])
            loss_total += loss

    return loss_total / len(valid_loader)


def train_model(model, train_dataset, dev_dataset, optimizer, loss_dict, args, batch_size=4, epochs =100, device ="cuda",\
     path_save="./teacher_weights.pth", opt_policy = False, stage = None, name_model = "teacher"):

    training_generator = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_generator = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

    # Early stopping
    last_loss = 0
    patience = 10
    triggertimes = 0
    #Set up mixed precision
    if opt_policy:
        scaler = GradScaler()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        idx = 0
        pr = tqdm(training_generator, total=len(training_generator), leave=False)
        for batch in pr:
            idx += 1
            input_ids, attention_mask, token_type_ids = batch[0]["input_ids"].type(torch.LongTensor), \
                                            batch[0]["attention_mask"].type(torch.LongTensor),batch[0]["token_type_ids"].type(torch.LongTensor)
            true = batch[1].type(torch.LongTensor)
            
            input_ids, attention_mask, token_type_ids = input_ids.to(device), attention_mask.to(device), token_type_ids.to(device)
            true = true.to(device)

            if opt_policy:
                with autocast():
                    if stage == None:
                        outputs,_ = model(input_ids, attention_mask, token_type_ids)
                    else: outputs = model(input_ids, attention_mask, token_type_ids, stage=stage)
                    loss = 0
                    if loss_dict["num"] == 1:
                        loss += loss_dict["loss_name"](outputs, true)
                    else:
                        for i in range(loss_dict["num"]):
                            loss += loss_dict["loss_name"](outputs[i], true[i])
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            else:
                if stage == None:
                    outputs,_ = model(input_ids, attention_mask, token_type_ids)
                else: outputs = model(input_ids, attention_mask, token_type_ids, stage=stage)
                loss = 0
                if loss_dict["num"] == 1:
                    loss += loss_dict["loss_name"](outputs, true)
                else:
                    for i in range(loss_dict["num"]):
                        loss += loss_dict["loss_name"](outputs[i], true[i])

                loss.backward()
                optimizer.step()

            optimizer.zero_grad()
            loss = loss.data.cpu().tolist()
            epoch_loss += loss
            pr.set_description("train loss: {}".format(epoch_loss / idx))
            torch.cuda.empty_cache()

        logging.info("\nEpoch {}, average train epoch loss={:.5}\n".format(epoch, epoch_loss / idx))

        # Early stopping
        # current_loss = validation(model, device, validation_generator, loss_dict, stage= stage)
        # print('The Current Loss:', current_loss)
        current_loss = ner_evaluate(model, dev_dataset , args["label_list"], args['special_tokens'] , args["seq_len"], batch_size=args["teacher_batch_size"], device =device,\
                                        name_model = name_model)
        print('The F1-score valid:', current_loss)

        if current_loss < last_loss:
            trigger_times += 1
            if trigger_times >= patience:
                print('Early stopping!\nStart to test process.')
                return model, last_loss

        else:
            print('trigger times: 0')
            trigger_times = 0
            torch.save(model.state_dict(), path_save)
            print("Save best model ----> ", path_save)

            last_loss = current_loss
        
    

    return model, last_loss



def ner_evaluate(model, test_dataset, labels, special_tokens, MAX_SEQUENCE_LENGTH, batch_size=32, device="cuda", name_model = "teacher", stage = None, check_time_process = None):

    test_generator = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model.to(device)
    model.eval()    
    pred_tags_all = []
    true_tags_all = []
    duration = 0
    times = 0
   
    for batch in tqdm(test_generator, total=len(test_generator), leave=False, desc="Predicting"):
        input_ids, attention_mask, token_type_ids = batch[0]["input_ids"].type(torch.LongTensor), batch[0]["attention_mask"].type(torch.LongTensor),\
                                                    batch[0]["token_type_ids"].type(torch.LongTensor)
        true = batch[1].type(torch.LongTensor)
        
        input_ids, attention_mask, token_type_ids = input_ids.to(device), attention_mask.to(device), token_type_ids.to(device)
        true = true.to(device)
        if check_time_process:
            time_start = time.time()
        if name_model == "teacher":
            outputs,_ = model.forward(input_ids, attention_mask, token_type_ids)
        else:
            outputs = model.forward(input_ids, attention_mask, token_type_ids, stage=stage)
        if check_time_process:
            time_end = time.time()
            duration += time_end - time_start
            times += 1
        for i, seq in enumerate(outputs):
            for j in range(MAX_SEQUENCE_LENGTH):
                indx = true[i][j]
                true_label = labels[indx]
                if special_tokens["pad_token"] in true_label or special_tokens["bos_token"] in true_label or special_tokens["eos_token"] in true_label:
                    continue

                true_tags_all.append(true_label)
                indx = torch.argmax(seq[j])
                pred_label = labels[indx]
                pred_tags_all.append(pred_label)
    if check_time_process:
        logger.info("Process time for running test dataset: {} s".format(duration/times))


    prec, rec, f1 = conlleval.evaluate(true_tags_all, pred_tags_all, special_tokens, verbose=True)
    logger.info ("Test scores {} {} {}".format(prec, rec, f1))

    return np.mean(f1)

def train_model_student(teacher_model, student_model, train_dataset, dev_dataset, optimizer, loss_dict, batch_size=4, epochs =100, device ="cuda",\
     path_save="./student_weights.pth", opt_policy = False, stage = 2):

    unlabel_generator = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_generator = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

    teacher_model.to(device)
    student_model.to(device)
    teacher_model.eval() 

    # Early stopping
    last_loss = 1000
    patience = 10
    triggertimes = 0
    #Set up mixed precision
    if opt_policy:
        scaler = GradScaler()

    for epoch in range(epochs):
        student_model.train()
        epoch_loss = 0
        idx = 0
        pr = tqdm(unlabel_generator, total=len(unlabel_generator), leave=False, desc="Training Student Model for unlabel dataset")
        for batch in pr:
            
            idx += 1
            input_ids, attention_mask, token_type_ids = batch["input_ids"].type(torch.LongTensor), batch["attention_mask"].type(torch.LongTensor),\
                                                        batch["token_type_ids"].type(torch.LongTensor)
  
            input_ids, attention_mask, token_type_ids = input_ids.to(device), attention_mask.to(device), token_type_ids.to(device)

            if opt_policy:
                with autocast():

                    if stage == 1:
                        _, output_teacher = teacher_model.forward(input_ids, attention_mask, token_type_ids)
                    else:  output_teacher, _ = teacher_model.forward(input_ids, attention_mask, token_type_ids)

                    outputs = student_model(input_ids, attention_mask, token_type_ids, stage=stage)
                    loss = 0
                    if loss_dict["num"] == 1:
                            loss += loss_dict["loss_name"](outputs, torch.argmax(output_teacher, dim =2))
                    else:
                        for i in range(loss_dict["num"]):
                            loss += loss_dict["loss_name"](outputs[i], output_teacher[i])
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            else:

                if stage == 1:
                    _, output_teacher = teacher_model.forward(input_ids, attention_mask, token_type_ids)
                else:  output_teacher, _ = teacher_model.forward(input_ids, attention_mask, token_type_ids)

                outputs = student_model(input_ids, attention_mask, token_type_ids, stage=stage)
                loss = 0
                if loss_dict["num"] == 1:
                    if stage < 3:
                        loss += loss_dict["loss_name"](outputs, output_teacher)
                    else:
                        loss += loss_dict["loss_name"](outputs, torch.argmax(output_teacher, dim =2))
                else:
                    for i in range(loss_dict["num"]):
                        loss += loss_dict["loss_name"](outputs[i], output_teacher[i])

                loss.backward()
                optimizer.step()

            optimizer.zero_grad()
            loss = loss.data.cpu().tolist()
            epoch_loss += loss
            pr.set_description("train loss: {}".format(epoch_loss / idx))
            torch.cuda.empty_cache()

        logging.info("\nEpoch {}, average train epoch loss={:.5}\n".format(epoch, epoch_loss / idx))

        # Early stopping
        current_loss = validation_student(teacher_model,student_model, device, validation_generator, loss_dict, stage=stage)
        print('The Current Loss:', current_loss)

        if current_loss > last_loss:
            trigger_times += 1
            if trigger_times >= patience:
                print('Early stopping!\nStart to test process.')
                return student_model, last_loss

        else:
            print('trigger times: 0')
            trigger_times = 0
            torch.save(student_model.state_dict(), path_save)
            print("Save best model ----> ", path_save)

            last_loss = current_loss
        
    

    return student_model, last_loss


def load_history_file(path: str):
    with open(path, 'r') as f:
        file_read = f.readlines()
    history = {}
    for i in file_read:
        line = i.strip().split(": ")
        x = {str(line[0]):float(line[1])}
        history.update(x)

    return history

def save_history_file(path: str, history: dict):
    f = open(path,"w")
    for key in history.keys():
        f.write(str(key)+ ": "+str(history.get(key))+"\n")

    f.close()


