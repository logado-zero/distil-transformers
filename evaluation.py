"""
Author: Subho Mukherjee (submukhe@microsoft.com)
Code for XtremeDistil for distilling massive multi-lingual models.
"""

import conlleval
import logging
import numpy as np
from tqdm import tqdm 
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger('xtremedistil')

def validation(model, device, valid_loader, loss_function):

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

            outputs, _ = model(input_ids, attention_mask, token_type_ids)
            loss = 0
            if loss_function["num"] == 1:
              loss += loss_function["loss_name"](outputs, true)
            else:
              for i in range(loss_function["num"]):
                
                loss += loss_function["loss_name"](outputs[i], true[i])
            loss_total += loss

    return loss_total / len(valid_loader)


def train_model(model, train_dataset, dev_dataset, optimizer, loss_dict, batch_size=4, epochs =100, device ="cuda", path_save="./teacher_weights.pth"):
    training_generator = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_generator = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

    # Early stopping
    last_loss = 1000
    patience = 10
    triggertimes = 0

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

            outputs,_ = model(input_ids, attention_mask, token_type_ids)
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
        current_loss = validation(model, device, validation_generator, loss_dict)
        print('The Current Loss:', current_loss)

        if current_loss > last_loss:
            trigger_times += 1
            if trigger_times >= patience:
                print('Early stopping!\nStart to test process.')
                return model

        else:
            print('trigger times: 0')
            trigger_times = 0
            torch.save(model.state_dict(), path_save)
            print("Save best model ----> ", path_save)

        last_loss = current_loss
        
    

    return model



def ner_evaluate(model, test_dataset, labels, special_tokens, MAX_SEQUENCE_LENGTH, batch_size=32):

    test_generator = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model.eval()    

    for batch in tqdm(test_generator, total=len(test_generator), leave=False, desc="Predicting"):
        input_ids, attention_mask, token_type_ids = batch[0]["input_ids"].type(torch.LongTensor), batch[0]["attention_mask"].type(torch.LongTensor),\
                                                    batch[0]["token_type_ids"].type(torch.LongTensor)
        true = batch[1].type(torch.LongTensor)
        
        input_ids, attention_mask, token_type_ids = input_ids.to('cpu'), attention_mask.to('cpu'), token_type_ids.to('cpu')
        true = true.to('cpu')

        outputs,_ = model.forward(input_ids, attention_mask, token_type_ids)
        pred_tags_all = []
        true_tags_all = []
        for i, seq in enumerate(outputs):
            for j in range(MAX_SEQUENCE_LENGTH):
                indx = true[i][j]
                true_label = labels[indx]
                if special_tokens["pad_token"] in true_label or special_tokens["bos_token"] in true_label or special_tokens["eos_token"] in true_label:
                    continue

                true_tags_all.append(true_label)
                indx = np.argmax(seq[j])
                pred_label = labels[indx]
                pred_tags_all.append(pred_label)



    prec, rec, f1 = conlleval.evaluate(true_tags_all, pred_tags_all, special_tokens, verbose=True)
    logger.info ("Test scores {} {} {}".format(prec, rec, f1))

    return np.mean(f1)