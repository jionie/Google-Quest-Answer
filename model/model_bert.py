from transformers import *
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

############################################ Define Net Class
class QuestNet(nn.Module):
    def __init__(self, model_type="xlnet-base-cased", tokenizer=None, n_classes=30, n_category_classes=5, \
                n_host_classes=64, hidden_layers=[-1, -3, -5, -7], extra_token=False):
        super(QuestNet, self).__init__()
        self.model_name = 'QuestModel'
        self.model_type = model_type
        self.hidden_layers = hidden_layers
        self.extra_token = extra_token
        
        if model_type == "bert-base-uncased":
            self.bert_model = BertModel.from_pretrained(model_type, hidden_dropout_prob=0, \
                                                    output_hidden_states=True)   
            self.hidden_size = 768
        elif model_type == "bert-large-uncased":
            self.bert_model = BertModel.from_pretrained(model_type, hidden_dropout_prob=0., \
                                                    output_hidden_states=True)   
            self.hidden_size = 1024
        elif model_type == "bert-large-cased":
            self.bert_model = BertModel.from_pretrained(model_type, hidden_dropout_prob=0, \
                                                    output_hidden_states=True)   
            self.hidden_size = 1024
        elif model_type == "bert-base-cased":
            self.bert_model = BertModel.from_pretrained(model_type, hidden_dropout_prob=0, \
                                                    output_hidden_states=True)   
            self.hidden_size = 768
        elif model_type == "flaubert-base-cased":
            self.flaubert_model = FlaubertModel.from_pretrained(model_type, dropout=0, \
                                                    output_hidden_states=True)   
            self.hidden_size = 768
        elif model_type == "flaubert-large-cased":
            self.flaubert_model = FlaubertModel.from_pretrained(model_type, dropout=0, \
                                                    output_hidden_states=True)   
            self.hidden_size = 1024
        elif model_type == "flaubert-base-uncased":
            self.flaubert_model = FlaubertModel.from_pretrained(model_type, dropout=0, \
                                                    output_hidden_states=True)   
            self.hidden_size = 768
        elif model_type == "xlnet-base-cased":
            self.xlnet_model = XLNetModel.from_pretrained(model_type, dropout=0, output_hidden_states=True)   
            self.hidden_size = 768
        elif model_type == "xlnet-large-cased":
            self.xlnet_model = XLNetModel.from_pretrained(model_type, dropout=0, output_hidden_states=True)   
            self.hidden_size = 1024
        elif model_type == "roberta-base":
            self.roberta_model = RobertaModel.from_pretrained(model_type, hidden_dropout_prob=0, output_hidden_states=True)
            self.roberta_model.resize_token_embeddings(len(tokenizer)) 
            self.hidden_size = 768
        elif model_type == "albert-base-v2":
            self.albert_model = AlbertModel.from_pretrained(model_type, hidden_dropout_prob=0, output_hidden_states=True)
            self.hidden_size = 768
        elif model_type == "albert-large-v2":
            self.albert_model = AlbertModel.from_pretrained(model_type, hidden_dropout_prob=0, output_hidden_states=True)
            self.hidden_size = 1024
        elif model_type == "albert-xlarge-v2":
            self.albert_model = AlbertModel.from_pretrained(model_type, hidden_dropout_prob=0, output_hidden_states=True)
            self.hidden_size = 1024 * 2
        elif model_type == "albert-xxlarge-v2":
            self.albert_model = AlbertModel.from_pretrained(model_type, hidden_dropout_prob=0, output_hidden_states=True)
            self.hidden_size = 1024 * 4
        elif model_type == "gpt2":
            self.gpt2_model = GPT2Model.from_pretrained(model_type, initializer_range=0, output_hidden_states=True)
            self.hidden_size = 768
            # OpenAIGPTModel
        else:
            raise NotImplementedError
        
        self.fc_1 = nn.Linear(self.hidden_size * len(hidden_layers), self.hidden_size)
        self.fc = nn.Linear(self.hidden_size, n_classes)
        
        if self.extra_token:
            self.fc_1_category = nn.Linear(self.hidden_size * len(hidden_layers), self.hidden_size)
            self.fc_category = nn.Linear(self.hidden_size, n_category_classes)
            
            self.fc_1_host = nn.Linear(self.hidden_size * len(hidden_layers), self.hidden_size)
            self.fc_host = nn.Linear(self.hidden_size, n_host_classes)
        
        self.selu = nn.SELU()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])
#         self.fcs_1 = nn.ModuleList(
#             [ nn.Linear(self.hidden_size, self.hidden_size) for _ in range(len(hidden_layers)) ])


    def get_hidden_states_by_index(self, hidden_states, index):
        
        # concat hidden
        for i in range(len(self.hidden_layers)):
            if i == 0:
                hidden_layer = self.hidden_layers[i]
                hidden_state = hidden_states[hidden_layer][:, index]
                fuse_hidden = torch.unsqueeze(hidden_state, dim=-1) # N * 768 * 1
            else:
                hidden_layer = self.hidden_layers[i]
                hidden_state = hidden_states[hidden_layer][:, index]
                h = torch.unsqueeze(hidden_state, dim=-1) # N * 768 * 1
                fuse_hidden = torch.cat([fuse_hidden, h], dim=-1)
                
        fuse_hidden = fuse_hidden.reshape(fuse_hidden.shape[0], -1)
        
        return fuse_hidden
    
    def get_hidden_states_by_mean(self, hidden_states):
        
        # concat hidden
        for i in range(len(self.hidden_layers)):
            if i == 0:
                hidden_layer = self.hidden_layers[i]
                hidden_state = torch.mean(hidden_states[hidden_layer], dim=1)
                fuse_hidden = torch.unsqueeze(hidden_state, dim=-1) # N * 768 * 1
            else:
                hidden_layer = self.hidden_layers[i]
                hidden_state = torch.mean(hidden_states[hidden_layer], dim=1)
                h = torch.unsqueeze(hidden_state, dim=-1) # N * 768 * 1
                fuse_hidden = torch.cat([fuse_hidden, h], dim=-1)
                
        fuse_hidden = fuse_hidden.reshape(fuse_hidden.shape[0], -1)
        
        return fuse_hidden
    
    def get_logits_by_random_dropout(self, fuse_hidden, fc_1, fc):
        
        h = self.relu(fc_1(fuse_hidden))
        
        for j, dropout in enumerate(self.dropouts):
            
            if j == 0:
                logit = fc(dropout(h))
            else:
                logit += fc(dropout(h))
                
        return logit / len(self.dropouts)

    def forward(self, ids, seg_ids):
        attention_mask = (ids > 0)
        
        if ((self.model_type == "bert-base-uncased") \
            or (self.model_type == "bert-base-cased") \
            or (self.model_type == "bert-large-uncased") \
            or (self.model_type == "bert-large-cased")):
        
            outputs = self.bert_model(input_ids=ids, token_type_ids=seg_ids, attention_mask=attention_mask)
            hidden_states = outputs[2]
            
            # pooled_out = outputs[1] #  N * 768
            # sequence_out = torch.unsqueeze(outputs[0][:, 0], dim=-1) # N * 512 * 768 * 1, hidden_states[-1]
            # 13 (embedding + 12 transformers) for base
            # 26 (embedding + 25 transformers) for large
            
            fuse_hidden = self.get_hidden_states_by_index(hidden_states, 0)
            logits = self.get_logits_by_random_dropout(fuse_hidden, self.fc_1, self.fc)
            
            if self.extra_token:
                fuse_hidden_category = self.get_hidden_states_by_index(hidden_states, 1)
                fuse_hidden_host = self.get_hidden_states_by_index(hidden_states, 2)
                
                logits_category = self.get_logits_by_random_dropout(fuse_hidden_category, self.fc_1_category, self.fc_category)
                logits_host = self.get_logits_by_random_dropout(fuse_hidden_host, self.fc_1_host, self.fc_host)
                
        elif ((self.model_type == "flaubert-base-cased") \
            or (self.model_type == "flaubert-base-uncased") \
            or (self.model_type == "flaubert-large-cased")):
        
            outputs = self.flaubert_model(input_ids=ids, token_type_ids=seg_ids, attention_mask=attention_mask)
            hidden_states = outputs[1]
            
            # last_hidden_out = outputs[0]
            # mem = outputs[1], when config.mem_len > 0
            
            fuse_hidden = self.get_hidden_states_by_index(hidden_states, 0)
            logits = self.get_logits_by_random_dropout(fuse_hidden, self.fc_1, self.fc)
            
            if self.extra_token:
                fuse_hidden_category = self.get_hidden_states_by_index(hidden_states, 1)
                fuse_hidden_host = self.get_hidden_states_by_index(hidden_states, 2)
                
                logits_category = self.get_logits_by_random_dropout(fuse_hidden_category, self.fc_1_category, self.fc_category)
                logits_host = self.get_logits_by_random_dropout(fuse_hidden_host, self.fc_1_host, self.fc_host)
            
            
        elif ((self.model_type == "xlnet-base-cased") \
            or (self.model_type == "xlnet-large-cased")):

            attention_mask = attention_mask.float()
            outputs = self.xlnet_model(input_ids=ids, token_type_ids=seg_ids, attention_mask=attention_mask)
            hidden_states = outputs[1]
            
            # last_hidden_out = outputs[0]
            # mem = outputs[1], when config.mem_len > 0
            
            fuse_hidden = self.get_hidden_states_by_index(hidden_states, 0)
            logits = self.get_logits_by_random_dropout(fuse_hidden, self.fc_1, self.fc)
            
            if self.extra_token:
                fuse_hidden_category = self.get_hidden_states_by_index(hidden_states, 1)
                fuse_hidden_host = self.get_hidden_states_by_index(hidden_states, 2)
                
                logits_category = self.get_logits_by_random_dropout(fuse_hidden_category, self.fc_1_category, self.fc_category)
                logits_host = self.get_logits_by_random_dropout(fuse_hidden_host, self.fc_1_host, self.fc_host)
            
        elif (self.model_type == "roberta-base"):

            attention_mask = attention_mask.float()
            outputs = self.roberta_model(input_ids=ids, attention_mask=attention_mask)
            # outputs = self.roberta_model(input_ids=ids, attention_mask=attention_mask)
            hidden_states = outputs[2]
            
            fuse_hidden = self.get_hidden_states_by_index(hidden_states, 0)
            logits = self.get_logits_by_random_dropout(fuse_hidden, self.fc_1, self.fc)
            
            if self.extra_token:
                fuse_hidden_category = self.get_hidden_states_by_index(hidden_states, 1)
                fuse_hidden_host = self.get_hidden_states_by_index(hidden_states, 2)
                
                logits_category = self.get_logits_by_random_dropout(fuse_hidden_category, self.fc_1_category, self.fc_category)
                logits_host = self.get_logits_by_random_dropout(fuse_hidden_host, self.fc_1_host, self.fc_host)
        
        elif ((self.model_type == "albert-base-v2") \
            or (self.model_type == "albert-large-v2") \
            or (self.model_type == "albert-xlarge-v2") \
            or (self.model_type == "albert-xxlarge-v2") ):

            attention_mask = attention_mask.float()
            outputs = self.albert_model(input_ids=ids, token_type_ids=seg_ids, attention_mask=attention_mask)
            hidden_states = outputs[2]
            
            fuse_hidden = self.get_hidden_states_by_index(hidden_states, 0)
            logits = self.get_logits_by_random_dropout(fuse_hidden, self.fc_1, self.fc)
            
            if self.extra_token:
                fuse_hidden_category = self.get_hidden_states_by_index(hidden_states, 1)
                fuse_hidden_host = self.get_hidden_states_by_index(hidden_states, 2)
                
                logits_category = self.get_logits_by_random_dropout(fuse_hidden_category, self.fc_1_category, self.fc_category)
                logits_host = self.get_logits_by_random_dropout(fuse_hidden_host, self.fc_1_host, self.fc_host)
            
        
        if self.extra_token:     
            return logits, logits_category, logits_host
        else:
            return logits
        
        

############################################ Define test Net function
def test_Net(extra_token=True):
    print("------------------------testing Net----------------------")

    x = torch.tensor([[1, 2, 3, 4, 5, 0, 0], [1, 2, 3, 4, 5, 0, 0]])
    seg_ids = torch.tensor([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]])
    model = QuestNet(extra_token=True)

    if extra_token:
        y, y_category, y_host = model(x, seg_ids)
        print(y)
        print(y_category)
        print(y_host)
    else:
        y = model(x, seg_ids)
        print(y)
    print("------------------------testing Net finished----------------------")

    return


if __name__ == "__main__":
    print("------------------------testing Net without extra token----------------------")
    test_Net(extra_token=False)
    print("------------------------testing Net with extra token----------------------")
    test_Net(extra_token=True)
    
