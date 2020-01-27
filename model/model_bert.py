from transformers import *
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

############################################ Define Net Class
class QuestNet(nn.Module):
    def __init__(self, model_type="bert-base-uncased", n_classes=30, hidden_layers=[-1, -3, -5, -7]):
        super(QuestNet, self).__init__()
        self.model_name = 'QuestModel'
        self.model_type = model_type
        self.hidden_layers = hidden_layers
        
        if model_type == "bert-base-uncased":
            self.bert_model = BertModel.from_pretrained(model_type, hidden_dropout_prob=0.1, \
                                                    output_hidden_states=True, force_download=True)   
            self.hidden_size = 768
        elif model_type == "bert-large-uncased":
            self.bert_model = BertModel.from_pretrained(model_type, hidden_dropout_prob=0.1, \
                                                    output_hidden_states=True, force_download=True)   
            self.hidden_size = 1024
        elif model_type == "bert-base-cased":
            self.bert_model = BertModel.from_pretrained(model_type, hidden_dropout_prob=0.1, \
                                                    output_hidden_states=True, force_download=True)   
            self.hidden_size = 768
        elif model_type == "xlnet-base-cased":
            self.xlnet_model = XLNetModel.from_pretrained(model_type, dropout=0.1, output_hidden_states=True)   
            self.hidden_size = 768
        elif model_type == "xlnet-large-cased":
            self.xlnet_model = XLNetModel.from_pretrained(model_type, dropout=0.1, output_hidden_states=True)   
            self.hidden_size = 1024
        elif model_type == "roberta-base":
            self.roberta_model = RobertaModel.from_pretrained(model_type, hidden_dropout_prob=0.1, output_hidden_states=True, force_download=True)   
            self.hidden_size = 768
        elif model_type == "albert-base-v2":
            self.albert_model = AlbertModel.from_pretrained(model_type, hidden_dropout_prob=0.01, output_hidden_states=True, force_download=True)
            self.hidden_size = 768
        elif model_type == "gpt2":
            self.gpt2_model = GPT2Model.from_pretrained(model_type, initializer_range=0.02, output_hidden_states=True, force_download=True)
            self.hidden_size = 768
            # OpenAIGPTModel
        else:
            raise NotImplementedError
        
        self.fc_1 = nn.Linear(self.hidden_size * len(hidden_layers), self.hidden_size)
        self.fc = nn.Linear(self.hidden_size, n_classes)
        
        self.selu = nn.SELU()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])
#         self.fcs_1 = nn.ModuleList(
#             [ nn.Linear(self.hidden_size, self.hidden_size) for _ in range(len(hidden_layers)) ])

    def forward(self, ids, seg_ids):
        attention_mask = (ids > 0)
        
        if ((self.model_type == "bert-base-uncased") \
            or (self.model_type == "bert-base-cased") \
            or (self.model_type == "bert-large-uncased") \
            or (self.model_type == "bert-large-cased")):
        
            outputs = self.bert_model(input_ids=ids, token_type_ids=seg_ids, attention_mask=attention_mask)
            hidden_states = outputs[2]
            print('-*-*-*-*-*-*-*-', hidden_states[0].shape, len(hidden_states), hidden_states[0][:, 0].shape)
            # print('******************************', type(hidden_states))
            
            # pooled_out = outputs[1] #  N * 768
        
            # sequence_out = torch.unsqueeze(outputs[0][:, 0], dim=-1) # N * 512 * 768 * 1, hidden_states[-1]
            # fuse_hidden = sequence_out
            
            # 13 (embedding + 12 transformers) for base
            # 26 (embedding + 25 transformers) for large
            
            # concat hidden
            for i in range(len(self.hidden_layers)):
                if i == 0:
                    hidden_layer = self.hidden_layers[i]
                    # hidden_state = torch.mean(hidden_states[hidden_layer], dim=1)
                    hidden_state = hidden_states[hidden_layer][:, 0]
                    fuse_hidden = torch.unsqueeze(hidden_state, dim=-1) # N * 768 * 1
                else:
                    hidden_layer = self.hidden_layers[i]
                    # hidden_state = torch.mean(hidden_states[hidden_layer], dim=1)
                    hidden_state = hidden_states[hidden_layer][:, 0]
                    h = torch.unsqueeze(hidden_state, dim=-1) # N * 768 * 1
                    fuse_hidden = torch.cat([fuse_hidden, h], dim=-1)
                    
            fuse_hidden = fuse_hidden.reshape(fuse_hidden.shape[0], -1)
            h = self.relu(self.fc_1(fuse_hidden))
        
        elif ((self.model_type == "xlnet-base-cased") \
            or (self.model_type == "xlnet-large-cased")):

            attention_mask = attention_mask.float()
            outputs = self.xlnet_model(input_ids=ids, token_type_ids=seg_ids, attention_mask=attention_mask)
            hidden_states = outputs[1]
            
            # last_hidden_out = outputs[0]
            # mem = outputs[1], when config.mem_len > 0
            
            # concat hidden, summary_type="last", first_dropout = 0
            for i in range(len(self.hidden_layers)):
                if i == 0:
                    hidden_layer = self.hidden_layers[i]
                    # hidden_state = torch.mean(hidden_states[hidden_layer], dim=1)
                    hidden_state = hidden_states[hidden_layer][:, -1]
                    fuse_hidden = torch.unsqueeze(hidden_state, dim=-1) # N * 768 * 1
                else:
                    hidden_layer = self.hidden_layers[i]
                    # hidden_state = torch.mean(hidden_states[hidden_layer], dim=1)
                    hidden_state = hidden_states[hidden_layer][:, -1]
                    h = torch.unsqueeze(hidden_state, dim=-1) # N * 768 * 1
                    fuse_hidden = torch.cat([fuse_hidden, h], dim=-1)
        
            fuse_hidden = fuse_hidden.reshape(fuse_hidden.shape[0], -1)
            h = self.relu(self.fc_1(fuse_hidden))
        elif (self.model_type == "roberta-base"):
            attention_mask = (ids>0).float()

            # layers, pool_out = self.roberta_model(input_ids=ids, token_type_ids=seg_ids, attention_mask=attention_mask)

            outputs = self.roberta_model(input_ids=ids, attention_mask=attention_mask, token_type_ids=seg_ids)#, attention_mask=attention_mask)
            hidden_states = outputs[2]
            # print('******************************', type(hidden_states))
            
            # last_hidden_out = outputs[0]
            # mem = outputs[1], when config.mem_len > 0
            
            # concat hidden, summary_type="last", first_dropout = 0
            for i in range(len(self.hidden_layers)):
                if i == 0:
                    hidden_layer = self.hidden_layers[i]
                    # hidden_state = torch.mean(hidden_states[hidden_layer], dim=1)
                    hidden_state = hidden_states[hidden_layer][:, -1]
                    fuse_hidden = torch.unsqueeze(hidden_state, dim=-1) # N * 768 * 1
                else:
                    hidden_layer = self.hidden_layers[i]
                    # hidden_state = torch.mean(hidden_states[hidden_layer], dim=1)
                    hidden_state = hidden_states[hidden_layer][:, -1]
                    h = torch.unsqueeze(hidden_state, dim=-1) # N * 768 * 1
                    fuse_hidden = torch.cat([fuse_hidden, h], dim=-1)
            fuse_hidden = fuse_hidden.reshape(fuse_hidden.shape[0], -1)
            h = self.relu(self.fc_1(fuse_hidden))
        elif (self.model_type == "albert-base-v2"):

            # attention_mask = attention_mask.float()
            outputs = self.albert_model(input_ids=ids, token_type_ids=seg_ids, attention_mask=attention_mask)
            hidden_states = outputs[2]
            # print('-*-*-*-*-*-*-*-', hidden_states[0].shape, hidden_states[0][:, 0, :].shape)
            # 
            # last_hidden_out = outputs[0]
            # mem = outputs[1], when config.mem_len > 0
            
            # concat hidden, summary_type="last", first_dropout = 0
            for i in range(len(self.hidden_layers)):
                if i == 0:
                    hidden_layer = self.hidden_layers[i]
                    hidden_state = hidden_states[hidden_layer][:, 0, :]
                    fuse_hidden = torch.unsqueeze(hidden_state, dim=-1) # N * 768 * 1
                    # fuse_hidden = torch.mean(hidden_states[hidden_layer], dim=1, keepdim=True).reshape((-1, 1, 768))
                else:
                    hidden_layer = self.hidden_layers[i]
                    # hidden_state = torch.mean(hidden_states[hidden_layer], dim=1)
                    hidden_state = hidden_states[hidden_layer][:, 0, :]
                    h = torch.unsqueeze(hidden_state, dim=-1) # N * 768 * 1
                    # h = torch.mean(hidden_states[hidden_layer], dim=1, keepdim=True).reshape((-1, 1, 768))
                    fuse_hidden = torch.cat([fuse_hidden, h], dim=-1)
                    
            fuse_hidden = fuse_hidden.reshape(fuse_hidden.shape[0], -1)
            h = torch.sigmoid(self.fc_1(fuse_hidden))
        elif (self.model_type == "gpt2"):

            attention_mask = attention_mask.float()
            outputs = self.gpt2_model(input_ids=ids, token_type_ids=seg_ids, attention_mask=attention_mask)
            hidden_states = outputs[2]
            # print('-*-*-*-*-*-*-*-', hidden_states.shape)
            
            # last_hidden_out = outputs[0]
            # mem = outputs[1], when config.mem_len > 0
            
            # concat hidden, summary_type="last", first_dropout = 0
            for i in range(len(self.hidden_layers)):
                if i == 0:
                    hidden_layer = self.hidden_layers[i]
                    # hidden_state = torch.mean(hidden_states[hidden_layer], dim=1)
                    hidden_state = hidden_states[hidden_layer][:, 0]
                    fuse_hidden = torch.unsqueeze(hidden_state, dim=-1) # N * 768 * 1
                else:
                    hidden_layer = self.hidden_layers[i]
                    # hidden_state = torch.mean(hidden_states[hidden_layer], dim=1)
                    hidden_state = hidden_states[hidden_layer][:, 0]
                    h = torch.unsqueeze(hidden_state, dim=-1) # N * 768 * 1
                    fuse_hidden = torch.cat([fuse_hidden, h], dim=-1)
                    
            fuse_hidden = fuse_hidden.reshape(fuse_hidden.shape[0], -1)
            h = self.relu(self.fc_1(fuse_hidden))
            
        for j, dropout in enumerate(self.dropouts):

            if j == 0:
                logit = self.fc(dropout(h))
            else:
                logit += self.fc(dropout(h))
        # logit = self.fc(h)
        
        return logit / len(self.dropouts)
        
        

############################################ Define test Net function
def test_Net():
    print("------------------------testing Net----------------------")

    x = torch.tensor([[1, 2, 3, 4, 5, 0, 0], [1, 2, 3, 4, 5, 0, 0]])
    seg_ids = torch.tensor([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]])
    model = QuestNet()

    y = model(x, seg_ids)
    print(y)

    print("------------------------testing Net finished----------------------")

    return


if __name__ == "__main__":
    test_Net()
