from transformers import *
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

############################################ Define SubModel Class
def gelu(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initially created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def gelu_new(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def swish(x):
    return x * torch.sigmoid(x)


def mish(x):
    return x * torch.tanh(nn.functional.softplus(x))


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish, "gelu_new": gelu_new, "mish": mish}


BertLayerNorm = torch.nn.LayerNorm

class BertPredictionHeadTransform(nn.Module):
    def __init__(self, hidden_size, hidden_act="gelu", layer_norm_eps=1e-12):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        if isinstance(hidden_act, str):
            self.transform_act_fn = ACT2FN[hidden_act]
        else:
            self.transform_act_fn = hidden_act
        self.LayerNorm = BertLayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertClassificationHead(nn.Module):
    def __init__(self, hidden_size, hidden_act="gelu", layer_norm_eps=1e-12):
        super(BertClassificationHead, self).__init__()
        self.transform = BertPredictionHeadTransform(hidden_size, hidden_act, layer_norm_eps)
#         self.decoder = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
#         hidden_states = self.decoder(hidden_states)
        return hidden_states

############################################ Define Net Class
class QuestNet(nn.Module):
    def __init__(self, model_type="bert-base-uncased", n_classes=30, hidden_layers=[-2, -3, -4]):
        super(QuestNet, self).__init__()
        self.model_name = 'QuestModel'
        self.bert_model = BertModel.from_pretrained(model_type, hidden_dropout_prob=0.1, \
                                                    output_hidden_states=True, force_download=True)   
        self.hidden_layers = hidden_layers
        
        if model_type == "bert-base-uncased":
            self.hidden_size = 768 * (len(hidden_layers) + 1)
        elif model_type == "bert-large-uncased":
            self.hidden_size = 1024 * (len(hidden_layers) + 1)
        elif model_type == "bert-base-cased":
            self.hidden_size = 768 * (len(hidden_layers) + 1)
        else:
            raise NotImplementedError
        
#         self.bert_classification_head = BertClassificationHead(self.hidden_size)
        
        self.fc = nn.Linear(self.hidden_size, n_classes)
        self.selu = nn.SELU()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])

    def forward(self, ids, seg_ids):
        attention_mask = (ids > 0)
        outputs = self.bert_model(input_ids=ids, token_type_ids=seg_ids, attention_mask=attention_mask)
        
        # pooled_out = outputs[1] #  N * 768
        
        sequence_out = torch.unsqueeze(outputs[0][:, 0], dim=-1) # N * 512 * 768 * 1, hidden_states[-1]
        
        # 13 (embedding + 12 transformers) for base
        # 26 (embedding + 25 transformers) for large
        hidden_states = outputs[2]
        fuse_hidden = sequence_out
        
        for hidden_layer in self.hidden_layers:
            h = torch.unsqueeze(hidden_states[hidden_layer][:, 0], dim=-1) # N * 768 * 1
            fuse_hidden = torch.cat([fuse_hidden, h], dim=-1)
            
#         fuse_hidden = fuse_hidden.reshape(fuse_hidden.shape[0], -1)
        
#         out = self.bert_classification_head(fuse_hidden)
        out = fuse_hidden.reshape(fuse_hidden.shape[0], -1)

        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                logit = self.fc(dropout(out))
            else:
                logit += self.fc(dropout(out))
                
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
