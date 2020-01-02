from transformers import *
import torch
import torch.nn as nn
import torch.nn.functional as F

############################################ Define Net Class
class QuestNet(nn.Module):
    def __init__(self, model_type="bert-base-uncased", n_classes=30):
        super(QuestNet, self).__init__()
        self.model_name = 'QuestModel'
        self.bert_model = BertModel.from_pretrained(model_type, output_hidden_states=True, force_download=True)   
        if model_type == "bert-base-uncased":
            self.fc = nn.Linear(768 * 1, n_classes)
            # self.fc_1 = nn.Linear(768 * 1, 512)
            # self.fc_2 = nn.Linear(512, n_classes)
        elif model_type == "bert-large-uncased":
            self.fc = nn.Linear(1024 * 1, n_classes)
            # self.fc_1 = nn.Linear(1024 * 1, 512)
            # self.fc_2 = nn.Linear(512, n_classes)
        elif model_type == "bert-base-cased":
            self.fc = nn.Linear(768 * 1, n_classes)
            # self.fc_1 = nn.Linear(768 * 1, 512)
            # self.fc_2 = nn.Linear(512, n_classes)
        else:
            raise NotImplementedError
        self.selu = nn.SELU()
        self.relu = nn.ReLU()
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])

    def forward(self, ids, seg_ids):
        attention_mask = (ids > 0)
        outputs = self.bert_model(input_ids=ids, token_type_ids=seg_ids, attention_mask=attention_mask)
        
        # sequence_output = outputs[0], N * 512 * 768
        # pooled_output = outputs[1],  N * 768
        
        # use sequence_out + global_average_pooling
        # out = torch.squeeze(torch.mean(sequence_out, dim=1))

        # use sequence_out + global_average_pooling cat sequence_out + global_max_pooling
        # out_mean = torch.squeeze(torch.mean(sequence_out, dim=1))
        # out_max, _ = torch.max(sequence_out, dim=1)
        # out = torch.cat([out_mean, out_max], dim=1)
        
        # sequence_out = outputs[0]
        # out = torch.squeeze(torch.mean(sequence_out[:, -4:, :], dim=1))
        
        hidden_states = outputs[2]
        # print(len(hidden_states)) 
        # 13 (embedding + 11 transformers + pooler) for base, 
        # 26 (embedding + 24 transformers + pooler) for large, 
        # we choose last 4 heads not including pooler
        h1 = hidden_states[-2][:, 0].reshape((-1, 1, 768))
        h2 = hidden_states[-3][:, 0].reshape((-1, 1, 768))
        h3 = hidden_states[-4][:, 0].reshape((-1, 1, 768))
        h4  = hidden_states[-5][:, 0].reshape((-1, 1, 768))

        all_h = torch.cat([h1, h2, h3, h4], 1)
        out = torch.mean(all_h, 1)
        
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
