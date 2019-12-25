from transformers import *
import torch
import torch.nn as nn
import torch.nn.functional as F

############################################ Define Net Class
class QuestNet(nn.Module):
    def __init__(self, model_type="bert-base-uncased", n_classes=30):
        super(QuestNet, self).__init__()
        self.model_name = 'QuestModel'
        self.bert_model = BertModel.from_pretrained(model_type)   
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

    def forward(self, ids, seg_ids):
        attention_mask = (ids > 0)
        sequence_out, pool_out = self.bert_model(input_ids=ids, token_type_ids=seg_ids, attention_mask=attention_mask)
        # sequence_out, N * 512 * 768
        # pooled_out = N * 768

        # use pooled_out
        # out = F.dropout(pooled_out, p=0.2, training=self.training)

        # use sequence_out + global_average_pooling
        out = torch.squeeze(torch.mean(sequence_out, dim=1))

        # use sequence_out + global_average_pooling cat sequence_out + global_max_pooling
        # out_mean = torch.squeeze(torch.mean(sequence_out, dim=1))
        # out_max, _ = torch.max(sequence_out, dim=1)
        # out = torch.cat([out_mean, out_max], dim=1)
        # out N * 768 * 2

        out = F.dropout(out, p=0.2, training=self.training)
        logit = self.fc(out)

        # out = self.relu(self.fc_1(out))
        # out = F.dropout(out, p=0.2, training=self.training)
        # logit = self.fc_2(out)

        return logit

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