import torch.nn

class BertCL(nn.Module):
    def __init__(self):
        super(BertCL, self).__init__()

        self.checkpoint = 'dmis-lab/biobert-v1.1'

        # Bert per andare da token a embedding
        self.bert_config = AutoConfig.from_pretrained(self.checkpoint)
        self.bert = AutoModel.from_pretrained(self.checkpoint)

        for param in self.bert.parameters():
            param.requires_grad = False

        # NN per effettuare il contrastive learning
        self.l1 = nn.Linear(self.bert_config.hidden_size, self.bert_config.hidden_size)

        self.drop = nn.Dropout(p=0.4)
        self.activation = nn.Tanh()

    def forward(self, tokenized_sentences):
        sentence_embeddings = self.bert(**tokenized_sentences)
        result = sentence_embeddings.last_hidden_state[:, 0, :]
        result = self.l1(result)
        return self.activation(result)
