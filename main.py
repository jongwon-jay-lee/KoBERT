import torch
from transformers import BertModel, BertConfig

bert_config = {
    'attention_probs_dropout_prob': 0.1,
    'hidden_act': 'gelu',
    'hidden_dropout_prob': 0.1,
    'hidden_size': 768,
    'initializer_range': 0.02,
    'intermediate_size': 3072,
    'max_position_embeddings': 512,
    'num_attention_heads': 12,
    'num_hidden_layers': 12,
    'type_vocab_size': 2,
    'vocab_size': 8002
}

if __name__ == "__main__":
    ctx = "cpu"
    # kobert
    kobert_model_file = "./kobert_resources/pytorch_kobert_2439f391a6.params"
    kobert_vocab_file = "./kobert_resources/kobert_news_wiki_ko_cased-ae5711deb3.spiece"

    bertmodel = BertModel(config=BertConfig.from_dict(bert_config))
    bertmodel.load_state_dict(torch.load(kobert_model_file))
    device = torch.device(ctx)
    bertmodel.to(device)
    # bertmodel.eval()

    # for name, param in bertmodel.named_parameters():
    #     print(name, param.shape)

    for name, param in bertmodel.named_parameters():
        if param.requires_grad:
            print(name, param.shape)
