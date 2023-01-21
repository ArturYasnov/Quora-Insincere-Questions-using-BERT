import torch
from transformers import BertTokenizer

from src.training_class import BertClassifier

if __name__ == "__main__":
    model = BertClassifier()
    checkpoint = torch.load("/other/bert_3epochs.pt")
    model.load_state_dict(checkpoint)

    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    text = "positive message."
    text = tokenizer(
        text, padding="max_length", max_length=64, truncation=True, return_tensors="pt"
    )

    mask = text["attention_mask"]
    input_id = text["input_ids"].squeeze(1)
    output = model(input_id, mask)
    predict = output.argmax(dim=1).item()

    if predict == 0:
        print("Sincere.")
    else:
        print("Insincere question.")
    print(output)
    print(output.argmax(dim=1).item())
