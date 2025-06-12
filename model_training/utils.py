# model_training/utils.py

label_list = [
    "O",        # Outside a named entity
    "B-MISC",   # Beginning of a miscellaneous entity
    "I-MISC",   # Inside a miscellaneous entity
    "B-PER",    # Beginning of a person name
    "I-PER",    # Inside a person name
    "B-ORG",    # Beginning of an organization
    "I-ORG",    # Inside an organization
    "B-LOC",    # Beginning of a location
    "I-LOC"     # Inside a location
]

label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}

def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=128
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx])
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs
