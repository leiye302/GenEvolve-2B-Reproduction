from .common import jsonl
from .data import sft_registry


def verify_labels(base, folder, config):
    from llamafactory.data.converter import SharegptDatasetConverter
    from llamafactory.data.parser import DatasetAttr
    from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
    from llamafactory.data.template import get_template_and_fix_tokenizer
    from llamafactory.hparams import DataArguments, ModelArguments
    from llamafactory.model.loader import load_tokenizer

    source = next(row for row in jsonl(folder / "train.jsonl") if row["images"])
    data_args = DataArguments(
        template=config["template"], cutoff_len=config["cutoff_len"],
        train_on_prompt=False, mask_history=False, media_dir=str(folder),
    )
    attr = DatasetAttr("file", "train.jsonl")
    attr.join(sft_registry()["genevolve_train"])
    converted = SharegptDatasetConverter(attr, data_args)(source)
    loaded = load_tokenizer(ModelArguments(
        model_name_or_path=str(base), trust_remote_code=False,
        image_min_pixels=config["image_min_pixels"], image_max_pixels=config["image_max_pixels"],
    ))
    tokenizer, processor = loaded["tokenizer"], loaded["processor"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    encoded = SupervisedDatasetProcessor(template, tokenizer, processor, data_args).preprocess_dataset(
        {name: [value] for name, value in converted.items()}
    )
    (ids,), (labels,) = encoded["input_ids"], encoded["labels"]
    targets = [label for label in labels if label != -100]
    text = tokenizer.decode(targets, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    for message in source["messages"]:
        content = message["content"].strip()
        if content and ((content in text) != (message["role"] == "assistant")):
            raise RuntimeError("The pinned trainer's SFT labels did not match the assistant turns.")
    image_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    if image_id not in ids or image_id in targets:
        raise RuntimeError("The SFT visual tokens are missing or present in the targets.")
    return {"sequence_tokens": len(ids), "assistant_tokens": len(targets), "image_tokens": ids.count(image_id)}
