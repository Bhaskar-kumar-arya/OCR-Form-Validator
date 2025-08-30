from transformers import TrOCRProcessor

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
tokenizer = processor.tokenizer

print("@", tokenizer.convert_tokens_to_ids("@"))
print("+", tokenizer.convert_tokens_to_ids("+"))