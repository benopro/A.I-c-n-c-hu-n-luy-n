from transformers import BartForConditionalGeneration, BartTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
import json

# Tải dữ liệu từ tệp JSON
def load_training_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Dataset.from_list(data)

# Tiền xử lý dữ liệu
def preprocess_data(dataset, tokenizer):
    def preprocess_function(examples):
        inputs = examples["prompt"]
        targets = examples["response"]

        # Tokenizer với padding và truncation
        model_inputs = tokenizer(inputs, max_length=1024, padding="max_length", truncation=True)
        labels = tokenizer(targets, max_length=512, padding="max_length", truncation=True).input_ids

        # Chuyển các giá trị pad_token_id trong labels thành -100 để không tính vào loss
        model_inputs["labels"] = [
            [(label if label != tokenizer.pad_token_id else -100) for label in sequence]
            for sequence in labels
        ]

        return model_inputs

    return dataset.map(preprocess_function, batched=True)

# Huấn luyện mô hình
def fine_tune_model(dataset, model, tokenizer):
    # Cấu hình huấn luyện
    training_args = TrainingArguments(
        output_dir="./finetuned_model",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=500,
        evaluation_strategy="no",  # Tắt đánh giá
        logging_dir="./logs",
        learning_rate=2e-5,
        save_total_limit=2
    )

    # Sử dụng DataCollator để padding động
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Tạo Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator
    )

    # Huấn luyện
    trainer.train()

    # Lưu mô hình
    model.save_pretrained("./finetuned_model")
    tokenizer.save_pretrained("./finetuned_model")

if __name__ == "__main__":
    # Tải mô hình và tokenizer
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    # Tải dữ liệu và tiền xử lý
    data_file = "training_data.json"  # Đường dẫn tới tệp dữ liệu
    dataset = load_training_data(data_file)
    processed_dataset = preprocess_data(dataset, tokenizer)

    # Huấn luyện mô hình
    fine_tune_model(processed_dataset, model, tokenizer)
