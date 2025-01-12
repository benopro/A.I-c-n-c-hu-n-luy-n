from flask import Flask, request, render_template
from transformers import BartForConditionalGeneration, BartTokenizer
import redis
import json

# Kết nối Redis
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)

# Tải mô hình BART
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Lưu hội thoại vào Redis
def save_conversation(prompt, response, feedback):
    data = {"prompt": prompt, "response": response, "feedback": feedback}
    redis_client.rpush("conversation_history", json.dumps(data))

# Lấy lịch sử hội thoại từ Redis
def get_conversations():
    conversations = redis_client.lrange("conversation_history", 0, -1)
    return [json.loads(convo) for convo in conversations]

# Xóa lịch sử hội thoại
def clear_conversations():
    redis_client.delete("conversation_history")

# Sinh phản hồi từ AI
def generate_response(prompt, feedback=None):
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
    outputs = model.generate(inputs.input_ids, max_length=150, num_beams=4, early_stopping=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Lưu vào lịch sử
    save_conversation(prompt, response, feedback)
    return response

# Flask App
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def chat():
    response = None
    feedback = None

    if request.method == "POST":
        question = request.form.get("question")
        feedback = request.form.get("feedback")

        if question:
            response = generate_response(question, feedback)
        elif feedback:
            print(f"Phản hồi người dùng: {feedback}")

    history = get_conversations()
    return render_template("chat_with_feedback.html", history=history, response=response)

if __name__ == "__main__":
    app.run(debug=True)
