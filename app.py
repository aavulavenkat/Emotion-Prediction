from flask import Flask, request, render_template
from emotion_model import predict_emotion

app = Flask(__name__)

# Map emotions to emojis
EMOTION_EMOJIS = {
    "joy": "😄",
    "sadness": "😭",
    "anger": "😡",
    "fear": "😱",
    "surprise": "😲",
    "love": "❤️"
}
'''
@app.route("/", methods=["GET", "POST"])
def home():
    text = ""
    label = ""
    emoji = ""
    confidence = 0.0

    if request.method == "POST":
        text = request.form["text"]
        label, confidence = predict_emotion(text)
        emoji = EMOTION_EMOJIS.get(label.lower(), "💭")

    return render_template(
        "index.html",
        text=text,
        label=label,
        confidence=f"{confidence:.2f}",
        label_emoji=emoji
    )

'''
@app.route("/", methods=["GET", "POST"])
def home():
    text = ""
    label = ""
    emoji = ""
    confidence = 0.0

    if request.method == "POST":
        text = request.form["text"]
        label, confidence = predict_emotion(text)
        emoji = EMOTION_EMOJIS.get(label.lower(), "💭")

    return render_template(
        "index.html",
        text=text if request.method == "POST" else "",
        label=label,
        confidence=f"{confidence:.2f}",
        label_emoji=emoji
    )

if __name__ == "__main__":
    app.run(debug=True)
