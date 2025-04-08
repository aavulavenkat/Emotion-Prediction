'''from emotion_model import train_model, predict_emotion

# Train the model
train_model()

# Predict example
text = "I am feeling really happy today!"
label, confidence = predict_emotion(text)
print(f"Predicted Emotion: {label} ({confidence:.2f})")'''

from emotion_model import train_model, predict_emotion

# Train the model
train_model()

# Predict example
text = "I am feeling really happy today!"

# Debug: see what the function returns
result = predict_emotion(text)
print("DEBUG result:", result)
