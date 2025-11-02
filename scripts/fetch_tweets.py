import tensorflow as tf

# Load your old model
model = tf.keras.models.load_model(
    r"C:\Users\shree\OneDrive\Desktop\Original_twitter\models\cnn_sentiment_model.h5"
)

# Save it in the new format
model.save(
    r"C:\Users\shree\OneDrive\Desktop\Original_twitter\models\cnn_sentiment_model.keras"
)

print("✅ Model converted to .keras format successfully!")
