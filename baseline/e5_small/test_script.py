# Use a pipeline as a high-level helper
import os

from src.evaluate_helpers import get_predicted_labels
from src.model_helpers import inference_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

pipe = pipeline(
    "text-classification", model="MayZhou/e5-small-lora-ai-generated-detector"
)
# Load model directly


tokenizer = AutoTokenizer.from_pretrained("MayZhou/e5-small-lora-ai-generated-detector")
our_model = AutoModelForSequenceClassification.from_pretrained(
    "MayZhou/e5-small-lora-ai-generated-detector"
)

# Generate a sentence by Claude3.5
example = [
    "The rain cascades endlessly from Vancouver's steel-grey winter skies, transforming the city streets into glistening mirrors that reflect the moody silhouettes of snow-dusted mountains."
]

# Get predicted probability of machine-generated texts
prediction = inference_model(our_model, example, nthreads=1)
print(prediction.predictions["Predicted_Probs(1)"])
# Get predicted label given a threshold
print(get_predicted_labels(prediction, 0.85))
