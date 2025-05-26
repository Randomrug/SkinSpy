import numpy as np
import tensorflow as tf
from PIL import Image
import torch
from torchvision import transforms
import torch.serialization

# ---- Load Keras Skin Lesion Model ----
best_model = tf.keras.models.load_model('skin_classification.h5', compile=False)
IMG_SIZE = best_model.input_shape[1:3]
class_labels = ['benign', 'malignant', 'insect_bite', 'no_bites']

# ---- Load PyTorch Insect Bite Model ----
try:
    with torch.serialization.safe_globals([transforms.Compose]):
        checkpoint = torch.load(
            'insect_modelformal.pt',
            map_location='cuda' if torch.cuda.is_available() else 'cpu',
            weights_only=False
        )
except Exception as e:
    raise ValueError(f"Model loading failed: {str(e)}") from None

required_keys = {'state_dict', 'classes', 'transform'}
if not all(key in checkpoint for key in required_keys):
    missing = required_keys - set(checkpoint.keys())
    raise KeyError(f"Checkpoint missing required keys: {missing}")

# ---- Define CNN Architecture ----
class CNN(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(64 * 32 * 32, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
insect_model = CNN(num_classes=len(checkpoint['classes'])).to(device)
insect_model.load_state_dict(checkpoint['state_dict'])
insect_model.eval()

def predict_image(image_path):
    try:
        # ---- Step 1: Predict with Keras model ----
        img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
        img_array = tf.keras.utils.img_to_array(img) / 255.0
        img_array = tf.expand_dims(img_array, 0)

        pred = best_model.predict(img_array, verbose=0)[0]
        keras_probs = {class_labels[i]: float(pred[i]) * 100 for i in range(len(class_labels))}
        class_idx = np.argmax(pred)
        confidence = np.max(pred)
        predicted_class = class_labels[class_idx]

        result = {
            'initial_prediction': predicted_class,
            'initial_confidence': float(confidence) * 100,
            'initial_probs': keras_probs,
            'image_path': image_path,
            'refined_prediction': None,
            'refined_probs': None
        }

        # ---- Step 2: If not 'insect_bite', return directly ----
        if predicted_class != "insect_bite":
            return result

        # ---- Step 3: If 'insect_bite', refine with PyTorch model ----
        img_pil = Image.open(image_path).convert('RGB')
        img_tensor = checkpoint['transform'](img_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = insect_model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            confidence_insect, pred_insect = torch.max(probs, 0)
            insect_class = checkpoint['classes'][pred_insect.item()]

        refined_probs = {checkpoint['classes'][i]: float(probs[i]) * 100 for i in range(len(checkpoint['classes']))}
        
        result.update({
            'refined_prediction': insect_class,
            'refined_confidence': float(confidence_insect.item()) * 100,
            'refined_probs': refined_probs
        })

        return result

    except Exception as e:
        return {'error': str(e)}