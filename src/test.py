import onnxruntime as ort
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
import os

# --- Configuration ---
MODEL_REPO = "qualcomm/LiteHRNet"
MODEL_FILENAME = "LiteHRNet.onnx"
# Image URL from the Hugging Face example (or use a local path)
# IMAGE_URL = "https://huggingface.co/qualcomm/RTMPose_Body2d/resolve/main/pose_sample.jpg"
IMAGE_PATH = "example2.png" # Assuming you download it first or provide your own path

# Model Input Size (check model specifics if different)
INPUT_WIDTH = 192
INPUT_HEIGHT = 256

# Preprocessing constants (common for ImageNet-trained models)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# --- Helper Functions ---

def download_image(url, save_path):
    """Downloads an image from a URL."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Image downloaded successfully to {save_path}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image: {e}")
        return False
    except IOError as e:
        print(f"Error saving image: {e}")
        return False

def preprocess_image(image_path, input_width, input_height):
    """Loads, resizes, and preprocesses an image."""
    try:
        img = Image.open(image_path).convert("RGB")
        original_width, original_height = img.size

        # Resize
        img_resized = img.resize((input_width, input_height), Image.BILINEAR) # Common practice uses BILINEAR

        # Convert to numpy array and normalize to 0-1
        img_np = np.array(img_resized, dtype=np.float32) / 255.0

        # Apply normalization (mean subtraction, std division)
        img_normalized = (img_np - MEAN) / STD

        # Transpose from (H, W, C) to (C, H, W)
        img_transposed = img_normalized.transpose(2, 0, 1)

        # Add batch dimension -> (1, C, H, W)
        input_tensor = np.expand_dims(img_transposed, axis=0)

        return input_tensor, original_width, original_height, img_resized

    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None, None, None, None
    except Exception as e:
        print(f"Error during image preprocessing: {e}")
        return None, None, None, None

def visualize_pose(image, keypoints, scores, threshold=0.3):
    """Draws keypoints and connections on the image."""
    plt.figure(figsize=(8, 10))
    plt.imshow(image)
    ax = plt.gca()

    # Define connections between keypoints (RTMPose 17 keypoints typical order)
    # 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
    # 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
    # 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
    # 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
    connections = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
        (11, 12), (5, 11), (6, 12), # Torso
        (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
    ]

    valid_kpts = {}
    for i, (x, y) in enumerate(keypoints):
        score = scores[i]
        if score > threshold:
            valid_kpts[i] = (x,y)
            plt.scatter(x, y, s=50, c='red', marker='o')
            # plt.text(x, y, f"{i}:{score:.2f}", fontsize=8, color='white') # Optional: label points

    for i, j in connections:
         if i in valid_kpts and j in valid_kpts:
            x1, y1 = valid_kpts[i]
            x2, y2 = valid_kpts[j]
            ax.plot([x1, x2], [y1, y2], linewidth=2, color='cyan') # Use a visible color

    plt.axis('off')
    plt.title("Detected Pose")
    plt.show()


# 1. Load the ONNX model file from local directory
# Define the path to your local clone of the Hugging Face repo
# if there were not CL args, use hardcoded path
path_to_lfs_model = r"/Users/jeremy/Git/ProjectKeypointInference/LiteHRNet"
import sys
# check the number of command line arguments. if there aren't any, proceed with setting LOCAL_MODEL_DIR
if len(sys.argv) == 1:
    LOCAL_MODEL_DIR = path_to_lfs_model
else:
    LOCAL_MODEL_DIR = sys.argv[1]


print(f"Looking for local ONNX model in {LOCAL_MODEL_DIR}...")
model_path = os.path.join(LOCAL_MODEL_DIR, MODEL_FILENAME)

if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    print("Please ensure the path is correct and you have run 'git lfs pull' in the directory.")
    exit()
else:
     print(f"Using local model: {model_path}")

# 2. Download the sample image (if needed)
if not os.path.exists(IMAGE_PATH):
     image_url_on_hf = "https://huggingface.co/qualcomm/RTMPose_Body2d/resolve/main/pose_sample.jpg"
     print(f"Downloading sample image {IMAGE_PATH}...")
     if not download_image(image_url_on_hf, IMAGE_PATH):
         exit()
else:
    print(f"Using existing image: {IMAGE_PATH}")


# 3. Preprocess the image
print("Preprocessing image...")
input_tensor, orig_w, orig_h, resized_img = preprocess_image(IMAGE_PATH, INPUT_WIDTH, INPUT_HEIGHT)

if input_tensor is None:
    exit()

print(f"Input tensor shape: {input_tensor.shape}") # Should be (1, 3, 256, 192) or similar

# 4. Load the ONNX model and run inference
print("Loading ONNX session...")
try:
    # Specify execution providers: CoreML for Apple Silicon (MPS), CPU as fallback
    providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
    # providers = ['CPUExecutionProvider'] # Force CPU if needed for debugging

    session = ort.InferenceSession(model_path, providers=providers)
    print(f"ONNX Runtime session created using provider: {session.get_providers()}")

    # Get input/output names
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()] # Model might have multiple outputs
    # inspect the model outputs and display. 
    outputs = session.run(output_names, {input_name: input_tensor})
    print(f"Inference complete. Received {len(outputs)} output tensor(s).")
    print("Inspecting output shapes:")
    for i, output_tensor in enumerate(outputs):
        print(f"  Output {i} ({output_names[i]}): shape={output_tensor.shape}, dtype={output_tensor.dtype}")

    print(f"Input name: {input_name}")
    print(f"Output name(s): {output_names}")

    # Run inference
    print("Running inference...")
    outputs = session.run(output_names, {input_name: input_tensor})
    print("Inference complete.")

    # Usually, the main output is the first one
    keypoints_output = outputs[0] # Shape is likely (1, num_keypoints, 3) -> (batch, keypoint_idx, [x, y, score])

    # Postprocess (extract keypoints and scores)
    # Squeeze the batch dimension
    if keypoints_output.ndim == 3:
        keypoints_with_scores = keypoints_output[0] # Shape (num_keypoints, 3)
        
        # keypoints are fine, but aren't normalized to the original image size
        # keypoints_with_scores = keypoints_with_scores * np.array([orig_w, orig_h, 1]) # Scale to original size
        keypoints = keypoints_with_scores[:, :2]  # Shape (num_keypoints, 2) -> (x, y)
        scores = keypoints_with_scores[:, 2]     # Shape (num_keypoints,)
    elif keypoints_output.ndim == 2:
        keypoints = keypoints_output # Shape (num_keypoints, 2) -> (x, y)
        scores = np.ones(keypoints.shape[0])
    else:
        raise ValueError("Unexpected output shape from the model.")

    print(f"\nDetected Keypoints (relative to {INPUT_WIDTH}x{INPUT_HEIGHT} input):")
    for i, ((x, y), score) in enumerate(zip(keypoints, scores)):
        print(f"  Keypoint {i}: (x={x:.2f}, y={y:.2f}), Score={score:.4f}")

    # 5. Visualize (optional)
    print("\nVisualizing results...")
    # Note: The keypoints are relative to the *resized* input image (192x256).
    # We visualize on the resized image. If you need to draw on the *original*
    # image, you'd need to scale the keypoints:
    # x_orig = x * (orig_w / INPUT_WIDTH)
    # y_orig = y * (orig_h / INPUT_HEIGHT)

    visualize_pose(resized_img, keypoints, scores, threshold=0.3)

except ort.OrtError as e:
    print(f"ONNX Runtime error: {e}")
    print("Ensure 'onnxruntime' is installed correctly.")
    print("If using CoreMLExecutionProvider, make sure you are on macOS with Apple Silicon.")
except ImportError as e:
     print(f"Import Error: {e}. Please install missing libraries.")
     print("You might need: pip install onnxruntime numpy Pillow requests matplotlib huggingface_hub")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
