import torch
from fastapi import FastAPI, UploadFile, File
import rasterio
from rasterio.io import MemoryFile
from PIL import Image
import numpy as np
import io
from src.unet_sentinel_resnet import unet,UnetResnetSentinel2
from fastapi.responses import StreamingResponse

# Initialize FastAPI app
app = FastAPI()

model = unet.load_from_checkpoint(
    checkpoint_path=r'model\arecanut-unetresnetsent2-128-epoch=333-val_JaccardIndex=0.46474990248680115.ckpt',
    encoder = UnetResnetSentinel2,
    nc=3,
    c=15,
    loss='jaccard'
        )

def inference(image: np.ndarray,model) -> np.ndarray:
    image_tensor = torch.from_numpy(image).float().unsqueeze(0)
    with torch.no_grad():
        model.eval()
        logits = model(image_tensor.to('cuda'))  # Get raw logits from the model

        # Apply softmax to get class probabilities
        # Shape: [batch_size, num_classes, H, W]

        pr_masks = logits.softmax(dim=1)
        # Convert class probabilities to predicted class labels
        pr_masks = pr_masks.argmax(dim=1).squeeze().cpu().numpy()  # Shape: [H, W]
        print(pr_masks)
    return pr_masks

# Define the FastAPI endpoint for prediction
@app.post("/predict/")
async def predict_endpoint(file: UploadFile = File(...)):
    # Read the uploaded image using rasterio
    with MemoryFile(await file.read()) as memfile:
        with memfile.open() as dataset:
            image = dataset.read()  # Read as numpy array
            mask = inference(image,model)  # Get the predicted mask

            # Save the mask to an in-memory file
            profile = dataset.profile  # Copy profile for metadata
            profile.update(dtype=rasterio.float32, count=1)  # Update profile for mask output
            with MemoryFile() as memfile_output:
                with memfile_output.open(**profile) as dst:
                    dst.write(mask, 1)  # Write mask to first band
                output_bytes = memfile_output.read()  # Save to bytes

    return StreamingResponse(io.BytesIO(output_bytes), media_type="image/tiff")
# To run the server, save this file and run it using:
# uvicorn <filename>:app --reloade
