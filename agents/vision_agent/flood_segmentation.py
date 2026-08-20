import torch
import cv2
import segmentation_models_pytorch as smp
import numpy as np

# load model once
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1
)

model.load_state_dict(torch.load("agents/vision_agent/unet_flood_modelN.pth", map_location="cpu"))
model.eval() #places in evaluation mode then batchnorm


def detect_flood(image):

    # resize for model
    img = cv2.resize(image, (256,256))
    img = img / 255.0

    tensor = torch.tensor(img).permute(2,0,1).unsqueeze(0).float()

    with torch.no_grad(): #no gradients
        pred = model(tensor)

    prob_map = torch.sigmoid(pred).squeeze().numpy()

    # resize back to original image size
    prob_map = cv2.resize(prob_map, (image.shape[1], image.shape[0]))

    return prob_map