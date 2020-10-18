"""
PyTorch implementation of:
Learning Deep Features for Discriminative Localization

"""
import argparse
import copy
import os

import cv2
import numpy as np
import torchvision
import torch
from PIL import Image
from torchvision.models.resnet import resnet152, resnet18, resnet50

import ImageNetLabels

model_name_to_func = {
    "resnet18": torchvision.models.resnet18,
    "resnet34": torchvision.models.resnet34,
    "resnet50": torchvision.models.resnet50,
    "resnet101": torchvision.models.resnet101,
    "resnet152": torchvision.models.resnet152,}


def parse_args():
    parser = argparse.ArgumentParser(
        "Class activation maps in pytorch")
    parser.add_argument('--model_name', type=str,
                        help='name of model to use', required=True)
    parser.add_argument('--input_image', type=str,
                        help='path to input image', required=True)
    parser.add_argument('--save_gif', default=False,
                        help='save a gif animation', required=False, action='store_true')
    args = parser.parse_args()

    assert args.model_name in list(model_name_to_func.keys()), 'Model [%s] not found in supported models in [%s]' % (
        args.model_name, list(model_name_to_func.keys()), )
    return args


class ReshapeModule(torch.nn.Module):
    def __init__(self):
        super(ReshapeModule, self).__init__()

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b*c, h*w).permute(1, 0)
        return x


def modify_model_cam(model):
    """Modifies a pytorch model object to remove last 
    global average pool and replaces with a custom reshape
    node that enables generating class activation maps as 
    forward pass

    Args:
        model: pytorch model graph

    Raises:
        ValueError: if no global average pool layer is found

    Returns:
        model: modified model with last global average pooling 
                replaced with custom reshape module
    """

    # fetch all layers + globalavgpoollayers
    alllayers = [n for n, m in model.named_modules()]
    globalavgpoollayers = [n for n, m in model.named_modules(
    ) if isinstance(m, torch.nn.AdaptiveAvgPool2d)]
    if globalavgpoollayers == []:
        raise ValueError('Model does not have a Global Average Pool layer')

    # check if last globalavgpool is second last layer - otherwise the method wont work
    assert alllayers.index(globalavgpoollayers[-1]) == len(
        alllayers)-2, 'Global Average Pool is not second last layer'

    # remove last globalavgpool with our custom reshape module
    model._modules[globalavgpoollayers[-1]] = ReshapeModule()

    return model


def infer_with_cam_model(cam_model, image):
    """Run forward pass with image tensor and get class activation maps
    as well as predicted class index

    Args:
        cam_model: pytorch model graph with custom reshape module modified using modify_model_cam()
        image: torch.tensor image with preprocessing applied

    Returns:
        class activation maps and most probable class index 
    """
    with torch.no_grad():
        output_cam_acti = cam_model(image)

        _, output_cam_idx = torch.topk(torch.mean(
            output_cam_acti, dim=0), k=10, dim=-1)

    return output_cam_acti, output_cam_idx


def postprocess_cam(cam_image, image):
    """Process class activation map to generate a heatmap
    overlay the heatmap on original image

    Args:
        cam_model: pytorch model graph with custom reshape module modified using modify_model_cam()
        image: numpy array for image to overlay heatmap on top

    Returns:
        numpy array with image + overlayed heatmap
    """
    h, w = image.shape[0:2]

    sp = int(np.sqrt(cam_image.shape[0]))

    assert cam_image.shape[0] == sp * \
        sp, 'Only activation maps that are square are supported at the moment'

    # make square class act map (if possible)
    cam_image = np.reshape(cam_image, [sp, sp])

    # normalise to be in range [0, 255]
    cam_image = cam_image - np.min(cam_image)
    cam_image = (cam_image/np.max(cam_image) * 255).astype(np.uint8)

    # resize to input image shape and make a heatmap
    cam_image_processed = cv2.applyColorMap(
        cv2.resize(cam_image, (w, h)), cv2.COLORMAP_JET)

    # BGR to RGB (opencv is BGR image, PIL output is RGB)
    cam_image_processed = cam_image_processed[:, :, ::-1]

    return cam_image_processed


if __name__ == '__main__':
    args = parse_args()

    # preprocessing for imagenet models
    preprocess_imagenet = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # run on cuda if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # fetch the model and apply cam modification
    orig_model = model_name_to_func[args.model_name](pretrained=True)
    orig_model.eval()
    class_act_map_model = modify_model_cam(copy.deepcopy(orig_model))
    class_act_map_model.to(device)

    # load image, preproceess
    filename = args.input_image
    input_image_pil = Image.open(filename)
    input_image = preprocess_imagenet(input_image_pil).unsqueeze(0)
    input_image = input_image.to(device)
    
    input_image_raw_np = np.asarray(input_image_pil)

    # run inference with input image
    selidx=0
    output_cam_acti, output_cam_idx = infer_with_cam_model(
        class_act_map_model, input_image)
    print('Prediction [%s] at index [%d]' % (
        ImageNetLabels.idx_to_class[output_cam_idx[selidx].item()], output_cam_idx[selidx]))

    cam_image_raw = output_cam_acti[:, output_cam_idx[selidx].item()].cpu().detach().numpy()
    
    cam_image_processed = postprocess_cam(
        cam_image_raw, input_image_raw_np)

    # overlay on top of original image
    alpha = 0.5
    cam_image_overlayed = (1-alpha) * input_image_raw_np + alpha * cam_image_processed

    # save
    Image.fromarray(cam_image_overlayed.astype(np.uint8)).save(
        os.path.join('results', os.path.basename(args.input_image)))
    
    # create gif animation if required
    if args.save_gif:
        cam_image_overlayed_gif = []
        for al in [x/100. for x in range(50)]:
            cam_image_overlayed_gif.append((1-al) * input_image_raw_np + al * cam_image_processed)

        for al in reversed([x/100. for x in range(50)]):
            cam_image_overlayed_gif.append((1-al) * input_image_raw_np + al * cam_image_processed)

        factor = min([300./x for x in cam_image_overlayed_gif[0].shape[0:2]])
        cam_image_overlayed_gif = [Image.fromarray(x.astype(np.uint8)).resize([int(factor * s) for s in reversed(x.shape[0:2])])
                        for x in cam_image_overlayed_gif]
        cam_image_overlayed_gif[0].save(os.path.join('results', os.path.basename(args.input_image).split('.')[
                            0] + '.gif'), save_all=True, append_images=cam_image_overlayed_gif[1:], optimize=True, duration=40, loop=0)
