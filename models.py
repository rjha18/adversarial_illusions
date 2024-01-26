import torch
import torch.nn as nn

from imagebind.models import imagebind_model
import imagebind.data as data

# from AudioCLIP2 import AudioCLIP
import open_clip
from open_clip import tokenizer

import boto3
import json
from botocore.config import Config
from torchvision.utils import save_image
import os
from utils import unnorm
import base64
from PIL import Image
from io import BytesIO

def load_model(model_flag, device):
    if model_flag == "titan":
        model = TitanWrapper(None,device=device)
        return model
    if model_flag == "imagebind":
        model = ImageBindWrapper(imagebind_model.imagebind_huge(pretrained=True), device=device)
    elif model_flag == 'audioclip':
        model = AudioCLIPWrapper(AudioCLIP(pretrained=f'bpe/AudioCLIP-Full-Training.pt'))
    elif model_flag == 'audioclip_partial':
        model = AudioCLIPWrapper(AudioCLIP(pretrained=f'bpe/AudioCLIP-Partial-Training.pt'))
    elif model_flag == 'openclip':
        m, _, p = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k', cache_dir='bpe/')
        model = OpenCLIPWrapper(m, p)
    elif model_flag == 'openclip_rn50':
        m, _, p = open_clip.create_model_and_transforms('RN50', pretrained='openai', cache_dir='bpe/')
        model = OpenCLIPWrapper(m, p)
    else:
        raise NotImplementedError()

    model.to(device)
    model.eval()
    return model

class TitanWrapper(nn.Module):
    def __init__(self, model, device):
        super(TitanWrapper, self).__init__()
        self.flag = 'titan'
    def forward(self, X, modality, normalize=True):
        if modality == 'text': 
            body = json.dumps(
                {
                    "inputText": X[0],
                } 
            )
        if modality == 'vision':
            save_image(torch.squeeze(unnorm(X)), 'dummy.png')
            with open('dummy.png', "rb") as image_file:
                input_image = base64.b64encode(image_file.read()).decode('utf8')
            body = json.dumps(
                {
                    "inputImage": input_image
                } 
            )
            # os.remove('dummy.png')

        self.aws_profile = 'default'
        boto3.setup_default_session(profile_name=self.aws_profile)
        bedrock_runtime = boto3.client(
            service_name="bedrock-runtime",
            region_name="us-east-1"
        )
        response = bedrock_runtime.invoke_model(
        	body=body, 
        	modelId="amazon.titan-embed-image-v1", 
        	accept="application/json", 
        	contentType="application/json"
        )
        
        response_body = json.loads(response.get("body").read())['embedding']
        ret=torch.tensor(response_body).unsqueeze(0)
        return ret

class ImageBindWrapper(nn.Module):
    def __init__(self, model, device):
        super(ImageBindWrapper, self).__init__()
        self.model = model
        self.device = device
        self.flag = 'imagebind'
    
    def forward(self, X, modality, normalize=True):
        if modality == 'text':
            if isinstance(X, str):
                X = [X]
            X = data.load_and_transform_text(X, self.device)
            X = X.to(next(self.model.parameters()).device)   
        # ret=self.model.forward({modality: X}, normalize=normalize)[modality]
        # print(ret.shape)
        # print(ret[0].shape)
        # input()
        return self.model.forward({modality: X}, normalize=normalize)[modality]


class AudioCLIPWrapper(nn.Module):
    def __init__(self, model):
        super(AudioCLIPWrapper, self).__init__()
        self.model = model
        self.flag = 'audioclip'
    
    def forward(self, X, modality, normalize=True):
        if modality == 'vision':
            modality = 'image'
        if modality == 'text':
            if isinstance(X, str):
                X = [X]
            X = [[i] for i in X]

        features = self.model.forward(**{modality: X}, normalize=normalize)[0][0]
        if modality == 'audio':
            return features[0]
        elif modality == 'image':
            return features[1]
        elif modality == 'text':
            return features[2]
        else:
            raise NotImplementedError()
        

class OpenCLIPWrapper(nn.Module):
    def __init__(self, model, preprocess):
        super(OpenCLIPWrapper, self).__init__()
        self.model = model
        self.preprocess = preprocess
        self.flag = 'openclip'
    
    def forward(self, X, modality, normalize=True):
        if modality == 'vision':
            modality = 'image'
        elif modality == 'text':
            X = tokenizer.tokenize(X)

        features = self.model.forward(**{modality: X})
        if modality == 'image':
            return features[0]
        elif modality == 'text':
            return features[1]
        else:
            raise NotImplementedError()
