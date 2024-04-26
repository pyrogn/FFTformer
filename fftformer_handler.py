import io
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
from ts.torch_handler.base_handler import BaseHandler

# Import your model definition
from basicsr.models.archs.fftformer_arch import fftformer


class FFTformerHandler(BaseHandler):
    """
    A custom model handler implementation for the fftformer model.
    """

    def __init__(self):
        super().__init__()
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        self.initialized = False

    def initialize(self, context):
        """
        Invoke by torchserve for loading a model.
        :param context: context contains model server system properties.
        """
        # Load the model
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        serialized_file = self.manifest["model"]["serializedFile"]
        model_pt_path = os.path.join(model_dir, serialized_file)

        # Check if the model file exists
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")

        # Load the model
        self.model = fftformer()
        self.model.load_state_dict(torch.load(model_pt_path))
        self.model.to(self.device)
        self.model.eval()

        self.initialized = True

    def preprocess(self, data):
        """
        Preprocessing involves transforming the input image to a tensor.
        :param data: Input data for prediction.
        :return: Preprocessed input data.
        """
        images = []

        for row in data:
            # Convert the input image to a tensor
            image = row.get("data") or row.get("body")
            image = Image.open(io.BytesIO(image))
            image = self.transform(image)
            images.append(image)

        return torch.stack(images).to(self.device)

    def inference(self, inputs):
        """
        Run model prediction on the preprocessed data.
        :param inputs: Preprocessed input data.
        :return: Model prediction output.
        """
        # Run inference
        with torch.no_grad():
            outputs = self.model(inputs)

        return outputs

    def postprocess(self, inference_output):
        """
        Postprocessing involves converting the model output tensors to image format.
        :param inference_output: Inference output.
        :return: Postprocessed inference output.
        """
        # Convert the output tensor to an image
        images = []
        for tensor in inference_output:
            image = transforms.ToPILImage()(tensor.cpu().squeeze(0))
            images.append(image)

        return images


class ModelHandler(object):
    """
    A custom model handler implementation.
    """

    def __init__(self):
        self._context = None
        self.initialized = False
        self.model = None
        self.device = None

    def initialize(self, context):
        """
        Invoke by torchserve for loading a model
        :param context: context contains model server system properties
        :return:
        """

        #  load the model
        self.manifest = context.manifest

        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available()
            else "cpu"
        )

        # Read model serialize/pt file
        serialized_file = self.manifest["model"]["serializedFile"]
        model_pt_path = os.path.join(model_dir, serialized_file)
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")

        self.model = torch.jit.load(model_pt_path)

        self.initialized = True

    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
        pred_out = self.model.forward(data)
        return pred_out


# The entry point for the script is the handler name
_service = FFTformerHandler()
