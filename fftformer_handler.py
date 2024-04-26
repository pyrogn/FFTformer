import io
import base64
from io import BytesIO
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
from ts.torch_handler.base_handler import BaseHandler

# Import your model definition
from basicsr.models.archs.fftformer_only import fftformer


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
        # Pad the input tensor to a compatible size
        pad_h = (8 - inputs.shape[2] % 8) % 8
        pad_w = (8 - inputs.shape[3] % 8) % 8
        inputs = torch.nn.functional.pad(inputs, (0, pad_w, 0, pad_h))

        # Run inference
        with torch.no_grad():
            outputs = self.model(inputs)

        return outputs

    def postprocess(self, inference_output):
        """
        Postprocessing involves converting the model output tensors to base64-encoded strings.
        :param inference_output: Inference output.
        :return: Postprocessed inference output.
        """
        # Convert the output tensor to base64-encoded strings
        output_data = []
        for tensor in inference_output:
            image = transforms.ToPILImage()(tensor.cpu().squeeze(0))
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            base64_img = base64.b64encode(buffered.getvalue()).decode("utf-8")
            output_data.append(base64_img)

        return output_data


# model = fftformer()


# def entry_point_function_name(data, context):
#     """
#     Works on data and context to create model object or process inference request.
#     Following sample demonstrates how model object can be initialized for jit mode.
#     Similarly you can do it for eager mode models.
#     :param data: Input data for prediction
#     :param context: context contains model server system properties
#     :return: prediction output
#     """
#     global model

#     if not data:
#         manifest = context.manifest

#         properties = context.system_properties
#         model_dir = properties.get("model_dir")
#         device = torch.device(
#             "cuda:" + str(properties.get("gpu_id"))
#             if torch.cuda.is_available()
#             else "cpu"
#         )

#         # Read model serialize/pt file
#         serialized_file = manifest["model"]["serializedFile"]
#         model_pt_path = os.path.join(model_dir, serialized_file)
#         if not os.path.isfile(model_pt_path):
#             raise RuntimeError("Missing the model.pt file")

#         model = torch.jit.load(model_pt_path)
#     else:
#         # infer and return result
#         return model(data)
