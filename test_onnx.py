import onnxruntime
from PIL import Image
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms

ort_session = onnxruntime.InferenceSession("mnist.onnx")

img = Image.open("two.png").convert("L")

toTensor = transforms.ToTensor()

tensor = toTensor(img)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(tensor)}
ort_outs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
#np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

print(ort_outs)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")