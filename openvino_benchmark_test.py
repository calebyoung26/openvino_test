#initialize variables in OpenVINO environment with: source /opt/intel/openvino/bin/setupvars.sh
import pytorch_test, openvino_test
import torch, time, subprocess
from datetime import datetime
import torchvision.models as models

input_image = 'input_images/dog.jpeg'
resnet18_model = models.resnet18(pretrained=True)
# Create some sample input in the shape this model expects
dummy_input = torch.randn(10, 3, 224, 224)


# It's optional to label the input and output layers
input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
output_names = [ "output1" ]

# Use the exporter from torch to convert to onnx 
# model (that has the weights and net arch)
torch.onnx.export(resnet18_model, dummy_input, "resnet18.onnx", verbose=False, input_names=input_names, output_names=output_names)

#Convert ONNX model to Intel OpenVino IR (run on command line)
py_test = pytorch_test.Pytorch_test()
vino_test = openvino_test.Openvino_test()



t = time.time()
py_test.run(input_image)
pytorch_test_time = time.time()-t

t = time.time()
vino_test.main('test_model.labels', 'fp32/resnet18.xml', input_image,)
vino_test_time = time.time()-t

print('this test ran in {} seconds for PyTorch compared to {} seconds in openVINO'.format(pytorch_test_time,vino_test_time))