#initialize variables in OpenVINO environment with: source /opt/intel/openvino/bin/setupvars.sh
import pytorch_test, openvino_test
import torch, time, subprocess
from datetime import datetime
import torchvision.models as models
resnet18_model = models.resnet18(pretrained=True)
# Create some sample input in the shape this model expects
dummy_input = torch.randn(10, 3, 224, 224)


# It's optional to label the input and output layers
input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
output_names = [ "output1" ]

# Use the exporter from torch to convert to onnx 
# model (that has the weights and net arch)
torch.onnx.export(resnet18_model, dummy_input, "resnet18.onnx", verbose=True, input_names=input_names, output_names=output_names)

#Convert ONNX model to Intel OpenVino IR (run on command line)
#/opt/intel/openvino/deployment_tools/model_optimizer/mo_onnx.py --input_model resnet18.onnx --scale_values=[58.395,57.120,57.375] --mean_values=[123.675,116.28,103.53] --reverse_input_channels --disable_resnet_optimization --disable_fusing --disable_gfusing --data_type=FP32 --output_dir fp32
#python3 classification_sample.py --labels test_model.labels  -m fp32/resnet18.xml -i dog.jpeg -d CPU

py_test = pytorch_test.Pytorch_test()
vino_test = openvino_test.Openvino_test()

t = time.time()
py_test.run('dog.jpeg')
print(time.time()-t)

t = time.time()
vino_test.main('test_model.labels', 'fp32/resnet18.xml', 'toyshop.jpeg',)
print(time.time()-t)
'''
t = datetime.utcnow()
#**********************************Import the line below as a module (import pytorch_onnx_openvino.classification_sampel???)
subprocess.call(['python3', 'classification_sample.py', '--labels', 'test_model.labels', '--model', 'fp32/resnet18.xml', '--input', 'toyshop.jpeg', '-d', 'CPU'])
elapsed_time = datetime.utcnow() - t
print(elapsed_time)

t = datetime.utcnow()

subprocess.call(['python3', 'pytorch_classification_sample.py'])
elapsed_time = datetime.utcnow() - t
print(elapsed_time) # cProfile.run better timer
'''