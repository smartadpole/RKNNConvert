import sys, os

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '../../'))

import time
import sys
import numpy as np
import cv2
from rknn.api import RKNN
import argparse
from onnxmodel import ONNXModel
from file import MkdirSimple, ReadImageList

W = 644
H = 392

def parse_args():
    parser = argparse.ArgumentParser(description="Convert ONNX model to RKNN format")
    parser.add_argument("--onnx_model", type=str, required=True, help="Path to the ONNX model.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image.")
    return parser.parse_args()

def test_dir(image_dir, rknn, output_dir, width, height, rgb):
    img_list = ReadImageList(image_dir)
    print("test image number: ", len(img_list))
    for file in img_list:
        image, depth = test_image(file, rknn, width, height, rgb)
        depth_file = os.path.join(output_dir, 'depth', os.path.basename(file))
        concat_file = os.path.join(output_dir, 'concat', os.path.basename(file))
        MkdirSimple(depth_file)
        MkdirSimple(concat_file)
        cv2.imwrite(concat_file, image)
        cv2.imwrite(depth_file, depth)

    print('output shape is {}'.format(depth.shape))

def test_image(image_path, rknn, width, height, rgb):
    # Set inputs
    img_org = cv2.imread(image_path)
    img = cv2.resize(img_org, (width, height), cv2.INTER_LANCZOS4)
    if rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.expand_dims(img, 0)
    inputs = [img,]

    start_time = time.time()
    outputs = rknn.inference(inputs, data_format='nhwc')
    end_time = time.time()
    print(f"Inference time: {(end_time - start_time) * 1000:.2f} ms")

    dis_array = outputs[0][0][0]
    dis_array = (dis_array - dis_array.min()) / (dis_array.max() - dis_array.min()) * 255.0
    dis_array = dis_array.astype("uint8")

    depth = cv2.resize(dis_array, (img_org.shape[1], img_org.shape[0]))
    depth = cv2.applyColorMap(cv2.convertScaleAbs(depth, 1), cv2.COLORMAP_PARULA)
    combined_img = np.vstack((img_org, depth))

    return combined_img, depth

if __name__ == '__main__':
    args = parse_args()
    output_file = args.onnx_model.replace('onnx', 'rknn')
    MkdirSimple(output_file)

    modelonnx = ONNXModel(args.onnx_model)
    c, height, width = modelonnx.get_input_size()
    # exit()

    # Create RKNN object
    rknn = RKNN(verbose=True)

    # pre-process config
    print('--> config model')
    rknn.config(mean_values=[x * 255 for x in [0.485, 0.456, 0.406]], std_values=[x * 255 for x in [0.229, 0.224, 0.225]], target_platform='rk3588')
    print('done')

    # Load model
    ONNXModel(args.onnx_model)
    print('--> Loading model')
    ret = rknn.load_onnx(model=args.onnx_model)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    ONNXModel(args.onnx_model)
    print('--> Building model')
    ret = rknn.build(do_quantization=False, dataset='./dataset.txt')
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn(output_file)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')


    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')
    print("export rknn to {}".format(output_file)

    test_dir(args.image, rknn, os.path.dirname(output_file), width, height, c == 3)

    rknn.release()