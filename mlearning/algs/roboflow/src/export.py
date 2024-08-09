from ultralytics import YOLO
from mlearning.algs.roboflow.utils.helpers import get_params

def export(weights: str, target: str, params):
    
    reload_onnx = False
    params = get_params(params)
    if 'input_names' in params and isinstance(params['input_names'], str):
        params['input_names'] = [name.rstrip().lstrip() for name in params['input_names'].split(',')]
        reload_onnx = True
    if 'output_names' in params and isinstance(params['output_names'], str):
        params['output_names'] = [name.rstrip().lstrip() for name in params['output_names'].split(',')]
        reload_onnx = True
        
    model = YOLO(weights)
    print(f"Successfully LOADED weights: {weights} ")
    
    if target == 'onnx':
        export2onnx(model, weights, reload_onnx)
    else:
        NotImplementedError(f'There is no such case for {target}')

def export2onnx(model, weights, reload_onnx=False):
    import onnx
    model.export(format='onnx', opset=14)
    onnx_path = weights.replace(".pt", '.onnx')
    print(f"Saved onnx: {onnx_path}")
    if reload_onnx:
        onnx_model = onnx.load(onnx_path)
        print(f"Successfully LOADED onnx model to change input/output names: {onnx_path} ")
        
        if 'input_names' in params:
            print(f"Start to change input-names for {onnx_path} ")
            assert len(onnx_model.graph.input) == len(params['input_names']), RuntimeError(f"The number of onnx model input ({len(onnx_model.graph.input)}) must be same to input-names from yaml({len(params['input_names'])})")
            for idx, (input_tensor, input_name) in enumerate(zip(onnx_model.graph.input, params['input_names'])):
                print(f"  * {idx} changing {input_tensor.name} to {input_name}")
                input_tensor.name = input_name

        if 'output_names' in params:
            print(f"Start to change output names for {onnx_path} ")
            assert len(onnx_model.graph.output) == len(params['output_names']), RuntimeError(f"The number of onnx model output ({len(onnx_model.graph.output)}) must be same to output-names from yaml({len(params['output_names'])})")
            for jdx, (output_tensor, output_name) in enumerate(zip(onnx_model.graph.output, params['output_names'])):
                print(f"  * {jdx} changing {output_tensor.name} to {output_name}")
                output_tensor.name = output_name

            onnx.save(onnx_model, weights.replace(".pt", '_.onnx'))
            print(f"Successfully SAVED onnx model: {onnx_path} after changing input/output names")

if __name__ == '__main__':
    weights = "/DeepLearning/_projects/kt_g/wonchul/240807/train4/weights/best.pt"
    target = 'onnx'
    params = '/HDD/datasets/projects/mlearning/yolo/obb_detection/split_dataset_yolo_obb/export.yaml'
    
    export(weights, target, params)