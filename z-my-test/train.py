#训练
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')

if __name__=='__main__':
    #模型权重文件路径
    model = YOLO(model=r'D:\YOLOV11\ultralytics-8.3.107\z-train-yamls\yolo11.yaml')
    #加载预训练权重，改进或者对比不建议使用，对整体精度提升不高
    model.load('yolo11n.pt')
    #数据参数配置
    model.train(data = r'D:\YOLOV11\ultralytics-8.3.107\datasets\BrokenSolarPanelDetection_test\data.yaml',#训练数据集文件路径
                imgsz=640,
                epochs=50, #训练轮数
                batch=4,
                workers=0,
                device='',optimizer='SGD',
                close_mosaic=10,
                resume=False,
                project='runs/train', #输出路径
                name = 'exp', #输出文件夹名称
                single_cls=False,
                cache=False
                )
