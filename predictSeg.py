import sys
sys.path.insert(0,"./")
# 加根目录、子目录，按需加
from pathlib import Path
root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "nets"))
sys.path.insert(0, str(root / "nets" / "basicsr"))
from nets.basicsr.models.archs.NAFNet_arch import NAFNet
from ultralytics import YOLO
from ultralytics.nn.modules.restoreBranch import RestoreBranchHasX,RestoreBranchNoX
from ultralytics.nn.modules.classBranch import ClassBranchHasX,ClassBranchNoX
# Load a model
model = YOLO("./runs/segment/1HasX/weights/epoch10.pt")  # load a pretrained model (recommended for training)

source = "./dataset1/images/train2017/0001.png"

model.predict(source, save=True, imgsz=1024, conf=0.2,iou=0.2)