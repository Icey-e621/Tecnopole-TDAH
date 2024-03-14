import visualkeras
import yolov5


# load pretrained model
model = yolov5.load("yolov5m.pt")
model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image
model.cpu  # i.e. device=torch.device(0)

visualkeras.layered_view(model, to_file='output.png').show() # write and show

visualkeras.layered_view(model)