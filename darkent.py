import darknet

configPath = "data/yolov3.cfg"
weightPath = "yolov3.weights"
metaPath = "data/coco.data"

filename = "person.jpg"
threshold = 0.25

net = darknet.load_net(bytes(configPath, "ascii"), bytes(weightPath, "ascii"), 0)
meta = darknet.load_meta(bytes(metaPath, "ascii"))
detection = darknet.detect(net, meta, bytes(filename, "ascii"), threshold)

print(detection)