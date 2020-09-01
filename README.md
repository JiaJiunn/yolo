# YOLOv3

A simple Tensorflow implementation of YOLOv3 I wrote to learn about its architecture. Only supports object inference.

The YOLOv3 object detection algorithm is based on [Joseph Redmon and Ali Farhadi’s famous paper](https://pjreddie.com/media/files/papers/YOLOv3.pdf). YOLO config files and weights are again from [Joseph Redmon](https://pjreddie.com/darknet/yolo/) himself. 

## Weights

After downloading the official YOLOv3 weights from [Joseph Redmon's website](https://pjreddie.com/darknet/yolo/), and placing them under `weights/`
```
./weights_to_ckpt.py
```
for converting the weight file into a ckpt format.

## Usage

For running YOLOv3 on image files, simply navigate into this directory and run
```
./yolo --images <location of image>
```
which will run YOLOv3 on the specified images.

<p align="center">
  <img src="example_output.png" width="400">
  <div align="center">
    example output
  </div>
</p>

By default, this implementation uses Joseph Redmon's YOLOv3 weights, trained on the COCO dataset. You can specify your own config file, weights file, and names file, using the `--cfg`, `--weights`, and `--names` flags respectively. You can also optionally use the `--nms_thresh`, `--iou_thresh`, and `--reso` flags to specify the respective parameters as desired (use the `-h` flag for usage details).

## References

I took quite some inspiration for the code structure from [Ayoosh Kathuria’s PyTorch implementation](https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/), as well as [Pawel Kapica's Medium article in TF-Slim](https://itnext.io/implementing-yolo-v3-in-tensorflow-tf-slim-c3c55ff59dbe). I also referenced [Pawel Kapica's implementation](https://github.com/mystic123/tensorflow-yolo-v3) on converting the official YOLOv3 weight formatting into checkpoint files.