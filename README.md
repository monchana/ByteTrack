<div align="center">   

# RideFlux AI Challenge : Object Tracking
</div>

> Forked from [**ByteTrack: Multi-Object Tracking by Associating Every Detection Box**](git@github.com:ifzhang/ByteTrack.git)
> 
> Yifu Zhang, Peize Sun, Yi Jiang, Dongdong Yu, Zehuan Yuan, Ping Luo, Wenyu Liu, Xinggang Wang
> 
> *[arXiv 2110.06864](https://arxiv.org/abs/2110.06864)*

## Installation

[DynamicHead](https://github.com/monchana/DynamicHead)와 달리 `Conda` 또는 `Docker` 상관없음.

### 1. Installing on the host machine
Step1. Install ByteTrack.
```shell
git clone https://github.com/ifzhang/ByteTrack.git
cd ByteTrack
pip3 install -r requirements.txt
python3 setup.py develop
```

Step2. Install [pycocotools](https://github.com/cocodataset/cocoapi).

```shell
pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

Step3. Others
```shell
pip3 install cython_bbox
```
### 2. Docker build
```shell
docker build -t bytetrack:latest .

# Startup sample
mkdir -p pretrained && \
mkdir -p YOLOX_outputs && \
xhost +local: && \
docker run --gpus all -it --rm \
-v $PWD/pretrained:/workspace/ByteTrack/pretrained \
-v $PWD/datasets:/workspace/ByteTrack/datasets \
-v $PWD/YOLOX_outputs:/workspace/ByteTrack/YOLOX_outputs \
-v /tmp/.X11-unix/:/tmp/.X11-unix:rw \
--device /dev/video0:/dev/video0:mwr \
--net=host \
-e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
-e DISPLAY=$DISPLAY \
--privileged \
bytetrack:latest
```

## Data preparation

- 만약 추가 데이터를 넣거나 학습을 시킬 경우, 원본 저장소의 [Data Preparation](https://github.com/ifzhang/ByteTrack#data-preparation) 구조를 따를 것. 
- 해당 저장소에서는 모델을 별도로 학습하지 않고, object detection으로 생성한 결과에 tracker만 돌렸음



## Model zoo

학습이 필요할 경우 원본 저장소의 [Model Zoo](https://github.com/ifzhang/ByteTrack#model-zoo) 참조

## Combining BYTE with other detectors

Object Detector 에서 참조한 결과를 Infer하여 Tracking 결과 생성

- `exps/yolox_m_mix_det.py`를 필요에 따라 적절히 수정
- `det2track.py`에서 `TODO`를 검색하고 OD Infer 결과 파일 위치와, 최종 결과를 저장할 위치 지정
- 파일 저장 형식은 [yolox/tracker/byte_tracker]((https://github.com/monchana/ByteTrack/blob/8d52fbdf9cd03757d8dd02c0631e526d164bd726/yolox/tracker/byte_tracker.py)) 참조
- 코드 작성 포맷은 [여기](https://github.com/ifzhang/ByteTrack/issues/69)를 참고
- `total_results.json`은 `IMG_ID`을 Key로, `real_results.json`은 `FILE_NAME`을 Key로 저장하는 딕셔너리 형태


```
python det2track.py
```

## Forked From 

```
@article{zhang2021bytetrack,
  title={ByteTrack: Multi-Object Tracking by Associating Every Detection Box},
  author={Zhang, Yifu and Sun, Peize and Jiang, Yi and Yu, Dongdong and Yuan, Zehuan and Luo, Ping and Liu, Wenyu and Wang, Xinggang},
  journal={arXiv preprint arXiv:2110.06864},
  year={2021}
}
```

