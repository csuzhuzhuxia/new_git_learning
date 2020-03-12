from mmdet.apis import inference_detector, init_detector
import os
import numpy as np
import argparse
from tqdm import tqdm
import json
import glob
import cv2
import time

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

def subAB(A_Img_Path, B_Img_Path, on):
    A_Img = cv2.imread(A_Img_Path)
    if on:
        B_Img = cv2.imread(B_Img_Path)
    else:
        B_Img = B_Img_Path
    err = cv2.subtract(B_Img, A_Img)
    err = cv2.bitwise_not(err)
    return err

class Result:
    def __init__(self, model_path_1,model_path_2, config_path, json_out_path, pic_path):
        self.init_start = time.time()
        self.model1 = None
        self.model2 = init_detector(config_path, model_path_2, device='cuda:0')
        self.pics = glob.glob(os.path.join(pic_path, 'img_*'))
        self.bottle_pics = np.unique([name.split('_')[-2] for name in glob.glob(os.path.join(pic_path, 'imgs_*'))])
        self.json_out_path = json_out_path
        self.pic_path = pic_path
        self.images = []
        self.annotations = []
        self.num_img_id = 1
        self.num_ann_id = 1
        self.init_end = time.time()

    def add_result(self, file_name, result_):
        # {"file_name":"cat.jpg", "id":1, "height":1000, "width":1000},
        images_anno = {}
        images_anno['file_name'] = file_name
        images_anno['id'] = self.num_img_id
        self.images.append(images_anno)

        # {"image_id":1, "bbox":[100.00, 200.00, 10.00, 10.00], "category_id": 1}
        for i, boxes in enumerate(result_, 1):
            # 标注的长度不为0
            if len(boxes):
                defect_label = i
                for box in boxes:
                    if(float(box[4])<0.3 and defect_label==1):
                        continue
                    else:
                        anno = {}
                        anno['image_id'] = self.num_img_id
                        anno['category_id'] = defect_label+10
                        anno['bbox'] = [round(float(i), 2) for i in box[0:4]]
                        anno['bbox'][2] = anno['bbox'][2] - anno['bbox'][0]
                        anno['bbox'][3] = anno['bbox'][3] - anno['bbox'][1]
                        anno['score'] = float(box[4])
                        self.annotations.append(anno)
                        self.num_ann_id += 1
        self.num_img_id += 1

    def det_result(self):
        self.inference_start = time.time()
        print('min pic ...')
        for im in tqdm(self.pics):
            # img = os.path.join(self.pic_path, im)
            img = im
            read_img = cv2.imread(img)
            if (read_img.shape[0]==3000 and read_img.shape[1]==4096):
                result_ = inference_detector(self.model2, read_img)
                file_name = img.split('/')[-1]
                self.add_result(file_name=file_name, result_=result_)
            else:
                pass

        self.inference_mid = time.time()
        print('max pic ...')
        for im_ in tqdm(self.bottle_pics):
            imgs = np.sort(glob.glob(os.path.join(self.pic_path, 'imgs_' + im_ + '*')))[::-1]
            first_data = cv2.imread(imgs[0])

            for i in range(len(imgs) - 1):
                err = subAB(imgs[i], imgs[i + 1], 1)
                result_ = inference_detector(self.model2, err)
                file_name = imgs[i].split('/')[-1]
                self.add_result(file_name=file_name, result_=result_)

            err = subAB(imgs[-1], first_data, 0)
            result_ = inference_detector(self.model2, err)
            file_name = imgs[-1].split('/')[-1]
            self.add_result(file_name=file_name, result_=result_)
        self.inference_end = time.time()

    def outJson(self):
        self.json_start = time.time()
        meta = {}
        meta['images'] = self.images
        meta['annotations'] = self.annotations
        with open(self.json_out_path, 'w') as fp:
            json.dump(meta, fp, cls=MyEncoder, indent=0, separators=(',', ': '))
        self.json_end = time.time()

    def outInfo(self):
        print('Num of Image:', self.num_img_id - 1)
        print('Num of Ann:', self.num_ann_id - 1)
        print('Time of Task', round(self.json_end - self.init_start, 2), 's')
        print('Time of Model Init:', round(self.init_end - self.init_start, 2), 's')
        print('Time of Inference Min Pic:', round(self.inference_mid - self.inference_start, 2), 's')
        print('Time of Inference Max Pic:', round(self.inference_end - self.inference_mid, 2), 's')
        print('Time of Inference All Pic:', round(self.inference_end - self.inference_start, 2), 's')
        print('Time of Json Out:', round(self.json_end - self.json_start, 2), 's')


#  python out_result.py -c ./submit.py -m ./submit.pth -im /tcdata/testA/images -o ./result.json
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate result")
    parser.add_argument("-m", "--model", help="Model path", type=str, )
    parser.add_argument("-c", "--config", help="Config path", type=str, )
    parser.add_argument("-im", "--im_dir", help="Image path", type=str, )
    parser.add_argument('-o', "--out", help="Save path", type=str, )
    args = parser.parse_args()
    model_path = args.model
    config_path = args.config
    json_out_path = args.out
    pic_path = args.im_dir
    print(pic_path)
    result = Result("",model_path_2=model_path, config_path=config_path, json_out_path=json_out_path, pic_path=pic_path)
    result.det_result()
    result.outJson()
    result.outInfo()
    print('Finish!')

