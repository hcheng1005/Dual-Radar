import argparse
import glob
from pathlib import Path

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

from skimage import io
from matplotlib import pyplot as plt

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, img_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        #print(dataset_cfg.DATA_AUGMENTOR)
        self.use_data_type = dataset_cfg.DATA_AUGMENTOR.get('USE_DATA_TYPE', None)
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]
        data_file_list.sort()
        
        img_format = '.png'
        img_file_list = glob.glob(str(img_path / f'*{img_format}')) if img_path.is_dir() else [img_path]
        img_file_list.sort()
        
        self.sample_file_list = data_file_list
        self.img_file_list = img_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            print(self.use_data_type)
            if self.use_data_type == 'lidar':
               points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 6)
               #points = points[:,:4]
            else:
               points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 5)
               #points = points[:,:4] 
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        # 获取img
        # img = np.array(io.imread(self.img_file_list[index]), dtype=np.int32).transpose([2,1,0])

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    
    cfg_file = 'cfgs/dual_radar_models/pointpillar_arbe.yaml'
    data_path = '/data/chenghao/mycode/private/Dual-Radar/data/dual_radar/radar_arbe/testing/arbe'
    img_path = '/data/chenghao/mycode/private/Dual-Radar/data/dual_radar/radar_arbe/testing/image'
    ckpt = 'ckpt/pointpillars_arbe_80.pth'
    
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=cfg_file,
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default=data_path,
                        help='specify the point cloud data file or directory')
    parser.add_argument('--img_path', type=str, default=img_path,
                        help='specify the image data directory')
    parser.add_argument('--ckpt', type=str, default=ckpt, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), img_path=Path(args.img_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            # # 获取图片信息
            # img = np.array(io.imread(demo_dataset.img_file_list[data_dict['frame_id'][0]]), dtype=np.int32)
            # plt.figure()
            # plt.imshow(img)
            # plt.axis('off')
            # plt.show()
            
            V.draw_scenes(
                points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            )
            
            if not OPEN3D_FLAG:
                mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
