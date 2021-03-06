import cv2
from tensorpack import *
import glob
import os

import argparse
import numpy as np
import zipfile
import random
from tensorpack import RNGDataFlow, MapDataComponent, dftools, FixedSizeData


zip_folder = '/Users/apple/Downloads/Computergrafik/zipfile'
#zip_folder = '/graphics/scratch/mallick/KITTI'


class ImageEncode(MapDataComponent):
    def __init__(self, ds, mode='.jpg', dtype=np.uint8, index=0):
        def func(img):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return np.asarray(bytearray(cv2.imencode(mode, img)[1].tostring()), dtype=dtype)

        super(ImageEncode, self).__init__(ds, func, index=index)


class ImageDecode(MapDataComponent):
    def __init__(self, ds, mode='.jpg', dtype=np.uint8, index=0):
        def func(im_data):
            img = cv2.imdecode(np.asarray(bytearray(im_data), dtype=dtype), cv2.IMREAD_COLOR)
            return img

        super(ImageDecode, self).__init__(ds, func, index=index)


class KittiFromZIPFile(RNGDataFlow):
    """ Produce images read from a list of zip files. """

    def __init__(self, zip_folder, shuffle=False):
        """
        Args:
            zip_file (list): list of zip file paths.
        """

        # assert os.path.isfile(zip_file)
        self.archivefiles = []
        zip_list = [str for str in os.listdir(zip_folder) if 'drive' in str]
        # print zip_list
        for zip_file in zip_list:
            zip_dir = os.path.join(zip_folder, zip_file)
            assert os.path.isfile(zip_dir)
            self.shuffle = shuffle
            archive = zipfile.ZipFile(zip_dir)
            imagesInArchive = archive.namelist()
            for img_name in imagesInArchive:
                if img_name.endswith('.png') and ('image_02' in img_name or 'image_03' in img_name):
                    self.archivefiles.append((archive, img_name))


    def get_data(self):
        if self.shuffle:
            self.rng.shuffle(self.archivefiles)
        # self.archivefiles = random.sample(self.archivefiles, self.max)
        i = 0
        for archive in self.archivefiles:
            if ('image_02' in archive[1]):
                left_img = archive[0].read(archive[1])
                right_img = archive[0].read(archive[1].replace('image_02', 'image_03'))

                yield [left_img, right_img]


def main():
    # create
    ds = KittiFromZIPFile(zip_folder)
    ds = FixedSizeData(ds, 500)
    dftools.dump_dataflow_to_lmdb(ds, 'train2.lmdb')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--str', help='dummy string', type=str)
    parser.add_argument('--apply', action='store_true', help='apply')
    args = parser.parse_args()
    main()



