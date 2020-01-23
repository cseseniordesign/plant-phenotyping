__author__ = 'plant-phenotyping'


from zipfile import ZipFile
from pathlib import Path
import os
import cv2
import numpy as np
import sys
import argparse


def info(plant_ID, date):
    try:
        image_type = set()
        path = "output/"
        files = os.listdir(path)
        for file in files:
            if (plant_ID in file) and (date in file):
                file_name = file
                break

        with ZipFile(path + file_name, 'r') as zip:
            for file in zip.namelist():
                image_type.add(file.split('/')[1].split("_")[0])

        print("available image types:")
        for t in image_type:
            if t in ["Fluo", "IR", "Hyp", "Vis","Nir"]:
                print(t)

    except BaseException:
        print("please input correct date or plant name")




def unzip(plant_ID, date, image_type):
    if image_type not in ["Fluo", "IR", "Hyp", "Vis","Nir"]:
        print("please input correct image type")
    else:
        path = "output/"
        try:
            i = 0
            files = os.listdir(path)
            for file in files:
                if (plant_ID in file) and (date in file):
                    file_name = file
                    break

            folder_name = file_name[0:-4]

            with ZipFile(path + file_name, 'r') as zip:
                for file in zip.namelist():
                    if file.startswith(folder_name + '/' + image_type):
                        zip.extract(file)
                        i += 1
            if i == 0:
                print("image type not available")

        except BaseException:
            print("please input correct date or plant name")

        else:
            print("successfully reconstructed!")


def preprocess(plant_ID, date):

    flag = 0
    path = "./"
    files = os.listdir(path)
    for file in files:
        if (plant_ID in file) and (date in file) and "npy" not in file:
            hyp_dir_name = file
            flag = 1
            break
    if flag == 0:
        sys.exit('please input correct date or plant name')

    output_name = hyp_dir_name.split("_")[2]+"_"+hyp_dir_name.split("_")[3]
    hyp_dir = hyp_dir_name
    out_fn = output_name

    discard_imgs = ['0_0_0.png', '1_0_0.png']
    dir_path = Path(hyp_dir)
    dir_path= dir_path/'Hyp_SV_90'
    if not dir_path.exists():
        sys.exit('Hyp images are compressed, please unzip it first')
    imgs = list(dir_path.glob('*.png'))
    imgs = sorted(imgs, key=lambda x: int(x.name.split('_')[0]))
    num_imgs = len(imgs)
    print('%s images found.'%num_imgs)
    img_arrs = []
    for i in imgs:
        if not i.name in discard_imgs:
            arr = cv2.imread(str(i), cv2.IMREAD_GRAYSCALE)
            img_arrs.append(arr)
    img_array = np.stack(img_arrs, axis=2)
    print(img_array.shape)
    np.save(out_fn, img_array)
    print("numpy array successfully reconstructed!")



def main():
    ap = argparse.ArgumentParser()
    subparsers = ap.add_subparsers(dest='command')

    reconstruct = subparsers.add_parser('unzip', help='reconstruct specified images')
    reconstruct.add_argument("-n", "--name", required=True, help="plant ID")
    reconstruct.add_argument("-d", "--date", required=True, help="date")
    reconstruct.add_argument("-t", "--type", required=True, help="image type")

    hyp2arr = subparsers.add_parser('preprocess', help='convert hyp images to numpy array')
    hyp2arr.add_argument("-n", "--name", required=True, help="plant ID")
    hyp2arr.add_argument("-d", "--date", required=True, help="date")

    information = subparsers.add_parser('info', help='show available image types for specified plant name and date')
    information.add_argument("-n", "--name", required=True, help="plant ID")
    information.add_argument("-d", "--date", required=True, help="date")

    args = ap.parse_args()
    if args.command == 'unzip':
        plant_ID = args.name
        date = args.date
        image_type = args.type
        unzip(plant_ID, date, image_type)
    elif args.command == 'preprocess':
        plant_ID = args.name
        date = args.date
        preprocess(plant_ID, date)
    elif args.command == 'info':
        plant_ID = args.name
        date = args.date
        info(plant_ID, date)


if __name__=='__main__':
    main()