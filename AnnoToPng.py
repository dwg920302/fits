import cv2
import numpy as np
import argparse
import json
from os import listdir
from os.path import isfile, join, splitext

parser = argparse.ArgumentParser()

parser.add_argument('--seg_dir', type=str, default= 'Model-Pose_resized',
                    help='input image directory path')
parser.add_argument('--output_dir', type=str,
                    help='output directory path')
parser.add_argument('--data_type', type=int,
                    help='0=Model, 1=Item')
args = parser.parse_args()

def main():
    if args.seg_dir is None:
        print("error. info_file is required")
        return 0
    if args.output_dir is None:
        print("error. pose_dir is required")
        return 0
    if args.data_type is None:
        print("error. img_dir is required")
        return 0

    seg_files = [ join(args.seg_dir,f) for f in listdir(args.seg_dir) if isfile(join(args.seg_dir,f)) ] 
    num_seg = 0
    
    if args.data_type == 0 :
        color_scheme = {
        'hat': (1, 1, 1),
        'hair': (2, 2, 2),
        'face': (3, 3, 3),
        'neck': (4, 4, 4),
        'outer_torso': (5, 5, 5),
        'outer_rsleeve': (6, 6, 6),
        'outer_lsleeve': (7, 7, 7),
        'inner_torso': (8, 8, 8),
        'inner_rsleeve': (9, 9, 9),
        'inner_lsleeve': (10, 10, 10),
        'right_arm': (11, 11, 11),
        'left_arm': (12, 12, 12),
        'pants_hip': (13, 13, 13),
        'pants_rsleeve': (14, 14, 14),
        'pants_lsleeve': (15, 15, 15),
        'skirt': (16, 16, 16),
        'right_leg': (17, 17, 17),
        'left_leg': (18, 18, 18),
        'right_shoe': (19, 19, 19),
        'left_shoe': (20, 20, 20),
        }
        anno_name = 'category_name'
    elif args.data_type == 1:
        color_scheme = {
        'hat': (1, 1, 1),
        'hat_hidden': (2, 2, 2),
        'rsleeve': (3, 3, 3),
        'lsleeve': (4, 4, 4),
        'torso': (5, 5, 5),
        'top_hidden': (6, 6, 6),
        'hip': (7, 7, 7),
        'pants_rsleeve': (8, 8, 8),
        'pants_lsleeve': (9, 9, 9),
        'pants_hidden': (10, 10, 10),
        'skirt': (11, 11, 11),
        'skirt_hidden': (12, 12, 12),
        'shoe': (13, 13, 13),
        'shoe_hidden': (14, 14, 14),
        }
        anno_name = 'product_type'
    else:
        print('**WRONG DATA TYPE!')
        exit(0)

    for i in range(len(seg_files)):
        tmpf = open(seg_files[i])
        ann = json.load(tmpf)

        ##<< [ SC load image for testing ]
        result = np.zeros((1280,720,3),np.uint8)
        for key in ann.keys():
            if not key.startswith('region'):
                #print('continue: ',key)
                continue
            ##<< [ SC : if 'segmentation' has multiple list ]
            for l in range(len(ann[key]['segmentation'])):
                tmpresult = np.zeros((1280,720,3),np.uint8)
                seg = ann[key]['segmentation'][l]
                poly = []
                for pt in seg:
                    x, y = round(pt[0]), round(pt[1])
                    poly.append([x,y])
                pts = np.array(poly, np.int32)
                pts = pts.reshape((-1,1,2))
                color = color_scheme[ann[key][anno_name]]
                tmpresult = cv2.fillPoly(tmpresult, [pts], color)
                result = np.maximum(result, tmpresult)
                #result += tmpresult

        #print('target file:',join(args.output_dir,seg_files[i][len(args.seg_dir)+1:-5]+'.png'))
        result_file = join(args.output_dir,seg_files[i][len(args.seg_dir)+1:-4]+'.png') 
        print(result_file)
        #exit(0)
        cv2.imwrite(result_file, result)

        #exit(0)
        
        #print(ann)
        #exit(0)



#for image_name, annon_name in zip(train_image_list, train_annon_list):
#        count += 1
#        image = cv2.imread(ORIGIN_TRAIN_IMAGE_DIR + image_name)
#        with open(ORIGIN_TRAIN_ANNON_DIR + annon_name) as f:
#            annon = json.load(f)
#            
#        result = np.zeros(image.shape, np.uint8)
#    #     result = image.copy() # 테스트용\n
    
#        for key in annon.keys():
#            if not key.startswith('region'):
#                continue
#            for seg in annon[key]['segmentation']:
#                poly = []
#                for point in seg:
#                    x, y = round(point[0]), round(point[1])
#                    poly.append([x, y])
#            
#                pts = np.array(poly, np.int32)
#                pts = pts.reshape((-1, 1, 2))
#                color = color_scheme[annon[key]['product_type']] # 착용샷 생성 시
#                result = cv2.fillPoly(result, [pts], color)
            
    ### 테스트용
    #     plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    #     plt.show()
    #     plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    #     plt.show()
    ###
        
#        cv2.imwrite(TRAIN_LABEL_DIR + os.path.splitext(annon_name)[0] + '.png', result)


if __name__ == '__main__':
    main()

