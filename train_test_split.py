import numpy as np
import random
import os


def get_files(in_dirname, class_list):
    in_filename = []

    classes = {'ONE':0, 'TWO':0, 'THREE':0, 'FOUR':0, 'FIVE':0, 'SIX':0, 'SEVEN':0, 'EIGHT':0, 'NINE':0, 'TEN':0}
    for root, dirs, files in os.walk(in_dirname):
        for file in files:
            if '.jpg' in file:
                temp1 = file.split('.')
                temp2 = temp1[0].split('_')
                if len(temp2) == 2:
                    classname = temp2[0]
                    classes[classname] += 1

    for root, dirs, files in os.walk(in_dirname):
        for class_name in class_list:
            temp_list = []
            for file in files:
                if '.jpg' in file:
                    temp1 = file.split('.')
                    temp2 = temp1[0].split('_')
                    classname = temp2[0]
                    if len(temp2) == 2:
                        if class_name == classname:
                            temp_list.append(root + file)
            in_filename.append(temp_list)
    return in_filename, classes


def main():
    root_dir = '/home/ai_competition10/Project/'
    image_dir_train = root_dir + "data/"
    image_dir_val = root_dir + "data_val/"
    train_dir = root_dir + "train/"
    val_dir = root_dir + "val/"

    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(val_dir):
        os.mkdir(val_dir)
    class_list = ['ONE', 'TWO', 'THREE', 'FOUR', 'FIVE', 'SIX', 'SEVEN', 'EIGHT', 'NINE', 'TEN']

    train_list = []
    val_list = []

    print("start program!")

    files_train, classes_train = get_files(image_dir_train, class_list)
    files_val, classes_val = get_files(image_dir_val, class_list)

    for i, class_name in enumerate(class_list):
        temp = random.sample(files_train[i], 500)
        train_list.append(temp[0:500])
        temp = random.sample(files_val[i], 100)
        val_list.append(temp[0:100])

    train_list = [j for sub in train_list for j in sub]
    val_list = [j for sub in val_list for j in sub]

    train_txt = open(root_dir + "darknet/data/train.txt", "w")
    for path in train_list:
        print("Train copy : {}".format(path))
        os.system("cp -r {} {}".format(path, train_dir))
        temp = path.split('.')
        os.system("cp -r {} {}".format(temp[0] + '.txt', train_dir))
        train_txt.write(path + '\n')

    val_txt = open(root_dir + "darknet/data/val.txt", "w")
    for path in val_list:
        print("Validation copy : {}".format(path))
        os.system("cp -r {} {}".format(path, val_dir))
        temp = path.split('.')
        os.system("cp -r {} {}".format(temp[0] + '.txt', val_dir))
        val_txt.write(path + '\n')


if __name__ == '__main__':
    main()
