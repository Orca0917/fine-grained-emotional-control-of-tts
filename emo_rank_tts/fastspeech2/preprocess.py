import os
import yaml
import random

from glob import glob

def prepare_dataset(preprocessed_path, speakers):

    # train, test 데이터셋 분리
    train_list, valid_list = [], []
    for speaker in speakers:
        paths = glob(os.path.join(preprocessed_path, speaker, '*.npz'))
        random.shuffle(paths)

        n_train = int(len(paths) * 0.8)
        train_list.extend(paths[:n_train])
        valid_list.extend(paths[n_train:])


    if not os.path.exists(os.path.join(preprocessed_path, 'fs2_train.txt')):
        # train, valid 데이터셋을 파일로 저장
        with open(os.path.join(preprocessed_path, 'fs2_train.txt'), 'w') as f:
            f.write('\n'.join(train_list) + '\n')

        with open(os.path.join(preprocessed_path, 'fs2_valid.txt'), 'w') as f:
            f.write('\n'.join(valid_list) + '\n')
    else:
        print('Already created. Skipping')


if __name__ == '__main__':

    config = yaml.safe_load(open('parameter.yaml', 'r'))

    preprocessed_path = config['path']['preprocessed_path']
    speakers = config['preprocessing']['speakers']

    prepare_dataset(preprocessed_path, speakers)
