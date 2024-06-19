import os
import ast
import json
import torch
import argparse

import numpy as np
import torch.nn as nn

from dataset import CRCDataset, CRCDemoDataset
from models.crc_model import CRCModel
from models.config import MODE_ORDER
from utils.utils import set_seed, test_model_on_dataset, test_model_demo


set_seed(seed=2)
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, default='N', help="mode to test"
    )
    parser.add_argument(
        "--batch_size", type=int, default=128,
    )

    return parser.parse_args()


def get_evaluate_results(model, data_loader, cls_threshold, model_name, m=''):
    """Evaluation results"""
    results = {'model_name': model_name}
    acc, auc, precision, sensitivity, specificity, f1score = test_model_on_dataset(model, data_loader, device, cls_threshold)
    if m == '':
        results['acc'] = round(acc, 3)
        results['auc'] = round(auc, 3)
        results['precision'] = round(precision, 3)
        results['sensitivity'] = round(sensitivity, 3)
        results['specificity'] = round(specificity, 3)
        results['f1score'] = round(f1score, 3)
    else:
        results[f'{m}_acc'] = round(acc, 3)
        results[f'{m}_auc'] = round(auc, 3)
        results[f'{m}_precision'] = round(precision, 3)
        results[f'{m}_sensitivity'] = round(sensitivity, 3)
        results[f'{m}_specificity'] = round(specificity, 3)
        results[f'{m}_f1score'] = round(f1score, 3)

    return results


def evaluate():
    """Evaluate"""

    # evaluation save path
    demo_save_path = os.path.join('run', 'model_test', demo_model_name + '_' + cls_post)
    if not os.path.exists(demo_save_path):
        os.makedirs(demo_save_path, exist_ok=True)

    post = 'img'  # img

    model_name = demo_model_name + '_' + model_path.split('/')[-1].split('.')[0]
    demo_results_log_path = f'run/model_test/{demo_model_name}_{cls_post}/demo_results_{post}.log'
    if os.path.exists(demo_results_log_path):
        os.remove(demo_results_log_path)
    print(model_name)

    for i in mode_index:
        m = MODE_ORDER[i]
        print(f'mode {m}')
        # single mode
        results = get_evaluate_results(model, test_loader_single_mode[0], cls_threshold, model_name + '_' + m, m)
        with open(INDIVIDUAL_MODEL_RESULTS_FILE, 'a+') as f:
            f.write(json.dumps(results) + '\n')
        with open(demo_results_log_path, 'a+') as f:
            f.write('model_name:{}\n{}_acc:{} {}_auc:{} {}_precision:{} {}_sensitivity:{} {}_specificity:{} {}_f1score:{}\n'.format(
                results['model_name'], m, results[f'{m}_acc'], m, results[f'{m}_auc'], m, results[f'{m}_precision'], m, results[f'{m}_sensitivity'], m, results[f'{m}_specificity'], m, results[f'{m}_f1score']))

        for t in threshold:
            print('{} threshold:{}'.format(m, t))
            results = test_model_demo(model, test_loader_demo[0], t, cls_threshold, np.array(crc_test_path)[:, i], demo_model_name, post, device)
            with open(demo_results_log_path, 'a+') as f:
                f.write('{} threshold:{}\n acc:{:.3f} precision:{:.3f} sensitivity:{:.3f} specificity:{:.3f} f1_score:{:.3f}\n'.format(
                    m, t, results[0], results[1], results[2], results[3], results[4]))
            print('\n')


if __name__ == '__main__':
    args = parse_arguments()

    INDIVIDUAL_MODEL_RESULTS_FILE = 'run/model_test/individual_model_results.jsonl'

    if not os.path.exists('run/model_test'):
        os.makedirs('run/model_test')

    # init model
    base_model = CRCModel()
    base_model = nn.DataParallel(base_model)

    base_path = 'run/ckpt/Nmt'
    model_paths = [os.path.join(base_path, 'best.pth')]

    crc_test_path = []
    data_crc_test = 'data/crc_val.txt'

    for line in open(data_crc_test):
        temp_path = ast.literal_eval(line.strip())
        crc_test_path.append(ast.literal_eval(line.strip()))

    # mode index
    mode_index = []
    for m in args.mode:
        mode_index.append(MODE_ORDER.index(m))

    # Slice-level single mode test
    test_loader_single_mode = []
    for i in mode_index:
        m = MODE_ORDER[i]
        test_dataset_single_mode = CRCDataset(crc_test_path, mode=m, type='test')
        test_loader_single_mode.append(torch.utils.data.DataLoader(test_dataset_single_mode, batch_size=args.batch_size, shuffle=False, num_workers=24, pin_memory=True))
        print("mode {} slices: {}\n".format(m, len(test_loader_single_mode[0].dataset)))

    print('-' * 50 + '\n')

    # Patient-level single mode test
    test_loader_demo = []
    for i in mode_index:
        m = MODE_ORDER[i]
        test_dataset_demo = CRCDemoDataset(crc_test_path, mode=m)
        test_loader_demo.append(torch.utils.data.DataLoader(test_dataset_demo, batch_size=1, shuffle=False, num_workers=24, pin_memory=True))
        print("mode {} patients: {}\n".format(m, len(test_loader_demo[0].dataset)))

    cls_threshold = 0.5
    cls_post = str(cls_threshold).replace('.', '')

    # number slices of lesion
    threshold = [t for t in range(5, 10)]

    for j, model_path in enumerate(model_paths):
        assert os.path.exists(model_path)

        state_dict = torch.load(model_path)
        model = base_model
        model.load_state_dict(state_dict)
        model.eval()
        model.to(device)

        # evaluate
        demo_model_name = model_path.split('/')[-2]
        evaluate()
