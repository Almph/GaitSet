from datetime import datetime
import numpy as np
import argparse

from model.initialization import initialization
from model.utils import evaluation
from config import conf
#所以import命令也可以导入变量。


def boolean_string(s):
    if s.upper() not in {'FALSE', 'TRUE'}:
        raise ValueError('Not a valid boolean string')
    return s.upper() == 'TRUE'


parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--iter', default='80000', type=int,
                    help='iter: iteration of the checkpoint to load. Default: 80000')
                    #指定该参数，以指定从第几个iteration的checkpoint加载模型参数进行测试。
parser.add_argument('--batch_size', default='1', type=int,
                    help='batch_size: batch size for parallel test. Default: 1')
parser.add_argument('--cache', default=False, type=boolean_string,
                    help='cache: if set as TRUE all the test data will be loaded at once'
                         ' before the transforming start. Default: FALSE')
opt = parser.parse_args()


# Exclude identical-view cases
def de_diag(acc, each_angle=False):
    result = np.sum(acc - np.diag(np.diag(acc)), 1) / 10.0
    if not each_angle:
        result = np.mean(result)
    return result


m = initialization(conf, test=opt.cache)[0]

# load model checkpoint of iteration opt.iter
print('Loading the model of iteration %d...' % opt.iter)
m.load(opt.iter)
#加载给定iter的模型参数。

print('Transforming...')
time = datetime.now()
test = m.transform('test', opt.batch_size)
#opt.batch_size默认是1。
#返回的test变量依次为：
# feature，数据总量*(62*255)，nparray，
# view_list, seq_type_list, label_list。

print('Evaluating...')
acc = evaluation(test, conf['data'])
#evaluation使用的cuda_dist函数没有被显式导入，却没有报错。

print('Evaluation complete. Cost:', datetime.now() - time)


#下面测试的部分是只适用于CASIA-B的，需要修改。



# Print rank-1 accuracy of the best model
# e.g.
# ===Rank-1 (Include identical-view cases)===
# NM: 95.405,     BG: 88.284,     CL: 72.041
print('Current dataset: ', conf['data']['dataset'])
for i in range(1):
    print('===Rank-%d (Include identical-view cases)===' % (i + 1))
    for j in range(acc.shape[0]):
        print('type %d: %.3f' % (j, np.mean(acc[j, :, :, i])))
    # print('NM: %.3f,\tBG: %.3f,\tCL: %.3f' % (
    #     np.mean(acc[0, :, :, i]),
    #     np.mean(acc[1, :, :, i]),
    #     np.mean(acc[2, :, :, i])))

# Print rank-1 accuracy of the best model，excluding identical-view cases
# e.g.
# ===Rank-1 (Exclude identical-view cases)===
# NM: 94.964,     BG: 87.239,     CL: 70.355
for i in range(1):
    print('===Rank-%d (Exclude identical-view cases)===' % (i + 1))
    for j in range(acc.shape[0]):
        print('type %d: %.3f' % (j, de_diag(acc[j, :, :, i])))
    # print('NM: %.3f,\tBG: %.3f,\tCL: %.3f' % (
    #     de_diag(acc[0, :, :, i]),
    #     de_diag(acc[1, :, :, i]),
    #     de_diag(acc[2, :, :, i])))

# Print rank-1 accuracy of the best model (Each Angle)
# e.g.
# ===Rank-1 of each angle (Exclude identical-view cases)===
# NM: [90.80 97.90 99.40 96.90 93.60 91.70 95.00 97.80 98.90 96.80 85.80]
# BG: [83.80 91.20 91.80 88.79 83.30 81.00 84.10 90.00 92.20 94.45 79.00]
# CL: [61.40 75.40 80.70 77.30 72.10 70.10 71.50 73.50 73.50 68.40 50.00]
np.set_printoptions(precision=2, floatmode='fixed')
for i in range(1):
    print('===Rank-%d of each angle (Exclude identical-view cases)===' % (i + 1))
    for j in range(acc.shape[0]):
        print('type %d:'%(j), de_diag(acc[j, :, :, i], True)))
    # print('NM:', de_diag(acc[0, :, :, i], True))
    # print('BG:', de_diag(acc[1, :, :, i], True))
    # print('CL:', de_diag(acc[2, :, :, i], True))
