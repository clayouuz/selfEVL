from selfEVL import selfEVL
import argparse
import torch
import time

parser = argparse.ArgumentParser(description='self EValuation Learning')
parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
parser.add_argument('--learning_rate', type=float, default=0.1, help='learning rate (default: 0.1)')
parser.add_argument('--total_nc', default=100, type=int, help='class number for the dataset')
parser.add_argument('--fg_nc', default=50, type=int, help='the number of classes in first task')
parser.add_argument('--task_num', default=10, type=int, help='the number of incremental steps')
parser.add_argument('--gpu', default='0', type=str, help='GPU id to use')


args = parser.parse_args()

def main():
    cuda_index = 'cuda:' + args.gpu
    device = torch.device(cuda_index if torch.cuda.is_available() else "cpu")
    
        #每步增量学习的类别数。task_size = (总类别数 - 第一步类别数) / 总步数
    task_size = int((args.total_nc - args.fg_nc) / args.task_num)  
    file_name = args.data_name + '_' + str(args.fg_nc) + '_' + str(args.task_num) + '+' + str(task_size) 

    #定义模型，详见PASS.py
    model = selfEVL(args, task_size, device,file_name)

    class_set = list(range(args.total_nc))
    model.setup_data(shuffle=True, seed=1993)
    # for循环遍历所有任务，oldclass存入上一个任务的类别数，然后训练
    # 初始化时oldclass=0，即第一个任务的类别数为0
    for i in range(args.task_num+1):
        if i == 0:
            old_class = 0
        else:
            old_class = len(class_set[:args.fg_nc + (i - 1) * task_size])
        model.beforeTrain(i)
        model.train()
        model.afterTrain()


if __name__ == "__main__":
    time_start = time.time()
    main()
    time_end = time.time()
    print('time cost', time_end - time_start, 's')