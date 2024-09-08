from selfEVL import selfEVL
import argparse
import torch
import time

parser = argparse.ArgumentParser(description='self EValuation Learning')

parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=51, help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=1, help='learning rate (default: 0.001)')
#sam
parser.add_argument("--depth", default=16, type=int, help="Number of layers.")
parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
parser.add_argument("--width_factor", default=8, type=int, help="How many times wider compared to normal ResNet.")
parser.add_argument("--adaptive", default=True, type=bool, help="True if you want to use the Adaptive SAM.")
parser.add_argument("--rho", default=2.0, type=int, help="Rho parameter for SAM.")
parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    

parser.add_argument('--total_nc', default=100, type=int, help='class number for the dataset')
parser.add_argument('--fg_nc', default=10, type=int, help='the number of classes in first task')
parser.add_argument('--task_num', default=10, type=int, help='the number of incremental steps')
parser.add_argument('--gpu', default='0', type=str, help='GPU id to use')
parser.add_argument('--data_name', default='CIFAR100', type=str, help='the name of dataset')


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