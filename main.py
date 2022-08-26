import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse
from sklearn.model_selection import KFold
import numpy as np
import random

import load_data
from model import Net, accuracy
from gpu_mem_track import MemTracker


def arg_parse():
    parser = argparse.ArgumentParser(description='Arguments.')
    parser.add_argument('--bmname', type=str,
                        help='Name of the benchmark  dataset')
    parser.add_argument('--lr', type=float,
                        help='Learning rate.')
    parser.add_argument('--batch-size', type=int,
                        help='Batch size.')
    parser.add_argument('--epochs', type=int,
                        help='Number of epochs to train.')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                        help='Ratio of number of graphs training set to all graphs.')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                        help='Ratio of number of graphs testing set to all graphs.')
    parser.add_argument('--num-workers', type=int, default=3,
                        help='Number of workers to load data.')
    parser.add_argument('--input-dim', type=int,
                        help='Input feature dimension')
    parser.add_argument('--hidden-dim', type=int,
                        help='Hidden dimension')
    parser.add_argument('--output-dim', type=int,
                        help='Output dimension')
    parser.add_argument('--dropout', type=float,
                        help='Dropout rate.')
    parser.add_argument('--final-dropout', type=float,
                        help='Final dropout rate.')
    parser.add_argument('--num_kfold', type=int, default=10,
                        help='The number of the kFold crossing validation.')
    parser.add_argument('--device', type=str,
                        help='CPU or CUDA')
    parser.add_argument('--node_feat_type', type=str,
                        help='The type of the node feature: feat/deg/one')
    parser.add_argument('--seed', type=int, default=0,
                        help='The random seed of the project.')
    parser.add_argument('--num_gcn_layer', type=int, default=3,
                        help='The number of the gcn layer.')
    parser.add_argument('--num_landmark_gcn_layer', type=int, default=1,
                        help='The number of the gcn layer for the landmark.')
    parser.add_argument('--num_pool_layer', type=int, default=1,
                        help='The number of the pool layer.')
    parser.add_argument('--num_coarsen_graphs_list', type=str,
                        help='the number of the coarsen graph for each pool layer')
    parser.add_argument('--alpha_dist', type=float, default=0.1,
                        help='the coefficient to control the spectral dist in the loss function')
    parser.add_argument('--size_coarsen_graphs_list', type=str,
                        help='the size of the coarsen graph for each pool layer')

    parser.set_defaults(
        bmname='NCI109',
        lr=0.001,
        batch_size=64,
        epochs=300,
        hidden_dim=128,
        dropout=0.1,
        final_dropout=0.5,
        device='cpu',
        node_feat_type='feat',
        num_gcn_layers=3,
        num_landmark_gcn_layers=2,
        num_pool_layer=1,
        num_coarsen_graphs_list='6',
        size_coarsen_graphs_list='8',
        alpha_dist=0.1,
    )
    return parser.parse_args()


def echo_result(epoch, train_loss, train_acc, val_loss, val_acc, test_loss=None, test_acc=None):
    if test_loss is None and test_acc is None:
        print("\n EPOCH:{}\t "
              "Train Loss:{:.4f}\tTrain ACC: {:.4f}\t"
              "Val Loss:{:.4f}\t Val ACC: {:.4f}\t"
              .format(epoch, train_loss, train_acc, val_loss, val_acc),
              end='')
    else:
        print("\n EPOCH:{}\t "
              "Train Loss:{:.4f}\tTrain ACC: {:.4f}\t"
              "Val Loss:{:.4f}\t Val ACC: {:.4f}\t"
              "Test Loss:{:.4f}\t Test ACC: {:.4f}\t"
              .format(epoch, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc),
              end='')


def val(args, model, dataloader):
    model.eval()
    loss_func = nn.CrossEntropyLoss()

    predict_list = list()
    label_list = list()
    loss_list = list()
    for batch_graph, label in dataloader:
        prediction, sp_dist_loss = model(batch_graph)
        loss = loss_func(prediction, label) + args.alpha_dist * sp_dist_loss
        predict_list.append(prediction)
        label_list.append(label)
        loss_list.append(loss.detach().item())
    acc = accuracy(torch.cat(predict_list, 0), torch.cat(label_list, 0))
    loss_sum = np.sum(loss_list)
    return acc, loss_sum


def train(args, model, train_loader):

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    predict_list = list()
    label_list = list()
    loss_list = list()
    for batch_graph, labels in train_loader:
        prediction, sp_dist_loss = model(batch_graph)
        loss = loss_func(prediction, labels) + args.alpha_dist * sp_dist_loss
        # loss = loss_func(prediction, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predict_list.append(prediction)
        label_list.append(labels)
        loss_list.append(loss.detach().item())

        # a = loss.detach().item()
        # b = accuracy(prediction, labels)
        # loss_list = loss_list.append(loss.detach().item())
        # acc_list = acc_list.append(accuracy(prediction, label))
        # torch.cuda.empty_cache()

    acc = accuracy(torch.cat(predict_list, 0), torch.cat(label_list, 0))
    loss_sum = np.sum(loss_list)
    return acc, loss_sum


def benchmark_task_val(args, train_loader, val_loader, fold_id, test_loader=None):
    # split training-set and valuation-set
    gpu_tracker = MemTracker()
    # train_set = task_val_set[:int(8.0 / 9.0 * len(task_val_set))]
    # val_set = task_val_set[int(8.0 / 9.0 * len(task_val_set)):]
    # if test_set is not None:
    #     # test_loader = GraphDataLoader(test_set, batch_size=args.batch_size, drop_last=False, shuffle=True)
    #     test_loader = GraphDataLoader(test_set, batch_size=1, drop_last=False, shuffle=True)

    # Train model
    model = Net(args).to(args.device)
    for name, parameters in model.named_parameters():
        print(name, ':', parameters.device)
    # print(model)

    saved_epoch_id = 0

    max_val_acc = 0
    min_val_loss = 1e10

    train_acc_list = list()
    train_loss_list = list()
    val_acc_list = list()
    val_loss_list = list()
    test_acc_list = list()
    test_loss_list = list()

    for epoch in tqdm(range(args.epochs)):
        # print('Start Epoch{}'.format(epoch))
        train_acc, train_loss = train(args, model, train_loader)
        val_acc, val_loss = val(args, model, val_loader)
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)
        val_acc_list.append(val_acc)
        val_loss_list.append(val_loss)

        # glance on test
        test_acc, test_loss = val(args, model, test_loader)
        test_acc_list.append(test_acc)
        test_loss_list.append(test_loss)

        echo_result(epoch, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc)

        # if val_acc > max_val_acc or val_loss < min_val_loss:
        # if val_acc >= max_val_acc:
        if val_loss < min_val_loss:
            torch.save(model.state_dict(), './dump/model/' + 'Model-' + args.bmname +'.kpl')
            max_val_acc = val_acc
            min_val_loss = val_loss
            saved_epoch_id = epoch
            glance_train_acc = train_acc
            glance_train_loss = train_loss
            glance_test_acc = test_acc
            glance_test_loss = test_loss
            print("\033[92mModel saved at epoch {}\033[0m".format(saved_epoch_id))
        else:
            print()
        # gpu_tracker.track()

    print("\033[92mSUCCESS: Model Training Finished.\033[0m")
    echo_result(saved_epoch_id, glance_train_loss, glance_train_acc, min_val_loss, max_val_acc,
                glance_test_loss, glance_test_acc)
    # if fold_id == 0:
    #     draw_result(args.epochs, train_acc_list, "accuracy", './result/figure/'+ bmname + '-')
    #     draw_result(args.epochs, train_loss_list, "loss", './result/figure/' + bmname + '-')
    print("")

    return model


def model_test(args, model, test_loader):
    # test_loader = GraphDataLoader(test_set, batch_size=args.batch_size, drop_last=False, shuffle=True)
    # test_loader = GraphDataLoader(test_set, batch_size=1, drop_last=False, shuffle=True)
    acc, loss_sum = val(args, model, test_loader)
    return acc, loss_sum


def main(args):
    # Load benchmark data. Use k-Fold
    dataset = load_data.load_graph(args)
    fold_result_acc_list = list()
    fold_result_loss_list = list()
    kf = KFold(n_splits=args.num_kfold, random_state=args.seed, shuffle=True)
    for fold_id, (train_val_idx, test_idx) in enumerate(kf.split(dataset)):
        print("Fold:{}\t BEGIN".format(fold_id))
        random.shuffle(train_val_idx)
        len_val = len(test_idx)
        train_idx = train_val_idx[:-len_val]
        val_idx = train_val_idx[-len_val:]
        train_loader, val_loader, test_loader = load_data.split_dataset(args, train_idx, val_idx, test_idx, dataset)

        # Train Model
        model = benchmark_task_val(args, train_loader, val_loader, fold_id, test_loader=test_loader)
        # Test Model
        model.load_state_dict(torch.load('./dump/model/' + 'Model-' + args.bmname +'.kpl'))
        # test_model = torch.load('./dump/model/' + 'Model-' + args.bmname +'.kpl')
        test_acc, test_loss = model_test(args, model, test_loader)

        print("\033[92mFold {} classification Result :  \033[0m".format(fold_id))
        print("\033[92mTest Set ACC:{}\t Test Set Loss:{} \033[0m".format(test_acc, test_loss))

        fold_result_acc_list.append(test_acc)
        fold_result_loss_list.append(test_loss)
    avg_acc = np.average(fold_result_acc_list)
    std_acc = np.std(fold_result_acc_list)
    avg_loss = np.average(fold_result_loss_list)
    print("\n ACC List:")
    print(fold_result_acc_list)
    print("\n\n\033[92m K-FOLD Result:\033[0m")
    print("AVG ACC :{}".format(avg_acc))
    print("STD ACC :{}".format(std_acc))
    print("AVG LOSS :{}".format(avg_loss))
    with open('./result/value/10-fold.txt','w', encoding='utf-8') as f:
        for i in range(len(fold_result_acc_list)):
            f.write("Fold {} result:\n".format(i))
            f.write("ACC:{}\n".format(fold_result_acc_list[i]))
            f.write("LOSS:{}\n\n".format(fold_result_loss_list[i]))
        f.write("FINALLY Result\n")
        f.write("AVG ACC :{}\n".format(avg_acc))
        f.write("AVG LOSS :{}\n".format(avg_loss))


def set_seed(args):
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    print('\033[92mHello World\033[0m')
    args = arg_parse()
    set_seed(args)
    main(args)
