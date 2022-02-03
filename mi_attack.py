import os
import torch as th
import torch
th.set_num_threads(1)
import argparse
from utils.perturb_adj import get_dp_graph
from utils.load_data import split_target_shadow, load_graphgallery_data, split_train_test
from utils.train import run_gnn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as Data
th.manual_seed(0)
import dgl
import numpy as np
from utils.model import GCN, MLP
import time
from utils.metrics import compute_acc, evaluate
import os
from scipy.special import softmax
import tqdm
from torch.autograd import Variable

def arg_parse():
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument('--gpu', type=int, default=-1,
        help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--dataset', type=str, default='cora')
    argparser.add_argument('--num_epochs', type=int, default=300)
    argparser.add_argument('--n_hidden', type=int, default=128)
    argparser.add_argument('--n_layers', type=int, default=2)
    argparser.add_argument('--batch_size', type=int, default=1000)
    argparser.add_argument('--lr', type=float, default=0.05)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--log-every', type=int, default=20)
    # argparser.add_argument('--eval-every', type=int, default=5)    
    argparser.add_argument('--model', type=str, default='gcn')
    argparser.add_argument('--mode', type=str, default='target')
    argparser.add_argument('--num_workers', type=int, default=4,
        help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--model_save_path', type=str, default='./data/save_model/gnn/')
    argparser.add_argument('--load_trained', type=str, default='no')
    argparser.add_argument('--dp', action='store_true')
    argparser.add_argument('--epsilon', type=int, default=8)
    argparser.add_argument('--delta', type=float, default=1e-5)
    argparser.add_argument('--noise_seed', type=int, default=42)
    argparser.add_argument('--noise_type', type=str, default='laplace')
    argparser.add_argument('--perturb_type', type=str, default='continuous')
    args = argparser.parse_args()

    if args.gpu >= 0:
        args.device = th.device('cuda:%d' % args.gpu)
    else:
        args.device = th.device('cpu')

    os.makedirs(args.model_save_path, exist_ok=True)
    os.makedirs('./log/', exist_ok=True)

    return args

def generate_attack_data(model, g, inputs, labels, val_nid, device, mode='member'):
    """
    Evaluate the model on the validation set specified by ``val_nid``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_nid : the node Ids for validation.
    batch_size : Number of nodes to compute at the same time.
    device : The GPU device to evaluate on.
    """
    model.eval()
    # print(model)
    # print(inputs.shape)
    with th.no_grad():
        pred = model.inference(g, inputs, device)
    att_data_x = softmax(pred[val_nid].detach().cpu().numpy(), axis=1)
    att_data_x = np.sort(att_data_x, axis=1)
    if mode == 'member':
        att_data_y = th.ones((att_data_x.shape[0])).type(th.LongTensor)
    elif mode == 'nonmember':
        att_data_y = th.zeros((att_data_x.shape[0])).type(th.LongTensor)
    att_data_x = th.FloatTensor(att_data_x)
    # print('len(pred)', len(pred))
    # print('len(labels)', len(labels))
    # print('val_nid', val_nid)
    return att_data_x, att_data_y

if __name__ == '__main__':

    args = arg_parse() 
    g, n_classes = load_graphgallery_data(args.dataset)
    
    in_feats = g.ndata['features'].shape[1]
    args.in_feats = in_feats
    args.n_classes = n_classes
    if args.dataset == 'citeseer':
        target_train_g, target_test_g = th.load('./data/citeseer_target.pt')
        shadow_train_g, shadow_test_g = th.load('./data/citeseer_shadow.pt')
    else:
        target_train_g, target_test_g = th.load('./data/target.pt')
        shadow_train_g, shadow_test_g = th.load('./data/shadow.pt')
    
    if args.dp:
        target_train_g = get_dp_graph(args, target_train_g)
        target_test_g = get_dp_graph(args, target_test_g)

        shadow_train_g = get_dp_graph(args, shadow_train_g)
        shadow_test_g = get_dp_graph(args, shadow_test_g)

        
    target_train_g.create_formats_()
    target_test_g.create_formats_()

    shadow_train_g.create_formats_()
    shadow_test_g.create_formats_()
        

    target_train_nid = th.tensor(range(0, len(target_train_g.nodes())))
    target_test_nid = th.tensor(range(0, len(target_test_g.nodes())))
    
    shadow_train_nid = th.tensor(range(0, len(shadow_train_g.nodes())))
    shadow_test_nid = th.tensor(range(0, len(shadow_test_g.nodes())))

    if args.dp:
        shadow_saving_path = os.path.join(args.model_save_path, 'dp_%s_%d_%s_%s_%s.pth'%(args.perturb_type, args.epsilon, args.dataset, args.model, 'shadow'))
        target_saving_path = os.path.join(args.model_save_path, 'dp_%s_%d_%s_%s_%s.pth'%(args.perturb_type, args.epsilon, args.dataset, args.model, 'target'))
    else:
        shadow_saving_path = os.path.join(args.model_save_path, '%s_%s_%s.pth'%( args.dataset, args.model, 'shadow'))
        target_saving_path = os.path.join(args.model_save_path, '%s_%s_%s.pth'%( args.dataset, args.model, 'target'))

    # Define model and optimizer
    shadow_model = GCN(args.in_feats, args.n_hidden, args.n_classes, args.n_layers, F.relu, args.batch_size, args.num_workers, args.dropout)
    shadow_model.load_state_dict(th.load(shadow_saving_path))
    shadow_model.eval()

    train_x1, train_y1 = generate_attack_data(shadow_model, shadow_train_g, shadow_train_g.ndata['features'], shadow_train_g.ndata['labels'], shadow_train_nid,  args.device)
    train_x0, train_y0 = generate_attack_data(shadow_model, shadow_test_g, shadow_test_g.ndata['features'], shadow_test_g.ndata['labels'], shadow_test_nid, args.device, mode='nonmember')
    
    train_x = th.cat((train_x1,train_x0),0)
    train_y = th.cat((train_y1,train_y0))
    train_x = Variable(train_x)
    train_y = Variable(train_y)
    # train_data_loader = Data.DataLoader(dataset=Data.TensorDataset(train_x, train_y), batch_size=10, shuffle=True)

    target_model = GCN(args.in_feats, args.n_hidden, args.n_classes, args.n_layers, F.relu, args.batch_size, args.num_workers, args.dropout)
    target_model.load_state_dict(th.load(target_saving_path))
    target_model.eval()
    test_x1, test_y1 = generate_attack_data(target_model, target_train_g, target_train_g.ndata['features'], target_train_g.ndata['labels'], target_train_nid,  args.device)
    test_x0, test_y0 = generate_attack_data(target_model, target_test_g, target_test_g.ndata['features'], target_test_g.ndata['labels'], target_test_nid, args.device, mode='nonmember')
    
    test_x = th.cat((test_x1,test_x0),0)
    test_y = th.cat((test_y1,test_y0))
    test_x = Variable(test_x)
    test_y = Variable(test_y)
    # test_data_loader = Data.DataLoader(dataset=Data.TensorDataset(test_x, test_y), batch_size=10, shuffle=True)

    attack_model = MLP(test_x.shape[1])
    optimizer = torch.optim.Adam(attack_model.parameters(), lr=args.lr)   
    loss_func = nn.CrossEntropyLoss() 

    ### training 
    for epoch in (range(args.num_epochs)):
        optimizer.zero_grad()
        output = attack_model(test_x)
        loss = loss_func(output, test_y)
        # print(train_x[:3], output[:3], train_y[:3])
        # exit()
        
        loss.backward()
        optimizer.step()
        # for step, (b_x, b_y) in enumerate(train_data_loader):   
        #     # print(b_x, b_y)
        #     output = attack_model(b_x)  

        #     loss = loss_func(output, b_y)
        #     optimizer.zero_grad()           
        #     loss.backward()                 
        #     optimizer.step()        
        #     # for name, parms in model.named_parameters():
	    #     #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight', torch.mean(parms.data), ' -->grad_value:', torch.mean(parms.grad))


        if epoch % 10 == 0:
            attack_model.eval()
            train_output = attack_model(train_x)
            train_acc = compute_acc(train_output, train_y)
            test_output = attack_model(test_x)
            test_acc = compute_acc(test_output, test_y)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| train accuracy: %.4f' % train_acc, '| test accuracy: %.4f' % test_acc)
            attack_model.train()
    # torch.save(attack_model, save_path)
    attack_model.eval()
    train_output = attack_model(train_x)
    train_acc = compute_acc(train_output, train_y)
    test_output = attack_model(test_x)
    test_acc = compute_acc(test_output, test_y)
    print('Epoch: ', args.num_epochs, '| train loss: %.4f' % loss.data.numpy(), '| train accuracy: %.4f' % train_acc, '| test accuracy: %.4f' % test_acc)
    with open("./log/attack_preformance.txt", "a") as wf:
        if args.dp:
            wf.write("%s, %s, %.3f, %.3f, %s, %s\n" % (args.dataset, args.model, train_acc, test_acc, args.perturb_type, args.epsilon))
        else:
            wf.write("%s, %s, %.3f, %.3f\n" % (args.dataset, args.model, train_acc, test_acc))

    if args.dp:
        saving_path = os.path.join(args.model_save_path, 'attack_dp_%s_%d_%s_%s_%s.pth'%(args.perturb_type, args.epsilon, args.dataset, args.model, args.mode))
    else:
        saving_path = os.path.join(args.model_save_path, 'attack_%s_%s_%s.pth'%( args.dataset, args.model, args.mode))
    print("Finish training, save model to %s"%(saving_path))
    th.save(attack_model.state_dict(), saving_path)


