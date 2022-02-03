import os
import torch as th
th.set_num_threads(1)
import argparse
from utils.perturb_adj import get_dp_graph
from utils.load_data import split_target_shadow, load_graphgallery_data, split_train_test
from utils.train import run_gnn

def arg_parse():
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument('--gpu', type=int, default=-1,
        help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--dataset', type=str, default='cora')
    argparser.add_argument('--num_epochs', type=int, default=200)
    argparser.add_argument('--n_hidden', type=int, default=128)
    argparser.add_argument('--n_layers', type=int, default=2)
    argparser.add_argument('--batch_size', type=int, default=1000)
    argparser.add_argument('--lr', type=float, default=0.001)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=5)    
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
    argparser.add_argument('--load_data', action='store_true')
    args = argparser.parse_args()

    if args.gpu >= 0:
        args.device = th.device('cuda:%d' % args.gpu)
    else:
        args.device = th.device('cpu')

    os.makedirs(args.model_save_path, exist_ok=True)
    os.makedirs('./log/', exist_ok=True)

    return args


if __name__ == '__main__':
    
    

    args = arg_parse() 
    g, n_classes = load_graphgallery_data(args.dataset)

    
    in_feats = g.ndata['features'].shape[1]
    args.in_feats = in_feats
    args.n_classes = n_classes

    if args.load_data:
        if args.dataset=='citeseer':
            target_train_g, target_test_g = th.load('./data/citeseer_target.pt')
            shadow_train_g, shadow_test_g = th.load('./data/citeseer_shadow.pt')
        else:

            target_train_g, target_test_g = th.load('./data/target.pt')
            shadow_train_g, shadow_test_g = th.load('./data/shadow.pt')
    elif args.dataset=='citeseer':
        target_g, shadow_g = split_target_shadow(g)

        target_train_g, target_test_g = split_train_test(target_g)
        th.save((target_train_g, target_test_g), './data/citeseer_target.pt')
        shadow_train_g, shadow_test_g = split_train_test(shadow_g)
        th.save((shadow_train_g, shadow_test_g), './data/citeseer_shadow.pt')
    else:
        target_g, shadow_g = split_target_shadow(g)

        target_train_g, target_test_g = split_train_test(target_g)
        th.save((target_train_g, target_test_g), './data/target.pt')
        shadow_train_g, shadow_test_g = split_train_test(shadow_g)
        th.save((shadow_train_g, shadow_test_g), './data/shadow.pt')

    if args.dp:
        if args.mode == 'target':
            # target_train_g, target_test_g = split_train_test(target_g)
            target_train_g = get_dp_graph(args, target_train_g)
            target_test_g = get_dp_graph(args, target_test_g)

            target_train_g.create_formats_()
            target_test_g.create_formats_()

            run_data = target_train_g, target_test_g
    
        elif args.mode == 'shadow':
            # shadow_train_g, shadow_test_g = split_train_test(shadow_g)
            shadow_train_g = get_dp_graph(args, shadow_train_g)
            shadow_test_g = get_dp_graph(args, shadow_test_g)

            shadow_train_g.create_formats_()
            shadow_test_g.create_formats_()

            run_data = shadow_train_g, shadow_test_g
    
        train_acc, test_acc = run_gnn(args, run_data)

        
        with open("./log/dp_target_preformance.txt", "a") as wf:
            wf.write("%s, %s, %s, %s, %d, %.3f, %.3f\n" % (args.dataset, args.model, args.mode, args.perturb_type, args.epsilon, train_acc, test_acc))

    else: 
        if args.mode == 'target':
            # target_train_g, target_test_g = split_train_test(target_g)

            target_train_g.create_formats_()
            target_test_g.create_formats_()

            run_data = target_train_g, target_test_g
        
        elif args.mode == 'shadow':
            # shadow_train_g, shadow_test_g = split_train_test(shadow_g)

            shadow_train_g.create_formats_()
            shadow_test_g.create_formats_()

            run_data = shadow_train_g, shadow_test_g
        
        train_acc, test_acc = run_gnn(args, run_data)

        with open("./log/target_preformance.txt", "a") as wf:
            wf.write("%s, %s, %s, %.3f, %.3f\n" % (args.dataset, args.model, args.mode, train_acc, test_acc))

