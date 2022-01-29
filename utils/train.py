import torch as th
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
th.manual_seed(0)
import dgl
import numpy as np
from utils.model import GCN
import time
from utils.metrics import compute_acc, evaluate
import os

def run_gnn(args, data):
    train_g, test_g = data
        
    train_nid = th.tensor(range(0, len(train_g.nodes())))
    test_nid = th.tensor(range(0, len(test_g.nodes())))
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    dataloader = dgl.dataloading.NodeDataLoader(
        train_g,
        train_nid,
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers)
    
    # Define model and optimizer
    model = GCN(args.in_feats, args.n_hidden, args.n_classes, args.n_layers, F.relu, args.batch_size, args.num_workers, args.dropout)
    print(model)
    model = model.to(args.device)
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    avg = 0
    iter_tput = []
    for epoch in range(args.num_epochs):
        tic = time.time()

        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.   
        tic_step = time.time()
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            # Load the input features as well as output labels

            blocks = [block.int().to(args.device) for block in blocks]
            batch_inputs = blocks[0].srcdata['features']
            batch_labels = blocks[-1].dstdata['labels'].to(device=args.device, dtype=th.long)

            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)
            batch_pred = F.softmax(batch_pred, dim=1)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_tput.append(len(seeds) / (time.time() - tic_step))
            if step % args.log_every == 0:
                acc = compute_acc(batch_pred, batch_labels)
                # gpu_mem_alloc = th.cuda.max_memory_allocated() / (1024*1024) if th.cuda.is_available() else 0
                print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f}'.format(
                    epoch, step, loss.item(), acc.item(), np.mean(iter_tput[3:])))
            tic_step = time.time()

        toc = time.time()
        print('Epoch %d, Time(s):%.4f'%(epoch, toc - tic))
        if epoch >= 5:
            avg += toc - tic
        if epoch % args.eval_every == 0 and epoch != 0:
            train_acc, train_pred = evaluate(model, train_g, train_g.ndata['features'], train_g.ndata['labels'], train_nid, args.device)
            print('Train Acc {:.4f}'.format(train_acc))
            # print(train_pred[0])

            test_acc, test_pred = evaluate(model, test_g, test_g.ndata['features'], test_g.ndata['labels'], test_nid, args.device)
            print('Test Acc: {:.4f}'.format(test_acc))
            # print(test_pred[0])
    
    if args.dp:
        saving_path = os.path.join(args.model_save_path, 'dp_%s_%d_%s_%s_%s.pth'%(args.perturb_type, args.epsilon, args.dataset, args.model, args.mode))
    else:
        saving_path = os.path.join(args.model_save_path, '%s_%s_%s.pth'%( args.dataset, args.model, args.mode))
    print("Finish training, save model to %s"%(saving_path))
    th.save(model.state_dict(), saving_path)

    #finish training
    train_acc, train_pred = evaluate(model, train_g, train_g.ndata['features'], train_g.ndata['labels'], train_nid,  args.device)
    print('Final Train Acc {:.4f}'.format(train_acc))
    # print(train_pred[0])

    test_acc, test_pred = evaluate(model, test_g, test_g.ndata['features'], test_g.ndata['labels'], test_nid, args.device)
    print('Final Test Acc {:.4f}'.format(test_acc))
    # print(train_pred[0])

    return train_acc, test_acc    


