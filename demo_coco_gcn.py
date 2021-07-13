import argparse
from engine import *
from models import *
from coco import *
from util import *


parser = argparse.ArgumentParser(description='WILDCAT Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset (e.g. data/')
parser.add_argument('--image-size', '-i', default=448, type=int,
                    metavar='N', help='image size (default: 224)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch_step', default=[30], type=int, nargs='+',
                    help='number of epochs to change learning rate')
parser.add_argument('--device_ids', default=[0], type=int, nargs='+',
                    help='number of epochs to change learning rate')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lrp', '--learning-rate-pretrained', default=0.1, type=float,
                    metavar='LR', help='learning rate for pre-trained layers')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=0, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--vocab_path', type=str, default='./data/dic_thyroid_gyt_20210629.json',
                    help='path for vocabulary wrapper')
parser.add_argument('--embed_size', type=int , default=256,
                    help='dimension of word embedding vectors')
parser.add_argument('--hidden_size', type=int , default=512,
                    help='dimension of lstm hidden states')
parser.add_argument('--num_layers', type=int , default=1,
                    help='number of layers in lstm')

def main_coco():
    global args, best_prec1, use_gpu
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()
    with open(args.vocab_path, 'rb') as f:
        vocab = json.load(f)['word2ix']

    train_dataset = Thyroid(args.data, phase='train',vocab=vocab)
    val_dataset = Thyroid(args.data, phase='val',vocab=vocab)
    num_classes = 27
    # gcn_resnet101(num_classes, embed_size, hidden_size, vocab, num_layers, pretrained=False, in_channel=256):
    model = gcn_resnet101(num_classes=num_classes, embed_size=args.embed_size, hidden_size=args.hidden_size, vocab=vocab, num_layers=args.num_layers, in_channel=args.embed_size)

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss()

    # define optimizer
    optimizer = torch.optim.SGD(model.get_config_optim(args.lr, args.lrp),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    state = {'batch_size': args.batch_size, 'image_size': args.image_size, 'max_epochs': args.epochs,
             'evaluate': args.evaluate, 'resume': args.resume, 'num_classes':num_classes}
    state['difficult_examples'] = True
    state['save_model_path'] = 'checkpoint/thyroid/'
    state['workers'] = args.workers
    state['epoch_step'] = args.epoch_step
    state['lr'] = args.lr
    state['vocab_size'] = len(vocab)
    state['embed_size'] = 256
    # state['device_ids'] = args.device_ids
    if args.evaluate:
        state['evaluate'] = True
    engine = GCNMultiLabelMAPEngine(state)
    engine.learning(model, criterion, train_dataset, val_dataset, optimizer)

if __name__ == '__main__':
    main_coco()
