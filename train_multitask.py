import torch.optim as optim
import time
from sklearn import metrics
from CLIP.clip import clip
from CLIP.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from helper_functions import *
from text_normalizer import *
import argparse
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch

parser = argparse.ArgumentParser(description='Train Multimodal Multi-task model for Misogyny Detection')
parser.add_argument('--bs', type=int, default=64,
                    help='64,128')
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--maxlen', type=int, default=77)
parser.add_argument('--lr', type=str, default='1e-4',
                    help='3e-5, 4e-5, 5e-5, 5e-4')
parser.add_argument('--vmodel', type=str, default='vit14',
                    help='resnet | vit32 | vit16 | vit14 | rn50 | rn101 | rn504 | rn5016 | rn5064')


args = parser.parse_args() 

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




class MMNetwork(nn.Module):
    def __init__(self, vdim, tdim, n_cls):
        super(MMNetwork, self).__init__()

        ## Linear layer for ResNet features
        self.vfc = nn.Linear(vdim, 256)

        ## Single Layer Bi-directional RNN with GRU cells. Projects 768 to 512 
        self.bigru = nn.LSTM(tdim, 256, 1, bidirectional=False, batch_first=True, bias=False)

        ## Concatenated Image and Text goes through this multi-layer network
        self.mfc1 = nn.Linear(512, 256)
        # self.mfc2 = nn.Linear(512, 256)
        # self.mfc3 = nn.Linear(256, 128)
        # self.mfc4 = nn.Linear(256, 128)

        self.cf1 = nn.Linear(256, 1)
        self.cf2 = nn.Linear(256, 1)
        self.cf3 = nn.Linear(256, 1)
        self.cf4 = nn.Linear(256, 1)
        self.cf5 = nn.Linear(256, 1)


        self.act = nn.ReLU()    ## ReLU
        self.vdp = nn.Dropout(0.2)
        self.tdp = nn.Dropout(0.2)

    def forward(self, vx, tx, masks=None):
        # vx = self.vdp(self.vfc(vx))
        vx = self.vdp(self.act(self.vfc(vx)))
        
        _, hidden_tx = self.bigru(tx)
        
        #hidden = [n layers * n directions, batch size, emb dim]
        # hidden_tx = self.tdp(torch.cat((hidden_tx[0][-2,:,:], hidden_tx[0][-1,:,:]), dim = 1))
        ## Concatenate Visual and Textual output
        # mx = torch.cat((vx, hidden_tx), dim=1)
        mx = torch.cat((vx, self.tdp(hidden_tx[0]).squeeze(0)), dim=1)

        mx = self.act(self.mfc1(mx))
        # mx = self.act(self.mfc2(mx))
        # mx = self.relu(self.mfc3(mx))
        # mx = self.relu(self.mfc4(mx))

        return torch.sigmoid(self.cf1(mx)), torch.sigmoid(self.cf2(mx)), torch.sigmoid(self.cf3(mx)), \
                torch.sigmoid(self.cf4(mx)), torch.sigmoid(self.cf5(mx))


_tokenizer = _Tokenizer()
def tokenize(text, context_length: int = 77):

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    # all_tokens = _tokenizer.encode(text)
    tokens = [sot_token] + _tokenizer.encode(text)[:context_length-2] + [eot_token]
    result = torch.zeros(context_length, dtype=torch.long)
    mask = torch.zeros(context_length, dtype=torch.long)
    result[:len(tokens)] = torch.tensor(tokens)
    mask[:len(tokens)] = 1

    return result, mask


def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        # p.grad.data = p.grad.data.float()

def train(model, optimizer, lr_scheduler, num_epochs):

    since = time.time()

    best_model = model
    best_acc = 0.0
    best_val_loss = 100
    best_epoch = 0
    best_f1 = 0

    for epoch in range(1, num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        since2 = time.time()

        model.train()  # Set model to training mode

        running_loss = 0.0
        running_corrects = 0

        tot = 0.0
        cnt = 0
        # Iterate over data.
        for img_inps, txt_tokens, masks, labels1, labels2, labels3, labels4, labels5 in tr_loader:

            img_inps, txt_tokens, masks, labels1, labels2, labels3, labels4, labels5 = img_inps.to(device), \
                txt_tokens.to(device), masks.to(device), labels1.to(device), labels2.to(device), labels3.to(device), \
                labels4.to(device), labels5.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            with torch.no_grad():
                img_feats = clip_model.module.encode_image(img_inps)
                _, txt_feats = clip_model.module.encode_text(txt_tokens)

            outputs1, outputs2, outputs3, outputs4, outputs5 = model(img_feats, txt_feats, masks)
            preds1 = (outputs1>0.5).int()

            loss = criterion(outputs1, labels1.unsqueeze(1).float()) + criterion(outputs2, labels2.unsqueeze(1).float()) + \
                    criterion(outputs3, labels3.unsqueeze(1).float()) + criterion(outputs4, labels4.unsqueeze(1).float()) + \
                    criterion(outputs5, labels5.unsqueeze(1).float())

            # backward + optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # statistics
            running_loss += loss.item()
            running_corrects += torch.sum(preds1 == labels1.data.view_as(preds1)).item()
            tot += len(labels1)

            if cnt % 40 == 0:
                print('[%d, %5d] loss: %.4f, Acc: %.2f' %
                      (epoch, cnt + 1, loss.item(), (100.0 * running_corrects) / tot))

            cnt = cnt + 1

        if scheduler:
            lr_scheduler.step()


        train_loss = running_loss / len(tr_loader)
        train_acc = running_corrects * 1.0 / (len(tr_loader.dataset))

        print('Training Loss: {:.6f} Acc: {:.2f}'.format(train_loss, 100.0 * train_acc))

        test_loss, test_acc, test_f1_1, test_f1_2, test_f1_3, test_f1_4, test_f1_5 = evaluate(model, vl_loader)

        print('Epoch: {:d}, Val Loss: {:.4f}, Acc: {:.2f}, F1_1: {:.2f}, F1_2: {:.2f}, F1_3: {:.2f}, F1_4: {:.2f}, F1_5: {:.2f}'.format(epoch, test_loss,test_acc*100, test_f1_1*100, test_f1_2*100, \
                 test_f1_3*100, test_f1_4*100, test_f1_5*100))


        # deep copy the model
        if test_f1_1 >= best_f1:
            best_acc = test_acc
            best_val_loss = test_loss
            best_model = copy.deepcopy(model)
            best_epoch = epoch
            best_f1 = test_f1_1

    time_elapsed2 = time.time() - since2
    print('Epoch complete in {:.0f}m {:.0f}s'.format(
        time_elapsed2 // 60, time_elapsed2 % 60))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return best_model, best_epoch


def evaluate(model, loader):
    model.eval()
    test_loss = 0
    all_preds1 = []
    all_labels1 = []
    all_preds2 = []
    all_labels2 = []
    all_preds3 = []
    all_labels3 = []
    all_preds4 = []
    all_labels4 = []
    all_preds5 = []
    all_labels5 = []

    with torch.no_grad():
        for img_inps, txt_tokens, masks, labels1, labels2, labels3, labels4, labels5 in loader:

            img_inps, txt_tokens, masks, labels1, labels2, labels3, labels4, labels5 = img_inps.to(device), \
                txt_tokens.to(device), masks.to(device), labels1.to(device), labels2.to(device), labels3.to(device), \
                labels4.to(device), labels5.to(device)

            img_feats = clip_model.module.encode_image(img_inps)
            _, txt_feats = clip_model.module.encode_text(txt_tokens)

            outputs1, outputs2, outputs3, outputs4, outputs5 = model(img_feats, txt_feats, masks)

            preds1 = (outputs1>0.5).int()
            preds2 = (outputs2>0.5).int()
            preds3 = (outputs3>0.5).int()
            preds4 = (outputs4>0.5).int()
            preds5 = (outputs5>0.5).int()
            
            test_loss += (criterion(outputs1, labels1.unsqueeze(1).float()).item() + criterion(outputs2, labels2.unsqueeze(1).float()).item() + \
                            criterion(outputs3, labels3.unsqueeze(1).float()).item() + criterion(outputs4, labels4.unsqueeze(1).float()).item() + \
                            criterion(outputs5, labels5.unsqueeze(1).float()).item())

            all_preds1.extend(preds1.cpu().numpy().flatten())
            all_labels1.extend(labels1.cpu().numpy().flatten())
            all_preds2.extend(preds2.cpu().numpy().flatten())
            all_labels2.extend(labels2.cpu().numpy().flatten())
            all_preds3.extend(preds3.cpu().numpy().flatten())
            all_labels3.extend(labels3.cpu().numpy().flatten())
            all_preds4.extend(preds4.cpu().numpy().flatten())
            all_labels4.extend(labels4.cpu().numpy().flatten())
            all_preds5.extend(preds5.cpu().numpy().flatten())
            all_labels5.extend(labels5.cpu().numpy().flatten())

        acc = metrics.accuracy_score(all_labels1, all_preds1)
        f1_1 = metrics.f1_score(all_labels1, all_preds1, average='macro')
        f1_2 = metrics.f1_score(all_labels2, all_preds2, average='macro')
        f1_3 = metrics.f1_score(all_labels3, all_preds3, average='macro')
        f1_4 = metrics.f1_score(all_labels4, all_preds4, average='macro')
        f1_5 = metrics.f1_score(all_labels5, all_preds5, average='macro')

    return test_loss/len(loader), acc, f1_1, f1_2, f1_3, f1_4, f1_5



## Arguments
batch_size = args.bs
init_lr = float(args.lr)
epochs = args.epochs
vmodel = args.vmodel

## Pre-trained Stream Models
clip_nms = {'vit32':'ViT-B/32', 'vit16':'ViT-B/16', 'rn50':'RN50', 'rn504':'RN50x4', 'rn101':'RN101', 'rn5016':'RN50x16', 'rn5064':'RN50x64', 'vit14':'ViT-L/14'}
clip_dim = {'vit32': 512, 'vit16': 512, 'vit14': 768, 'rn50': 1024, 'rn504': 640, 'rn101': 512, 'rn5016': 768, 'rn5064':1024}
clip_model, _ = clip.load(clip_nms[vmodel],jit=False)
input_resolution = clip_model.visual.input_resolution
clip_model.float().eval()
clip_model = nn.DataParallel(clip_model)

dim = clip_dim[vmodel]


## Transforms
transform_config = {'train': transforms.Compose([
        transforms.RandomResizedCrop(input_resolution, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.2),
        transforms.RandomGrayscale(p=0.1),
        # transforms.RandomPerspective(),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                    std=[0.26862954, 0.26130258, 0.27577711])
    ]), 
    'test': transforms.Compose([
        transforms.Resize((input_resolution,input_resolution), interpolation=transforms.InterpolationMode.BICUBIC),
        # transforms.CenterCrop(clip_model.visual.input_resolution),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                    std=[0.26862954, 0.26130258, 0.27577711])
    ])
}

## Dataset
tr_df = pd.read_csv('data/train.csv', sep='\t')
vl_df = pd.read_csv('data/validation.csv', sep='\t')

## Find max length in training text
# max_length = 0
# for txt in tr_df['text']:
#     input_ids = tokenize(preprocess(txt))
#     max_length = max(max_length, len(input_ids))

# print("Maximum number of tokens:%d"%(max_length))
## -------------------------------

if args.maxlen != 0:
    max_length = args.maxlen

tr_data = CustomDatasetFixed(tr_df, 'training', transform_config['test'], preprocess, tokenize, max_length)
vl_data = CustomDatasetFixed(vl_df, 'training', transform_config['test'], preprocess, tokenize, max_length)
ts_data = CustomDatasetFixed(vl_df, 'test', transform_config['test'], preprocess, tokenize, max_length)
tr_loader = DataLoader(tr_data, shuffle=True, num_workers=4, batch_size=batch_size)
vl_loader = DataLoader(vl_data, num_workers=2, batch_size=batch_size)
ts_loader = DataLoader(ts_data, num_workers=2, batch_size=batch_size)


## Model
model = MMNetwork(dim, dim, 1)

model.to(device)
print(model)

optimizer = optim.Adam(model.parameters(), init_lr, betas=(0.99,0.98), weight_decay=1e-4)
criterion = nn.BCELoss()

num_train_steps = int(len(tr_data)/batch_size)*epochs
num_warmup_steps = int(0.1*num_train_steps)
print(num_train_steps, num_warmup_steps)    ## Print Number of total and warmup steps
# scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [5,10,15],gamma=0.5)
# scheduler = None

model_ft, best_epoch = train(model, optimizer, scheduler, num_epochs=epochs)

torch.save(model_ft.state_dict(), 'saved_models/trained_model_%s.pt'%(args.net, vmodel))

vl_loss, vl_acc, vl_f1, _, _, _, _ = evaluate(model_ft, vl_loader)
print('Validation best epoch: %d, Val Loss: %.4f, ACC: %.2f, F1: %.2f'%(best_epoch, np.round(vl_loss,4), vl_acc*100, vl_f1*100))

ts_loss, ts_acc, ts_f1, _, _, _, _ = evaluate(model_ft, ts_loader)
print('Test results:, Test Loss: %.4f, ACC: %.2f, F1: %.2f'%(np.round(ts_loss,4), ts_acc*100, ts_f1*100))
