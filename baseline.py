import torch
import torch.utils.data as data
from torch import optim
import torchvision.transforms as T
import torchvision.datasets as datasets
import torchvision
import torch.nn as nn
from tqdm import tqdm

# 固定seed
def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

train_root = ''  # 训练集路径
test_root = ''  # 测试集路径
save_pth_root = ''  # 保存模型文件

use_cuda = torch.cuda.is_available()
# 数据增强，考虑加入更多增强模块
transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
#     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    T.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.])
])

def train_one_epoch(model, loader):
    model.train()
    train_loss = 0  # 没用上
    train_acc = 0
    for img, label in loader:
        img, label = img.cuda(), label.cuda()
        optimizer.zero_grad()
        out = model(img)
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        out = out.data.argmax(dim=1, keepdim=True).t().squeeze()  # 取值更大的作为结果，t+squeeze保证shape为(batch,)
        target = label.data
        train_acc += sum(out.eq(target)).item()
    train_acc /= len(train_data)
    
    return train_acc, train_loss


@torch.no_grad()
def validate(model, loader): # 应该叫test
    model.eval()
    test_acc = 0
    for img, label in loader:
        img, label = img.cuda(), label.cuda()
        out = model(img)
        out = out.data.argmax(dim=1, keepdim=True).t().squeeze()  # 取值更大的作为结果，t+squeeze保证shape为(batch,)
        target = label.data
        test_acc += sum(out.eq(target)).item()
    test_acc = test_acc / len(test_data)  
    
    return test_acc


# 定义模型，直接从torchvision导入
model = torchvision.models.resnet50(pretrained=True)
fc = nn.Linear(2048, out_features=2) 
nn.init.xavier_uniform_(fc.weight)  # 初始化，应该还有个bias
model.fc = fc  # 替换最后一层
print(model)
if use_cuda:
    model = model.cuda()
lr = 3e-4
batch_size = 64  # 根据自己的显存调整

epoch = 30
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
criterion = nn.CrossEntropyLoss().cuda() if use_cuda else nn.CrossEntropyLoss()

train_data = datasets.ImageFolder(train_root, transform=transform)  # 要求格式train_root下包含已经按照标签分好类的子文件夹
train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

test_data = datasets.ImageFolder(test_root, transform=transform)
test_loader = data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

fix_seed(0)  # 固定seed
best_acc = 0  # acc为第一条件

for ep in tqdm(range(epoch)):
    train_acc, train_loss = train_one_epoch(model, train_loader)

    print(
        'Train Accuracy : {:.3f}, loss : {:.3f}'.format(train_acc, train_loss))
    test_acc = validate(model, test_loader)

    print('Test Accuracy : {:.4f}'.format(test_acc))
    
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model, save_pth_root)

print('Best Test Accuracy : {:.4f}'.format(best_acc))

