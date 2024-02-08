# import librosa
# import torch
# from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
#
# input_audio, sample_rate = librosa.load("a.wav",  sr=16000)
#
# model_name = "facebook/wav2vec2-base"
# feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
# model = Wav2Vec2Model.from_pretrained(model_name)
#
# i= feature_extractor(input_audio, return_tensors="pt", sampling_rate=sample_rate)
# with torch.no_grad():
#   o= model(i.input_values)
#
# print(o.last_hidden_state.shape)
# print(o.extract_features.shape)
# import torch
# from towhee import pipeline
# audio_path = 'file_example_WAV_2MG.wav'
# embedding_pipeline = pipeline('towhee/audio-embedding-wav2vec2-base')#-base
# embedding = embedding_pipeline(audio_path)
# audio_emb = torch.FloatTensor(embedding)
# print(type(audio_emb))
# print(audio_emb.shape)
import librosa
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

input_audio, sample_rate = librosa.load("300_AUDIO.wav",  sr=16000)#16000 was used for training of wav2vec

model_name = "facebook/wav2vec2-base"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
model = Wav2Vec2Model.from_pretrained(model_name)

i= feature_extractor(input_audio, return_tensors="pt", sampling_rate=sample_rate)
with torch.no_grad():
  o= model(i.input_values)
print(o.keys())
last_hidden_states=o.last_hidden_state.data.resize_(3,2,128)
print(last_hidden_states)
# extracted_feayures=o.extract_features.data.resize_(3,2,128)
# print(o.extract_features)


































import pandas as pd
import datetime

e = datetime.datetime.now()
count=0
print ("Current date and time = %s" % e)

print ("Today's date:  = %s/%s/%s" % (e.day, e.month, e.year))

print ("The time is now: = %s:%s:%s" % (e.hour, e.minute, e.second))
from PIL import Image
from torchvision.transforms import ToTensor
import torch
import torch.nn as nn
#from torch.autograd._functions import Resize
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt

from video_swin_transformer import SwinTransformer3D
from transformers import BertTokenizer, BertModel
from torchvision import transforms
import torch
import random
from typing import Dict, Optional
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
correct=0
wrong=0
from torch import Tensor
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
class CustomDataset(Dataset):
    def __init__(self,directory):
        self.imgs_path=directory
        #self.imgs_path = "/media/data1/jian/I-CONECT/crafts_hobbies/"
        file_list = glob.glob(self.imgs_path + "*")

        self.data = []
        for class_path in file_list:
            class_name = class_path.split("/")[-1]

            for img_path in glob.glob(class_path + "/*.jpg"):


                self.data.append([img_path, class_name])

        #self.class_map = {"dogs": 0, "cats": 1}
        self.img_dim = (224, 224)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = Image.open(img_path)
        transform = transforms.Resize((224, 224))
        out = transform(img)
        # print("the formated size is :",out.size)
        image = ToTensor()(out).unsqueeze(1)
        # print(image.shape)
        dummy_x = image
        return dummy_x  # , class_id

'''
initialize a SwinTransformer3D model for visual embeddings
'''
model = SwinTransformer3D()
model.cuda()
#print(model)

#init
img = Image.open("C1007_9999.jpg")
image = ToTensor()(img).unsqueeze(0)
dummy_x = image[:, :, :, :, None]
#dummy_x = torch.rand(1, 3, 15,224, 224)
logits = model(dummy_x.cuda())
#print(logits.shape)

from mmcv import Config, DictAction
from mmaction.models import build_model
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

config = './configs/recognition/swin/swin_base_patch244_window1677_sthv2.py'
checkpoint = './checkpoints/swin_base_patch244_window1677_sthv2.pth'

cfg = Config.fromfile(config)
model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
model=model.cuda()
load_checkpoint(model, checkpoint, map_location='cpu')

'''
use the pretrained SwinTransformer3D as feature extractor
'''
def create_src_lengths_mask(batch_size: int, src_length: Tensor, max_src_length:Optional[int] = None):

    if max_src_length is None:
        max_src_length=int(src_length.max())
    src_indices=torch.arange(0,max_src_length).unsqueeze(0).type_as(src_length)
    src_indices=src_indices.expand(batch_size,max_src_length)
    src_length=src_length.unsqueeze(dim=1).expand(batch_size,max_src_length)
    return (src_indices<src_length).int().detach()
def masked_softmax(scores,src_length,src_length_masking=True):
    src_length_copy=src_length
    if src_length_masking:
        bsz, src_length=scores.size()
        src_mask=create_src_lengths_mask(bsz,src_length_copy)
        scores=scores.masked_fill(src_mask==0,-np.inf)
    return F.softmax(scores.float(),dim=-1).type_as(scores)
class ParallelCoAttentionNetwork(nn.Module):
    def __init__(self, hidden_dim, co_attention_dim, src_length_masking=True):
        super(ParallelCoAttentionNetwork, self).__init__()
        self.hidden_dim=hidden_dim

        self.co_attention_dim=co_attention_dim
        self.src_length_masking=src_length_masking
        self.Wb = nn.Parameter(torch.randn(self.hidden_dim,self.hidden_dim))
        self.W_v=nn.Parameter(torch.randn(self.co_attention_dim,self.hidden_dim))
        self.W_q = nn.Parameter(torch.randn(self.co_attention_dim, self.hidden_dim))
        self.W_hv = nn.Parameter(torch.randn(self.co_attention_dim, 1))
        self.W_hq = nn.Parameter(torch.randn(self.co_attention_dim, 1))
    def forward(self,V,Q,Q_lengths):
        C = torch.matmul(Q.cuda(), torch.matmul(self.Wb.cuda(), V.cuda()))
        H_v = nn.Tanh()(torch.matmul(self.W_v.cuda(), V.cuda()) + torch.matmul(
            torch.matmul(self.W_q.cuda(), Q.cuda().permute(0, 2, 1)), C.cuda()))
        H_q = nn.Tanh()(torch.matmul(self.W_q.cuda(), Q.cuda().permute(0, 2, 1)) + torch.matmul(
            torch.matmul(self.W_v.cuda(), V.cuda()), C.cuda().permute(0, 2, 1)))
        a_v = F.softmax(torch.matmul(torch.t(self.W_hv.cuda()), H_v.cuda()), dim=2)
        a_q = F.softmax(torch.matmul(torch.t(self.W_hq.cuda()), H_q.cuda()), dim=2)

        masked_a_q = masked_softmax(a_q.cuda().squeeze(1), Q_lengths.cuda(), self.src_length_masking).unsqueeze(1)
        v = torch.squeeze(torch.matmul(a_v.cuda(), V.cuda().permute(0, 2, 1)))
        q = torch.squeeze(torch.matmul(masked_a_q.cuda(), Q.cuda()))
        return a_v, masked_a_q, v, q


# [batch_size, channel, temporal_dim, height, width]
def vision_embedding(directory):
    dataset = CustomDataset(directory)
    data_loader = DataLoader(dataset, batch_size=10, shuffle=True)
    for imgs in data_loader:
            # #print(f)
            # img = Image.open(f)
            # transform = transforms.Resize((224, 224))
            # out = transform(img)
            # #print("the formated size is :",out.size)
            # image = ToTensor()(out).unsqueeze(0)
            # #print(image.shape)
            # dummy_x = image[:, :, None,:, :]
            # print(dummy_x.shape)
            dummy_x=imgs
            #dummy_x = torch.rand(1, 3, 15,224, 224)

            # SwinTransformer3D without cls_head
            backbone = model.backbone

            # [batch_size, hidden_dim, temporal_dim/2, height/32, width/32]
            feat = backbone(dummy_x.cuda())
            # alternative way
            feat = model.extract_feat(dummy_x.cuda())
            # mean pooling
            feat = feat.mean(dim=[2,3,4]) # [batch_size, hidden_dim]

            # project
            batch_size, hidden_dim = feat.shape
            feat_dim = 768
            proj = nn.Parameter(torch.randn(hidden_dim, feat_dim))

        # final output
    output = feat.cuda() @ proj.cuda() # [batch_size, feat_dim]
    #Resize.apply(output,(3,128,2))
    output=output.data.resize_(3, 128, 2)
    #print(output.shape)
    return output

def word_embedding_with_tokens(directory):
    import os
    model_name = 'bert-base-uncased'

    tokenizer = BertTokenizer.from_pretrained(model_name)
    # load
    model = BertModel.from_pretrained(model_name)
    model=model.cuda()
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            my_file = open(f, "r")
            data = my_file.read()
            # print(data)
            # replacing end splitting the text
            # when newline ('\n') is seen.
            data_into_list = data.split("\n")
            # print(data_into_list)
            my_file.close()
            sentences=data_into_list
            for item in sentences:
                input_text = item
                # tokenizer-> token_id
                input_ids = tokenizer.encode(input_text, add_special_tokens=True)

                input_ids = torch.tensor([input_ids])

                with torch.no_grad():
                    last_hidden_states = model(input_ids.cuda())[0]  # Models outputs are now tuples
                last_hidden_states = last_hidden_states.mean(1)
    #print("word emb")
    last_hidden_states=last_hidden_states.data.resize_(3,2,128)
    #Resize.apply(last_hidden_states, (3, 2, 128))
    return last_hidden_states #word embedding of sentneces of each subject

            #print(data_into_list)
            #embeddings = lang_model.encode(sentences)
    #         for item in sentences:
    #             tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #             marked_text = "[CLS] " + item + " [SEP]"
    #             tokenized_text = tokenizer.tokenize(marked_text)
    #             indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    #             segments_ids = [1] * len(tokenized_text)
    #             tokens_tensor = torch.tensor([indexed_tokens])
    #             segments_tensors = torch.tensor([segments_ids])
    #             model = BertModel.from_pretrained('bert-base-uncased',
    #                                               output_hidden_states=True,  # Whether the model returns all hidden-states.
    #                                               )
    #             model.cuda()
    #             with torch.no_grad():
    #                 outputs = model(tokens_tensor.cuda(), segments_tensors.cuda())
    #                 hidden_states = outputs[2]
    #             token_embeddings = torch.stack(hidden_states, dim=0)
    #             token_embeddings.size()
    #             token_embeddings = torch.squeeze(token_embeddings, dim=1)
    # print(token_embeddings)
    # print(token_embeddings.shape)
    #return token_embeddings
import torch.nn.functional as nnf
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(768,6)
        self.fc2 = nn.Linear(6, 1)
        #self.fc3=nn.Softmax(1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
import os
craft_hobbies_sub=["C1007","C1017","C1019","C1032","C1033","C1034","C1039","C1042","C1046","C1054","C1066","C1083","C1097","C1106","C1125","C1127","C1133","C1142","C1144","C1159","C1166","C1168","C1170","C1179","C1184","C1207","C1224","C1225","C1234","C1238","C1243","C2039","C2040","C2046"]
df_trans=pd.read_csv('/media/data1/farida1/t/Schoolsubject/school_fold0.csv')
dfy=pd.read_csv('/media/data1/farida1/t/craft-hobbies/subjects_basic_info.csv')
for k in range(10):
    test_sub_y = df_trans[df_trans.fold == k].iloc[:, 0].tolist()
    df_test_y = dfy[dfy['subject_id'].isin(test_sub_y)]

    train_sub_y = df_trans[df_trans.fold != k].iloc[:, 0].tolist()
    df_train_y = dfy[dfy['subject_id'].isin(train_sub_y)]
    # print(df_test_y)
    # print(df_train_y)
    test_sub_x = df_trans[df_trans.fold == k].iloc[:, 0].tolist()
    df_test_x = (test_sub_x)
    train_sub_x = df_trans[df_trans.fold != k].iloc[:, 0].tolist()
    df_train_x = (train_sub_x)
    for i in range(len(df_train_x)):
        directory = "/media/data1/jian/I-CONECT/school_subjects/%s" % df_train_x[i]
        last_hidden_states_v=vision_embedding(directory)
        directory = "/media/data1/farida1/t/Schoolsubject/%s"% df_train_x[i]
        last_hidden_states_w=word_embedding_with_tokens(directory)
        pcan = ParallelCoAttentionNetwork(128, 3)
        v = torch.randn((3, 2, 128))  # visual embeddings
        # q = torch.randn(3, 2, 128)  # word embeddings
        v = last_hidden_states_v
        q = last_hidden_states_w
        a = random.sample(range(1, 2), 1)
        q_lens = torch.LongTensor(a)
        a_v, a_q, v, q = pcan(v.cuda(), q.cuda(), q_lens.cuda())
        # v=torch.squeeze(v,dim=1)
        # q=torch.squeeze(q,dim=1)
        joint_rep = torch.cat((v, q))
        joint_rep = joint_rep.reshape(1, 768)
        # q_lens=torch.LongTensor([5,1,2,3,4,6,7]) #lengths of words' embedding

        # print("joint")
        # print(joint_rep)
        import numpy

        FCmodel = MyModel().cuda()
        optimizer = torch.optim.Adam(FCmodel.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()

        data = joint_rep
        # print(data.shape)
        # data = torch.randn(1, 768)

        dfy = pd.read_csv("/media/data1/farida1/t/craft-hobbies/subjects_basic_info.csv",
                          usecols=['subject_id', 'normcog'])
        dfy=dfy.loc[dfy.subject_id.isin([df_train_x[i]])]
        labels = dfy['normcog'].values.tolist()
        labels_int = [int(j) for j in labels]
        t = torch.tensor(labels_int)
        target = t.reshape(1, 1).float()
        # dfy = dfy.loc[dfy.subject_id.isin(
        #     ["C1007", "C1018", "C1019", "C1033", "C1034", "C1039", "C1046", "C1054", "C1070", "C1083", "C1103", "C1104",
        #      "C1113", "C1124", "C1127", "C1133", "C1156", "C1159", "C1162", "C1166", "C1168", "C1170", "C1179", "C1184",
        #      "C1202", "C1229", "C1238", "C2040", "C2046"])]
        # labels = dfy['normcog'].values.tolist()
        # labels_int = [int(j) for j in labels]
        #
        # t = torch.tensor(labels_int)
        # if count==29:
        #     count=25
        #     print("cnt=25")
        # target = t[count].reshape(1, 1).float()
        # # target = torch.randint(0, 2, (1, 1)).float()
        # count=count+1

        for epoch in range(100):
            optimizer.zero_grad()
            output = FCmodel(data.cuda())
            loss = criterion(output.cuda(), target.cuda())
            loss.backward(retain_graph=True)
            optimizer.step()
    for i in range(len(df_test_x)):
        directory = "/media/data1/jian/I-CONECT/school_subjects/%s" % df_test_x[i]
        last_hidden_states_v=vision_embedding(directory)
        directory = "/media/data1/farida1/t/Schoolsubject/%s"% df_test_x[i]
        last_hidden_states_w=word_embedding_with_tokens(directory)
        pcan = ParallelCoAttentionNetwork(128, 3)
        #v=torch.randn((3,2,128)) #visual embeddings
        #q = torch.randn(3, 2, 128)  # word embeddings
        v = last_hidden_states_v
        q=last_hidden_states_w

        a = random.sample(range(1, 2), 1)
        q_lens = torch.LongTensor(a)
        a_v, a_q, v, q = pcan(v.cuda(), q.cuda(), q_lens.cuda())
        # v=torch.squeeze(v,dim=1)
        # q=torch.squeeze(q,dim=1)
        joint_rep = torch.cat((v, q))
        joint_rep = joint_rep.reshape(1, 768)
        # q_lens=torch.LongTensor([5,1,2,3,4,6,7]) #lengths of words' embedding

        #print("joint")
        # print(joint_rep)
        import numpy
        data = joint_rep
        #print(data.shape)
        # data = torch.randn(1, 768)

        dfy = pd.read_csv("/media/data1/farida1/t/craft-hobbies/subjects_basic_info.csv", usecols=['subject_id', 'normcog'])

        df_temp = dfy.loc[dfy.subject_id.isin([df_test_x[i]])]

        output = FCmodel(data.cuda())
        probs = torch.sigmoid(output.cuda())
        #comment the training
        pred=0
        if probs>0.5:
            pred=1
        else:
            pred=0
        print("info: ")
        print(df_test_x[i])
        print(probs)
        labels = df_temp['normcog'].values.tolist()
        labels_int = [int(j) for j in labels]
        print(labels_int)
import datetime

e = datetime.datetime.now()

print ("Current date and time = %s" % e)

print ("Today's date:  = %s/%s/%s" % (e.day, e.month, e.year))

print ("The time is now: = %s:%s:%s" % (e.hour, e.minute, e.second))
# import torch
# import torch.nn as nn
# from video_swin_transformer import SwinTransformer3D
# from PIL import Image
# from torchvision.transforms import ToTensor
# from transformers import BertTokenizer, BertModel
# #lang_model = SentenceTransformer('bert-base-uncased')
# from PIL import Image
# import matplotlib.pyplot as plt
# import torchvision.transforms as transforms
# import os
# model = SwinTransformer3D()
#
# model=model.cuda()
# #print(model)
#
# # image = Image.open("C1007_9998.jpg")
# #
# #
# # transform = transforms.Compose([
# #                     transforms.PILToTensor()
# #                 ])
# #
# #
# # img_tensor = transform(image)
# # c=img_tensor.unsqueeze(-4)
# # dummy_x=c.unsqueeze(3)
# # dummy_x=dummy_x.type('torch.FloatTensor')
# #
# # print("image size: ")
# # print(dummy_x.shape)
# #
# # print(dummy_x.shape)
# dummy_x = torch.rand(1, 3, 32,224, 224)
# dummy_x=dummy_x.to("cuda")
# logits = model(dummy_x)
# print(logits.shape)
# from transformers import AutoImageProcessor, SwinModel
# from datasets import load_dataset
# from mmcv import Config, DictAction
# from mmaction.models import build_model
# from mmcv.runner import get_dist_info, init_dist, load_checkpoint
#
#
# config = './configs/recognition/swin/swin_base_patch244_window1677_sthv2.py'
# checkpoint = './checkpoints/swin_base_patch244_window1677_sthv2.pth'
#
# cfg = Config.fromfile(config)
# model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
# model=model.cuda()
# load_checkpoint(model, checkpoint, map_location='cpu')
#
# def img_embedding():
#
#     import os
#
#
#                 # image = Image.open("C1007_999%d.jpg" % (i+8))
#                 #
#                 # # Read a PIL image
#                 #
#                 # # Define a transform to convert PIL
#                 # # image to a Torch tensor
#                 # transform = transforms.Compose([
#                 #     transforms.PILToTensor()
#                 # ])
#                 #
#                 # # transform = transforms.PILToTensor()
#                 # # Convert the PIL image to Torch tensor
#                 # img_tensor = transform(image)
#                 # c=img_tensor.unsqueeze(-4)
#                 # dummy_x=c.unsqueeze(3)
#                 # #dummy_x = img_tensor[:, :, :, None,None]
#                 # dummy_x=dummy_x.type('torch.FloatTensor')
#                 # print(dummy_x.shape)
#     dummy_x = torch.rand(1, 3, 32,224, 224)
#     dummy_x=dummy_x.to("cuda")
#     # SwinTransformer3D without cls_head
#     backbone = model.backbone
#
#     # [batch_size, hidden_dim, temporal_dim/2, height/32, width/32]
#     feat = backbone(dummy_x)
#
#     # alternative way
#
#     feat = model.extract_feat(dummy_x)
#
#                 # mean pooling
#     feat = feat.mean(dim=[2,3,4]) # [batch_size, hidden_dim]
#
#     # project
#     batch_size, hidden_dim = feat.shape
#     feat_dim = 512
#     proj = nn.Parameter(torch.randn(hidden_dim, feat_dim))
#                 # model = model.to('cpu')
#                 # gc.collect()
#                 # torch.cuda.empty_cache()
#             # final output
#     output = feat @ proj # [batch_size, feat_dim]
#     #print(output) #vision embeddings based on swin video
#     return output
#
# def word_embedding_with_tokens():
#
#     my_file = open("C1018p.txt", "r")
#     data = my_file.read()
#     print(data)
#     # replacing end splitting the text
#     # when newline ('\n') is seen.
#     data_into_list = data.split("\n")
#     print(data_into_list)
#     my_file.close()
#     sentences=data_into_list
#     #print(data_into_list)
#     #embeddings = lang_model.encode(sentences)
#     for item in sentences:
#         tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#         marked_text = "[CLS] " + item + " [SEP]"
#         tokenized_text = tokenizer.tokenize(marked_text)
#         indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
#         segments_ids = [1] * len(tokenized_text)
#         tokens_tensor = torch.tensor([indexed_tokens])
#         segments_tensors = torch.tensor([segments_ids])
#         model = BertModel.from_pretrained('bert-base-uncased',
#                                           output_hidden_states=True,  # Whether the model returns all hidden-states.
#                                           )
#         with torch.no_grad():
#             outputs = model(tokens_tensor, segments_tensors)
#             hidden_states = outputs[2]
#         token_embeddings = torch.stack(hidden_states, dim=0)
#         token_embeddings.size()
#         token_embeddings = torch.squeeze(token_embeddings, dim=1)
#     print(token_embeddings)
#     print(token_embeddings.shape)
# # def sentence_embeddings():
# #     my_file = open("C1018p.txt", "r")
# #     data = my_file.read()
# #     print(data)
# #     # replacing end splitting the text
# #     # when newline ('\n') is seen.
# #     data_into_list = data.split("\n")
# #     print(data_into_list)
# #     my_file.close()
# #     sentences = data_into_list
# #     # print(data_into_list)
# #     # embeddings = lang_model.encode(sentences)
# #
# #     embeddings = lang_model.encode(sentences)
# #     print(embeddings)
# #     print(embeddings.shape)
# #txt=word_embedding_with_tokens()
# # def samedim(img_embedding,word_embedding):
# # #pass to FC layer to have smae dim
# # #return img,txt
# # def co_attention(img,txt):
# # #implement co-attention to fuse features and pass to fc layer
# #print(img_embedding())
# print(word_embedding_with_tokens())

# import torch
#
# from typing import Dict,Optional
# import numpy as np
# import torch.nn as nn
# import torch.nn.functional as F
#
# from torch import Tensor
#
#
# def create_src_lengths_mask(batch_size: int, src_length: Tensor, max_src_length:Optional[int] = None):
#
#     if max_src_length is None:
#         max_src_length=int(src_length.max())
#     src_indices=torch.arange(0,max_src_length).unsqueeze(0).type_as(src_length)
#     src_indices=src_indices.expand(batch_size,max_src_length)
#     src_length=src_length.unsqueeze(dim=0).expand(batch_size,max_src_length)
#     return (src_indices<src_length).int().detach()
# def masked_softmax(scores,src_length,src_length_masking=True):
#     src_length_copy=src_length
#     if src_length_masking:
#         bsz, src_length=scores.size()
#         src_mask=create_src_lengths_mask(bsz,src_length_copy)
#         scores=scores.masked_fill(src_mask==0,-np.inf)
#     return F.softmax(scores.float(),dim=-1).type_as(scores)
# class ParallelCoAttentionNetwork(nn.Module):
#     def __init__(self, hidden_dim, co_attention_dim, src_length_masking=True):
#         super(ParallelCoAttentionNetwork, self).__init__()
#         self.hidden_dim=hidden_dim
#         print(self.hidden_dim)
#         self.co_attention_dim=co_attention_dim
#         self.src_length_masking=src_length_masking
#         self.Wb = nn.Parameter(torch.randn(self.hidden_dim,self.hidden_dim))
#         self.W_v=nn.Parameter(torch.randn(self.co_attention_dim,self.hidden_dim))
#         self.W_q = nn.Parameter(torch.randn(self.co_attention_dim, self.hidden_dim))
#         self.W_hv = nn.Parameter(torch.randn(self.co_attention_dim, 1))
#         self.W_hq = nn.Parameter(torch.randn(self.co_attention_dim, 1))
#     def forward(self,V,Q,Q_lengths):
#         C=torch.matmul(Q,torch.matmul(self.Wb,V))
#         H_v=nn.Tanh()(torch.matmul(self.W_v,V)+torch.matmul(torch.matmul(self.W_q,Q.permute(0,2,1)),C))
#         H_q=nn.Tanh()(torch.matmul(self.W_q,Q.permute(0,2,1))+torch.matmul(torch.matmul(self.W_v,V),C.permute(0,2,1)))
#         a_v=F.softmax(torch.matmul(torch.t(self.W_hv),H_v),dim=2)
#         a_q = F.softmax(torch.matmul(torch.t(self.W_hq), H_q), dim=2)
#
#         masked_a_q=masked_softmax(a_q.squeeze(1),Q_lengths,self.src_length_masking).unsqueeze(-1)
#         v=torch.squeeze(torch.matmul(a_v,V.permute(0,2,1)))
#         q=torch.squeeze(torch.matmul(masked_a_q,Q))
#         return a_v,masked_a_q,v,q
# pcan=ParallelCoAttentionNetwork(768,768)
# # v=torch.randn((5,6,10)) #visual embeddings
# # q=torch.randn(5,8,6) #word embeddings
# v=torch.randn((1,768,1)) #visual embeddings
# q=torch.randn(1,1,768) #word embeddings
# # v=last_hidden_states
# # #v=v[:,:,None]
# # q=last_hidden_states
# # #q=q[:,None,:]
# #
# # print(v.shape)
# # print(q.shape)
# #q_lens=torch.LongTensor([3,4,5,8,2]) #lengths of words' embedding
# q_lens=torch.LongTensor([3,2,1,4,5,6,7,8,9,10,11,12,13])
# a_v,a_q,v,q=pcan(v,q,q_lens)
# print(a_v)
# print(a_q)
# # joint_rep=torch.cat(v,a_v)
# # print(joint_rep.shape)
