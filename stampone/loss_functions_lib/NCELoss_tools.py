"""
12/12/2022
Author: github/farhadsh1992
INFO:
    Patch sampling and PatchNCE contrastive-loss layers used by
    QS_Attn_Loss to compare cover/encoded feature patches
    (query-selected attention, https://github.com/sapphire497/query-selected-attention).
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
#import torch
from stampone.FarhadCV.Tools import tcolors

import warnings
tf.get_logger().setLevel("INFO")
warnings.filterwarnings('ignore')


class Normalize(tf.keras.layers.Layer):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def call(self, x):
        norm = tf.math.pow(x, self.power)
        #print(norm.shape)
        norm = tf.math.reduce_sum(norm, axis=1, keepdims=True) #???
        #norm = tf.math.sum(norm) #???
        #print(norm.shape)
        norm = tf.math.pow(norm, 1. / self.power)
        out = tf.math.divide(x, norm + 1e-7)
        return out

#<><>>>>>><><><>><><><><><><><><><><><><><><><><><><><><><><><>><><><><><><><><><><><><>><>><><><><><><><><><>><><><><><><><><><><><><>><>><><>
#<><>>>>>><><><>><><><><><><><><><><><><><><><><><><><><><><><>><><><><><><><><><><><><>><>><><><><><><><><><>><><><><><><><><><><><><>><>><><>
#<><>>>>>><><><>><><><><><><><><><><><><><><><><><><><><><><><>><><><><><><><><><><><><>><>><><><><><><><><><>><><><><><><><><><><><><>><>><><>

class PatchSample(tf.keras.layers.Layer):
    def __init__(self,netEF="", n_layers="", use_mlp=False, init_type='normal', init_gain=0.02, nc=256, gpu_ids=[]):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(PatchSample, self).__init__()
        
        self.l2norm = Normalize(2)
        self.use_mlp = use_mlp
        self.nc = nc  # hard-coded
        self.mlp_init = False
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids
        
        
        self.len_ncs = n_layers
        #self.NetExtractFeatures = netEF()
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(self.len_ncs)

    def create_mlp(self, feats):
        for mlp_id in range(feats):
            mlp = tf.keras.layers.Sequential([tf.keras.layers.Dense(self.nc), 
                                               tf.keras.layers.Activation("relu"), 
                                               tf.keras.layers.Dense(self.nc)])
           
            #setattr(self, 'mlp_%d' % mlp_id, mlp)
        init_net(self, self.init_type, self.init_gain, self.gpu_ids)
        self.mlp_init = True

    def apply_cover(self, feats, num_patches=64):
        
        #feats = self.NetExtractFeatures(cover_image, self.nce_layers, encode_only=True)
        
        return_ids = []
        return_feats = []
        return_mats = []
        k_s = 7 # kernel size in unfold
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)
            
        for feat_id, feat in enumerate(feats): #nce_layers

            
            if len(feat.shape) < 4:
                B, W, C = feat.shape[0], feat.shape[1], feat.shape[2]
                H = 1
                feat = tf.expand_dims(feat, axis=1)
                feat_reshape = tf.keras.layers.Reshape((H*W, C))(feat) #V: B*HW*C
            else:
                B, H, W, C = feat.shape[0], feat.shape[1], feat.shape[2], feat.shape[3]
                feat_reshape = tf.keras.layers.Reshape((H*W, C))(feat) #V: B*HW*C

            
            
            if num_patches > 0:
                if feat_id < 3: # all layers instead of last layer
                    patch_id =  tf.experimental.numpy.random.randint(low=0, high=feat_reshape.shape[1], size=feat_reshape.shape[1])  # random id in [0, HW]
             
                    patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]  # .to(patch_ids.device)
                   
                    feat_reshape = tf.keras.layers.Reshape((-1, C))(feat_reshape) #V: B*HW*C
              
                    
                    x_sample1 = feat_reshape.numpy()[:, patch_id, :]#.flatten()  # reshape(-1, x.shape[1])
                    
                    x_sample = tf.reshape(x_sample1, (x_sample1.shape[0]*x_sample1.shape[1], x_sample1.shape[2]))
                    #x_sample = torch.from_numpy(x_sample1).flatten(0, 1).detach().numpy()
                    #x_sample = tf.cast(x_sample, dtype="float32")
                    
                    attn_qs = tf.zeros(1)
                else: # last layer
                    
                    
                    # feat >>> (2, H, w, C)
                    (B, H, w, C) =  feat.shape
                    #feat_local = F.unfold(feat, kernel_size=k_s, stride=1, padding=3)  # (B, ks*ks*C, L)
                    feat_local = tf.image.extract_patches(feat, sizes=[1, k_s, k_s,1], strides=[1,1,1,1], rates=[1,1,1,1], padding="SAME") # (B, L1,l2, ks*ks*C) where l1=l2=L/2
                    
                    
                    L = feat_local.shape[1]*feat_local.shape[2]
                    #feat_k_local = feat_local.permute(0, 2, 1).reshape(B, L, k_s*k_s, C).flatten(0, 1) # (B*L, ks*ks, C)
                    feat_k_local = tf.reshape(feat_local, (B*L, k_s*k_s, C))# (B*L, ks*ks, C)
                    
                    feat_q_local = tf.reshape(feat_reshape, (B*L, C, 1))
                    #feat_q_local = feat_reshape.reshape(B*L, C, 1)
                    
                    #dots_local = torch.bmm(feat_k_local, feat_q_local)  # (B*L, ks*ks, 1)
                    dots_local =  tf.linalg.matmul(feat_k_local, feat_q_local)  # (B*L, ks*ks, 1)
                    
                    attn_local = tf.nn.softmax(dots_local, axis=1, name=None) # (B*L, ks*ks, 1)
                    attn_local = tf.reshape(attn_local, (B, L, -1))  # (B, L, ks*ks)
                    prob = - tf.math.log(attn_local)
                    prob = tf.where(tf.math.is_inf(prob), tf.experimental.numpy.full_like(prob, fill_value=0, dtype=None), prob)
                   
                    entropy = tf.math.reduce_sum(tf.math.multiply(attn_local, prob), axis=2)  # attn_local X prob= (B, L, ks*ks)
                    #sorted_entropy = tf.sort(entropy)
                    index = tf.argsort(entropy) #(B, L)

                    patch_id = index[:, :num_patches] # (B, num_patches) num_patches=64
                    
                    feat_q_global = feat_reshape #V: B*HW*C
                    #feat_k_global = feat_reshape.permute(0, 2, 1)
                    feat_k_global = tf.transpose(feat_reshape, (0, 2, 1))
                    #dots_global = torch.bmm(feat_q_global, feat_k_global)  # (B, HW, HW)
                    dots_global = tf.linalg.matmul(feat_q_global, feat_k_global) # (B, HW, HW)
                    
                    
                    attn_global = tf.nn.softmax(dots_global, axis=2) # or axis=1
                    
                    attn_qs = attn_global.numpy()[tf.range(B)[:, None], patch_id.numpy(), :]
                    feat_reshape = tf.linalg.matmul(attn_qs, feat_reshape) # (B, n_p, C)
                    
                    x_sample = tf.reshape(feat_reshape, (feat_reshape.shape[0]*feat_reshape.shape[1], feat_reshape.shape[2]))
                    #x_sample = torch.from_numpy(feat_reshape.numpy()).flatten(0, 1).detach().numpy()
                    #x_sample = feat_reshape.nump().flatten(0, 1) # ????
                    patch_id = []
            else:
                x_sample = feat_reshape
                patch_id = []
                
            if self.use_mlp:
                mlp = getattr(self, 'mlp_%d' % feat_id)
                x_sample = mlp(x_sample)
                
            return_ids.append(patch_id)
            return_mats.append(attn_qs)
            
            
            x_sample = self.l2norm(x_sample)

            if num_patches == 0:
                x_sample = tf.reshape(x_sample, [B, x_sample.shape[-1], H, W])
                #x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])
                
            return_feats.append(x_sample)
        return return_feats, return_ids, return_mats 
    def apply_encoded(self, feats, num_patches=64, patch_ids=None, attn_mats=None):
        #feats = self.NetExtractFeatures(cover_image, self.nce_layers, encode_only=True)
        
        return_ids = []
        return_feats = []
        return_mats = []
        k_s = 7 # kernel size in unfold
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)
        for feat_id, feat in enumerate(feats): #nce_layers
            
            

            if len(feat.shape) < 4:
                B, W, C = feat.shape[0], feat.shape[1], feat.shape[2]
                H = 1
                feat = tf.expand_dims(feat, axis=1)
                feat_reshape = tf.keras.layers.Reshape((H*W, C))(feat) #V: B*HW*C
            else:
                B, H, W, C = feat.shape[0], feat.shape[1], feat.shape[2], feat.shape[3]
                feat_reshape = tf.keras.layers.Reshape((H*W, C))(feat) #V: B*HW*C

            
            
            if num_patches > 0:
                if feat_id < 3: # all layers instead of last layer
                    patch_id = patch_ids[feat_id]
                    feat_reshape = tf.keras.layers.Reshape((-1, C))(feat_reshape) #V: B*HW*C
                    x_sample1 = feat_reshape.numpy()[:, patch_id, :]#.flatten()  # reshape(-1, x.shape[1])
                    
                    x_sample = tf.reshape(x_sample1, (x_sample1.shape[0]*x_sample1.shape[1], x_sample1.shape[2]))
                    #x_sample = torch.from_numpy(x_sample1).flatten(0, 1).detach().numpy()
                    x_sample = tf.cast(x_sample, dtype="float32")
                    
                    attn_qs = tf.zeros(1)
                else:
                    
                    attn_qs = attn_mats[feat_id]
                    feat_reshape = tf.linalg.matmul(attn_qs, feat_reshape) # (B, n_p, C)
                    
                    x_sample = tf.reshape(feat_reshape, (feat_reshape.shape[0]*feat_reshape.shape[1], feat_reshape.shape[2]))
                    #x_sample = torch.from_numpy(feat_reshape.numpy()).flatten(0, 1).detach().numpy()
                    #x_sample = feat_reshape.nump().flatten(0, 1) # ????
                    patch_id = []
            else:
                x_sample = feat_reshape
                patch_id = []
            if self.use_mlp:
                mlp = getattr(self, 'mlp_%d' % feat_id)
                x_sample = mlp(x_sample)
                
            #return_ids.append(patch_id)
            #return_mats.append(attn_qs)
            
            
            x_sample = self.l2norm(x_sample)

            if num_patches == 0:
                x_sample = tf.reshape(x_sample, [B, x_sample.shape[-1], H, W])
                #x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])
                
            return_feats.append(x_sample)
        return return_feats



#<><>>>>>><><><>><><><><><><><><><><><><><><><><><><><><><><><>><><><><><><><><><><><><>><>><><><><><><><><><>><><><><><><><><><><><><>><>><><>
#<><>>>>>><><><>><><><><><><><><><><><><><><><><><><><><><><><>><><><><><><><><><><><><>><>><><><><><><><><><>><><><><><><><><><><><><>><>><><>
#<><>>>>>><><><>><><><><><><><><><><><><><><><><><><><><><><><>><><><><><><><><><><><><>><>><><><><><><><><><>><><><><><><><><><><><><>><>><><>



# https://github.com/sapphire497/query-selected-attention/blob/main/models/patchnce.py
class PatchNCELoss(tf.keras.layers.Layer):
    """
    L_con
    """
    def __init__(self, opt):
        super(PatchNCELoss, self).__init__()
        
        self.opt = opt # batch_size, 
        self.nce_T = 0.07
        self.cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False,
                                                                         reduction=tf.keras.losses.Reduction.NONE)
        #self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.cross_entropy_loss =  tf.nn.sigmoid_cross_entropy_with_logits #()
        self.cross_entropy_loss =  tf.nn.softmax_cross_entropy_with_logits

       
        
        #self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def call(self, feat_q, feat_k):
        
        batchSize = feat_q.shape[0]
        dim = feat_q.shape[1]  # channel
        feat_k = np.asarray(feat_k)

        # pos logit
        l_pos = tf.linalg.matmul(tf.reshape(feat_q, (batchSize, 1, -1)), tf.reshape(feat_k, (batchSize, -1, 1)))
        l_pos = tf.reshape(l_pos, (batchSize, 1))

        # neg logit -- current batch
        # reshape features to batch size
        feat_q = tf.reshape(feat_q, (self.opt.batch, -1, dim))
        feat_k = tf.reshape(feat_k, (self.opt.batch, -1, dim))
        npatches = feat_q.shape[1]

       
        
        
        feat_k = tf.transpose(feat_k, (0, 2, 1))
        l_neg_curbatch = tf.linalg.matmul(feat_q, feat_k) # b*np*np
       

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = np.eye(npatches,  dtype=np.bool_)[None, :, :]
        #l_neg_curbatch.masked_fill_(diagonal, -10.0)
        
        #print()
        #print(diagonal.shape)
        #print(l_neg_curbatch.shape)
        #diagonal = tf.concat((diagonal, diagonal), axis=0)
        #print(diagonal.shape)
      
        l_neg_curbatch = np.array(l_neg_curbatch, dtype="float32")
 
        #l_neg_curbatch = tf.boolean_mask(l_neg_curbatch,  mask=diagonal, axis=1)
        #l_neg_curbatch = tf.where( tf.math.logical_not(diagonal), l_neg_curbatch, -10.0)
        
        #print(l_neg_curbatch.shape)
      
        l_neg = tf.reshape(l_neg_curbatch, (-1, npatches))
        

        out = tf.concat((l_pos, l_neg), axis=1) / self.nce_T
        
        
        
        loss = self.cross_entropy_loss(out, tf.zeros((out.shape[0], out.shape[1]), dtype=tf.float32))
        #loss = tf.math.reduce_mean(loss)
       
        
        return loss