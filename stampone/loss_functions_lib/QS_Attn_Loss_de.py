"""
12/12/2022
Author: github/farhadsh1992
INFO:
    Query-selected-attention NCE loss (decoder side): same PatchNCE
    machinery as QS_Attn_Loss.py but compares the recovered message
    against the original message instead of cover vs. encoded image.
"""



import tensorflow as tf
from stampone.loss_functions_lib.NCELoss_tools import PatchSample
from stampone.loss_functions_lib.NCELoss_tools import PatchNCELoss
import numpy as np
from stampone.FarhadCV.Tools import tcolors



import warnings
tf.get_logger().setLevel("DEBUG")
warnings.filterwarnings('ignore')



def Extract_Features_for_Vnet(Net):
  """ """
  list_name_down_sample_layers = ["concatenate_{}".format(i) for i in range(1, 8)]
  # outs = [Net.get_layer(name=name).output for name in list_name_down_sample_layers]
  # outs.append(Net.outputs[0])
  # outs = list(reversed(outs[:]))

  outs = [Net.outputs[0]]

  features_extraction_net = tf.keras.models.Model(inputs=Net.inputs,
                                                outputs=outs)
  return features_extraction_net

def Extract_Features_for_SwinUnet(Net):
    """ """
    #list_name_down_sample_layers = ["swin_unet_concat_{}".format(i) for i in range(0, 3)]
    #list_name_down_sample_layers = ["swin_transformer_block_{}".format(i) for i in range(1, 7)]

    #for layer in Net.layers:
    #    print(layer.name)
    # list_name_down_sample_layers = ["swin_transformer_block_{}".format(i) for i in range(14, 22)]
    # outs = [Net.get_layer(name=name).output for name in list_name_down_sample_layers]
    # outs.append(Net.outputs[0])
    # outs = list(reversed(outs[:]))

    outs = [Net.outputs[0]]

    features_extraction_net = tf.keras.models.Model(
                                   inputs  = Net.inputs,
                                   outputs = outs,
                                )
    return features_extraction_net

def Extract_Features_for_VGG(Net):
    """ """
    list_name_down_sample_layers = ["block5_conv4","block5_conv3", "block4_conv4","block4_conv3"]

    outs = [Net.outputs[0]]
    for name in list_name_down_sample_layers:
        outs.append(Net.get_layer(name=name).output)


    features_extraction_net = tf.keras.models.Model(
        inputs=Net.inputs,
        outputs=outs,
    )
    return features_extraction_net
  
  
class QS_Attn_Loss_de(tf.keras.layers.Layer):
    def __init__(self, args, net="", n_layers="", kind_net=""):
        super(QS_Attn_Loss_de, self).__init__()
        
        self.args = args
        if kind_net == "vnet":
            self.featureExtractor = Extract_Features_for_Vnet(net)

        elif kind_net == "swinunet":
            self.featureExtractor = Extract_Features_for_SwinUnet(net)
        elif kind_net == "vgg":
            # net = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
            self.featureExtractor = Extract_Features_for_VGG(net)


        
        self.num_patches = 64 #self.args.num_patches #64 
        self.lambda_NCE = 1.0 #self.args.lambda_NCE #1.0
        
        self.n_layers = n_layers
        
        self.PS_router = PatchSample(netEF="", 
                                     n_layers=n_layers, 
                                     use_mlp=False, 
                                     init_type='normal', 
                                     init_gain=0.02, 
                                     nc=256, 
                                     gpu_ids=[])
        
        # self.criterionNCE = []
        # for nce_layer in self.nce_layers:
        #     self.criterionNCE.append(PatchNCELoss(opt).to(self.device))
        
        self.nceloss_router = PatchNCELoss(self.args) #  batch_size = 2 , nce_T = 0.07
        
    def call(self, recovered, message):
        
        #mzeors = np.zeros(message.shape, dtype="float32")
        message_re = tf.image.resize(message, (256, 256))
        recovered_re = tf.image.resize(recovered, (256, 256))

        feat_k = self.featureExtractor([message_re])
        feat_q = self.featureExtractor([recovered_re])
       
        

        
        feat_k_pool, sample_ids, attn_mats = self.PS_router.apply_cover(feat_k, num_patches=self.num_patches)
        feat_q_pool = self.PS_router.apply_encoded(feat_q, num_patches=self.num_patches, patch_ids=sample_ids, attn_mats=attn_mats)
        
        # print()
        # print(tcolors.RED,"feat_k_pool:", feat_k_pool[0][0],tcolors.ENDC)
        # print(tcolors.RED,"feat_q_pool: ", feat_q_pool[0][0],tcolors.ENDC)
        # print()


        total_nce_loss = 0.0
        for f_q, f_k in zip(feat_q_pool, feat_k_pool):
            loss = self.nceloss_router(f_q, f_k) * self.lambda_NCE
            total_nce_loss += tf.math.reduce_mean(loss) # tf.math.reduce_mean
            # print(tcolors.RED, total_nce_loss, tcolors.ENDC)

        return total_nce_loss / self.n_layers