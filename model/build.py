from model import objectives
from .clip_model import (
    Transformer, 
    QuickGELU, 
    LayerNorm, 
    build_CLIP_from_openai_pretrained, 
    convert_weights, 
    VisionTransformer, 
    resize_pos_embed,
    # resize_text_pos_embed
)

import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
import copy
import random
import torch.nn.functional as F

from transformers import BartConfig

from .attn import Config,SelfAttention,OutputLayer
from .bartmodel import BartForLM,BartModel

def tokenize(caption: str, tokenizer, text_length=77, truncate=True) -> torch.LongTensor:
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    tokens = [sot_token] + tokenizer.encode(caption) + [eot_token]

    result = torch.zeros(text_length, dtype=torch.long)
    if len(tokens) > text_length:
        if truncate:
            tokens = tokens[:text_length]
            tokens[-1] = eot_token
        else:
            raise RuntimeError(
                f"Input {caption} is too long for context length {text_length}"
            )
    result[:len(tokens)] = torch.tensor(tokens)
    return result


class vaeLinearEncoder(nn.Module):
    def __init__(self,vec_dim,hidden_space_dim):
        super(vaeLinearEncoder,self).__init__()
        self.linears = nn.Sequential(
            nn.Linear(vec_dim,vec_dim),
            # 激活函数十分重要，在VAE中必须保证非负，否则会很难训练
            nn.ReLU(),
            nn.Linear(vec_dim,hidden_space_dim),
            nn.ReLU(),
            nn.Linear(hidden_space_dim,hidden_space_dim),
            nn.ReLU(),
        )
    def forward(self,inputs):
        out = self.linears(inputs)
        return out
    
class vaeLinearDecoder(nn.Module):
    def __init__(self,latent_space_dim,hidden_space_dim,vec_dim):
        super(vaeLinearDecoder,self).__init__()
        self.linears = nn.Sequential(
            nn.Linear(latent_space_dim,hidden_space_dim),
            nn.ReLU(),
            nn.Linear(hidden_space_dim,hidden_space_dim),
            nn.ReLU(),
            nn.Linear(hidden_space_dim,vec_dim),
            # 这个sigmoid可以换tanh。或者其他类似的，必须保证非线性和非负
            nn.Sigmoid(),
        )
    def forward(self,z_hidden):
        imgs_re = self.linears(z_hidden)
        return imgs_re
    
class CVAEModel(nn.Module):
    def __init__(self,
                 vec_dim,
                 hidden_space_dim,
                 latent_space_dim
                 ):
        super(CVAEModel,self).__init__()
        
        self.encoder = vaeLinearEncoder(vec_dim,hidden_space_dim)
        self.decoder = vaeLinearDecoder(latent_space_dim,hidden_space_dim,vec_dim)
        
        self.fitting_mean = nn.Linear(in_features=hidden_space_dim,out_features=latent_space_dim)
        self.fitting_log_var = nn.Linear(in_features=hidden_space_dim,out_features=latent_space_dim)
        self.rec_loss = nn.MSELoss(reduction="sum")
        
    def forward(self,x_g,cond):
        x_hidden = self.encoder(x_g)
        cond_hidden = self.encoder(cond)
        
        mean_pred = self.fitting_mean(x_hidden)

        cond_mean = self.fitting_mean(cond_hidden)
        log_var_pred = self.fitting_log_var(x_hidden)
        
        # sample from N(0,I)
        # re-parameter skill
        std = torch.exp(log_var_pred/2)
        eps = torch.randn_like(std)
        z = mean_pred + eps * std
        #miu_norm = torch.randn_like(x_hidden)
        #z = torch.exp(log_var_pred) * miu_norm + mean_pred
        
        x_g_re = self.decoder(z)
        
        #reconst_loss = F.binary_cross_entropy(x_g_re, x_g, size_average=False)
        reconst_loss = self.rec_loss(x_g_re, x_g)
        
        kl_div = - 0.5 * torch.sum(1 + log_var_pred - (mean_pred-cond_mean).pow(2) - log_var_pred.exp())
        
        return x_g_re,kl_div,reconst_loss

@torch.no_grad()
def sinkhorn_knopp(out,sinkhorn_iterations=3,epsilon=0.05, world_size=1):
    Q = torch.exp(out / epsilon).t() # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1] * world_size # number of samples to assign
    K = Q.shape[0] # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    # dist.all_reduce(sum_Q)
    Q /= sum_Q

    for it in range(sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        # dist.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q.t()

class IRRA(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self.proj_token_num = args.proj_token_num
        self._set_task()

        #self.base_model, base_cfg, self.pe_enlarge = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim']
        #self.text_pe = resize_text_pos_embed(copy.deepcopy(self.base_model.positional_embedding), new_seqlen = args.atr_length)

        # self.pe_enlarge = resize_pos_embed(
        #     self.pe_enlarge, 
        #     torch.rand((self.args.enl_img_size[0]//self.args.stride_size*self.args.enl_img_size[1]//self.args.stride_size+1,self.pe_enlarge.shape[-1])), 
        #     self.args.enl_img_size[0]//self.args.stride_size, 
        #     self.args.enl_img_size[1]//self.args.stride_size)
        # # # v = resize_pos_embed(v, self.visual.positional_embedding, self.visual.num_y, self.visual.num_x)


        self.logit_scale = torch.ones([]) * (1 / args.temperature) 
        self.proj_dec_cfg = None
        if "cap" in args.loss_names:
            self.cap_cfg = BartConfig(encoder_layers=0,decoder_layers=args.decoder_depth,d_model=self.base_model.visual.transformer.width)
            self.caption_decoder = BartForLM(self.cap_cfg)
        
        if "match" in args.loss_names:
            if "dot" in self.current_task:
                self.atr_matching_head = nn.Linear(self.embed_dim,2)
            elif "cat" in self.current_task:
                self.atr_matching_head = nn.Linear(self.embed_dim*2,2)
            pass
        if "queue" in args.loss_names:
            self.queue_t = torch.zeros(
                args.cont_queue_size,
                self.embed_dim,
                requires_grad=False
            ).cuda()
            self.queue_i = torch.zeros(
                args.cont_queue_size,
                self.embed_dim,
                requires_grad=False
            ).cuda()
            self.queue_y = torch.zeros(
                args.cont_queue_size,
                self.embed_dim,
                requires_grad=False
            ).cuda()
            self.queue_id = torch.zeros(
                args.cont_queue_size,
                1,
                requires_grad=False
            ).long().cuda()

        if "prot" in args.loss_names:
            self.prototypes = nn.Linear(self.embed_dim, args.nmb_prototypes, bias=False)
            if args.queue_size>0:
                self.queue = torch.zeros(
                    len(args.crops_for_assign),
                    args.queue_size,
                    self.embed_dim,
                    requires_grad=False
                ).cuda()

        if "vae" in args.loss_names:
            self.cvae = CVAEModel(
                vec_dim=self.embed_dim,
                hidden_space_dim=256,
                latent_space_dim=8
            )
        self.linear_proj = None
        if "linear" in args.loss_names:
            embed_dim = self.base_model.visual.transformer.width
            fc_std = (2 * embed_dim)**-0.5
            scale = embed_dim**-0.5
            proj_std = scale * ((2 * 1)**-0.5)
        
            # self.linear_proj = nn.Sequential(
            #     OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
            #                 ('gelu', QuickGELU()),
            #                 ('ln', LayerNorm(self.embed_dim)),
            #                 ('fc', nn.Linear(self.embed_dim, self.embed_dim))]))
            # # init mlm head
            # nn.init.normal_(self.linear_proj.dense.weight, std=fc_std)
            # nn.init.normal_(self.linear_proj.fc.weight, std=proj_std)
            self.linear_proj = nn.Sequential(
                OrderedDict([('dense1', nn.Linear(self.embed_dim, self.embed_dim)),
                            ('relu', nn.ReLU()),
                            ('ln', LayerNorm(self.embed_dim)),
                            ('dense2', nn.Linear(self.embed_dim, self.embed_dim)),
                            ('relu', nn.ReLU()),
                            ('ln2', LayerNorm(self.embed_dim)),
                            ('fc', nn.Linear(self.embed_dim, self.embed_dim))]))
            # init mlm head
            nn.init.normal_(self.linear_proj.dense1.weight, std=fc_std)
            nn.init.normal_(self.linear_proj.dense2.weight, std=fc_std)
            nn.init.normal_(self.linear_proj.fc.weight, std=proj_std)

        if "token" in args.loss_names:
            embed_dim = self.base_model.visual.transformer.width
            self.proj_dec_cfg = BartConfig(
                encoder_layers=2,
                decoder_layers=2,
                d_model=embed_dim
            )
            self.token_proj = BartModel(self.proj_dec_cfg)
            pass

        if 'fusion' in args.loss_names:
            self.dim=64
            self.fusion_config_ff = Config(hidden_size=self.dim,num_attention_heads=1)
            self.fusion_attn_ff = SelfAttention(config=self.fusion_config_ff)
            self.fusion_output_ff = OutputLayer(config=self.fusion_config_ff)
            
            if 'evafusion' in args.loss_names:
                self.fusion_config_all = Config(hidden_size=self.dim+self.embed_dim,num_attention_heads=1+self.embed_dim//64)
                self.fusion_attn_all = SelfAttention(config=self.fusion_config_all)
                self.fusion_output_all = OutputLayer(config=self.fusion_config_all)

                self.img_projection = nn.Linear(self.dim+self.embed_dim, self.embed_dim)
                nn.init.normal_(self.img_projection.weight.data, std=0.001)
                nn.init.constant_(self.img_projection.bias.data, val=0.0)

            if 'id' in args.loss_names:
                self.classifier_fusion = nn.Linear(self.dim, self.num_classes)
                nn.init.normal_(self.classifier_fusion.weight.data, std=0.001)
                nn.init.constant_(self.classifier_fusion.bias.data, val=0.0)

                pass
        if 'id' in args.loss_names:
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)

        if 'proj' in args.loss_names:
            embed_dim = self.base_model.visual.transformer.width
            self.cross_attn = nn.MultiheadAttention(embed_dim,
                                                    embed_dim // 64,
                                                    batch_first=True)
            self.cross_modal_transformer = Transformer(width=embed_dim,
                                                       layers=args.cmt_depth,
                                                       heads=embed_dim //
                                                       64)
            scale = self.cross_modal_transformer.width**-0.5
            
            self.ln_pre_t = LayerNorm(embed_dim)
            self.ln_pre_i = LayerNorm(embed_dim)
            self.ln_post = LayerNorm(embed_dim)

            proj_std = scale * ((2 * self.cross_modal_transformer.layers)**-0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width)**-0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            # init cross attn
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

            self.proj_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                            ('gelu', QuickGELU()),
                            ('ln', LayerNorm(self.embed_dim)),
                            ('fc', nn.Linear(self.embed_dim, self.embed_dim))]))
            
            # # init proj_head head
            # nn.init.normal_(self.proj_head.dense.weight, std=fc_std)
            # nn.init.normal_(self.proj_head.fc.weight, std=proj_std)
            ## concat feature by token
            #### V2 62
            # self.cross_attn_mm = nn.MultiheadAttention(self.embed_dim,
            #                                         self.embed_dim // 64,
            #                                         batch_first=True)
            # self.cross_modal_transformer_mm = Transformer(width=self.embed_dim,
            #                                            layers=args.cmt_depth,
            #                                            heads=self.embed_dim //
            #                                            64)
            # scale = self.cross_modal_transformer_mm.width**-0.5
            
            # self.ln_pre_t_mm = LayerNorm(self.embed_dim)
            # self.ln_pre_i_mm = LayerNorm(self.embed_dim)
            # self.ln_post_mm = LayerNorm(self.embed_dim)

            # proj_std = scale * ((2 * self.cross_modal_transformer_mm.layers)**-0.5)
            # attn_std = scale
            # fc_std = (2 * self.cross_modal_transformer_mm.width)**-0.5
            # for block in self.cross_modal_transformer_mm.resblocks:
            #     nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            #     nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            #     nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            #     nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            # # init cross attn
            # nn.init.normal_(self.cross_attn_mm.in_proj_weight, std=attn_std)
            # nn.init.normal_(self.cross_attn_mm.out_proj.weight, std=proj_std)
            ### V3
            self.cross_attn_mm = copy.deepcopy(self.base_model.visual.transformer.resblocks[-1])

            self.proj_dec_cfg = BartConfig(
                decoder_layers=1,
                d_model=embed_dim
            )
            
            self.proj_prefix = nn.Parameter(torch.randn((self.proj_token_num,embed_dim)))
            self.proj_dec = BartForLM(self.proj_dec_cfg)
            pass

        if 'mlm' in args.loss_names:
            self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            self.cross_modal_transformer = Transformer(width=self.embed_dim,
                                                       layers=args.cmt_depth,
                                                       heads=self.embed_dim //
                                                       64)
            scale = self.cross_modal_transformer.width**-0.5
            
            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)

            proj_std = scale * ((2 * self.cross_modal_transformer.layers)**-0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width)**-0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            # init cross attn
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

            self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                            ('gelu', QuickGELU()),
                            ('ln', LayerNorm(self.embed_dim)),
                            ('fc', nn.Linear(self.embed_dim, args.vocab_size))]))
            # init mlm head
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)


    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')
    
    def set_sent_list(self,tokenizer,text_length,truncate):
        tmp_sent = ["The person in the picture is {}."]
        tmp_sent_light = ["The light in the picture is {}."]
        tmp_sent_angle = ["The person in the picture is {}.", "The picture is taken from {}."]
        sentence_list = [
            [###gender: male=0, female = 1
                [t_s.format(kw) for t_s in tmp_sent for kw in ["male","a man","not female", "not a woman"]],
                [t_s.format(kw) for t_s in tmp_sent for kw in ["female","a woman","not male", "not a man"]]
            ],
            [### age: young=1, old = 2, adult=0
                [t_s.format(kw) for t_s in tmp_sent for kw in ["adult","adulthood","come of age","middle age", "middle-aged"]],
                [t_s.format(kw) for t_s in tmp_sent for kw in ["youth","youngster","the young", "young"]],
                [t_s.format(kw) for t_s in tmp_sent for kw in ["old","wrinkly","the older"]]
            ],
            [### light: # dim=1, bright=0
                [t_s.format(kw) for t_s in tmp_sent_light for kw in ["bright","brilliant","light", "lighting", "brightness"]],
                [t_s.format(kw) for t_s in tmp_sent_light for kw in ["dark","dim","half-light", "gloom"]]
            ],
            [### weight: slim=1, fat=2, middle=0
                [t_s.format(kw) for t_s in tmp_sent for kw in ["medium built", "average weight", "middle weight", "medium size"]],
                [t_s.format(kw) for t_s in tmp_sent for kw in ["thin","slim","slender", "lean"]],
                [t_s.format(kw) for t_s in tmp_sent for kw in ["fat","solid"]]
            ],
            [### height: tall=1, short=2, side=0
                [t_s.format(kw) for t_s in tmp_sent for kw in ["average height", "not tall or short", "medium height"]],
                [t_s.format(kw) for t_s in tmp_sent for kw in ["tall"]],
                [t_s.format(kw) for t_s in tmp_sent for kw in ["short"]]
            ],
            [### angle: # front=1, back = 2, other=0
                [tmp_sent_angle[1].format(kw) for kw in ["the side","side-shot"]],
                [tmp_sent_angle[0].format(kw) for kw in ["facing the camera",]]+\
                [tmp_sent_angle[1].format(kw) for kw in ["the front",]],
                [tmp_sent_angle[0].format(kw) for kw in ["away from the camera",]]+\
                [tmp_sent_angle[1].format(kw) for kw in ["the back",]],
            ],
            [### attitude: # Aerial=1, ground=0
                ["This person seems to be at the same altitude as camera.",
                "This person is at the horizontal position of the camera."],
                ["This photo is from a top view.",
                "The picture was taken from a height down.",
                "This person's horizontal position is below the camera."]
            ],
        ]
        merge_list = []
        for atr in sentence_list:
            for label in atr:
                merge_list += label
        tokenized_sent = torch.stack([tokenize(sent,tokenizer,text_length=text_length, truncate=truncate) for sent in merge_list],dim=0)
        self.size_list = [len(atr) for atr in sentence_list]
        self.sent_size_list = [[len(label) for label in atr] for atr in sentence_list]
        self.tokenized_sent = tokenized_sent
    
    def get_idx_pos_and_neg(self,idx_atr,idx_label):
        # idx_atr代表第几个属性，idx_label代表这个属性对应的label
        bias_atr = sum([sum(label) for label in self.sent_size_list[:idx_atr]])
        bias_idx = sum(self.sent_size_list[idx_atr][:idx_label])
        idx_rdm_pos = random.randint(0,self.sent_size_list[idx_atr][idx_label]-1)
        idx_neg = (idx_label+1)%self.size_list[idx_atr]
        bias_neg_idx = sum(self.sent_size_list[idx_atr][:idx_neg])
        idx_rdm_neg = random.randint(0,self.sent_size_list[idx_atr][idx_neg]-1)

        return bias_atr+bias_idx+idx_rdm_pos,bias_atr+bias_neg_idx+idx_rdm_neg

    
    def cross_former(self, q, k, v):
        x = self.cross_attn(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x

# # V2 func
#     def cross_former_mm(self, q, k, v):
#         x = self.cross_attn_mm(
#                 self.ln_pre_i_mm(q),
#                 self.ln_pre_i_mm(k),
#                 self.ln_pre_i_mm(v),
#                 need_weights=False)[0]
#         x = x.permute(1, 0, 2)  # NLD -> LND
#         x = self.cross_modal_transformer_mm(x)
#         x = x.permute(1, 0, 2)  # LND -> NLD

#         x = self.ln_post_mm(x)
#         return x
    def cross_former_mm(self,q,k,v):
        # V3 func
        x = self.cross_attn_mm(q)
        return x
    
    def encode_pair_image(self, image, image_pair):
        if "proj" in self.current_task:
            bs = image.shape[0]
            # x_self = self.base_model.encode_image(image).half()
            x_self_conv = self.base_model.visual.conv_embedding(image.half())
            # x_self = self.base_model.visual.forward_body(x_self_conv)

            #x_pair = self.base_model.encode_image(image_pair).half()
            x_pair_conv = self.base_model.visual.conv_embedding(image_pair.half())
            y_gt = self.select_topk_token(x_pair_conv[:,1:,:],x_pair_conv[:, :1, :],self.proj_token_num)
            # x_pair = self.base_model.visual.forward_body(x_pair_conv)


            # i_y_feats = x_pair[:, 0, :]
            # sim_tkn = torch.nn.functional.cosine_similarity(i_y_feats.unsqueeze(1),x_pair[:,1:,:],dim=-1)
            # _, idx = torch.sort(sim_tkn,dim=1)
            # y_idx = idx[:,-self.proj_token_num:].reshape(-1)+1
            # x_idx = torch.arange(bs).unsqueeze(0).reshape(-1,1).repeat(1,self.proj_token_num).reshape(-1)
            # y_gt = x_pair[x_idx,y_idx]

            x_add = torch.cat([y_gt.reshape(bs,self.proj_token_num,-1),x_self_conv],dim=1)
            x_add = self.base_model.visual.forward_body_pe(x_add,self.pe_enlarge)
            #i_add = self.proj_head(torch.cat([i_feats.to(i_y_feats.dtype),i_y_feats],dim=-1)).float()
            # x_add = torch.cat([image_feats[:, :1, :],y_gt.reshape(bs,self.proj_token_num,-1),image_feats[:, 1:, :]],dim=1)
            # x_add = self.cross_former_mm(x_add,x_add,x_add)
            # x_add = x_add @ self.base_model.visual.proj

            # x_add = torch.cat([x_self[:, :1, :],y_gt.reshape(bs,self.proj_token_num,-1),x_self[:, 1:, :]],dim=1)
            # x_add = self.cross_former_mm(x_add,x_add,x_add)
            #x = self.proj_head(x_add[:, 0, :]).float()
            x = (x_add @ self.base_model.visual.proj)[:, 0, :].float()

            #x = self.proj_head(torch.cat([x_self,x_pair],dim=-1)).float()
            return x 
        else:
            return self.encode_image(image)
    
    # def encode_pair_image(self, image, image_pair):
    #     x_self = self.base_model.encode_image(image)[:, 0, :].half()
    #     x_pair = self.base_model.encode_image(image_pair)[:, 0, :].half()
    #     x = self.proj_head(torch.cat([x_self,x_pair],dim=-1)).float()
    #     return x 
    #     pass
        

    def encode_image(self, image):
        x = self.base_model.encode_image(image)
        if isinstance(self.base_model.visual, VisionTransformer):
            if 'evafusion' in self.current_task:
                x_ff = x[:,:,:self.dim]
                bs,seq_len,_ = x.shape
                attn_mask = torch.LongTensor([[1]]).repeat((bs,seq_len)).to(x.device)
                x_all_out = self.fusion_attn_all(hidden_states=torch.cat([x_ff,x],dim=-1),attention_mask = attn_mask[:, None, None, :])[0]
                x_all_out = self.fusion_output_all(hidden_states = x_all_out,input_tensor = torch.cat([x_ff,x],dim=-1))
                # CLS feature of all image information
                x = self.img_projection(x_all_out[:, 0, :]).float()
                pass
            else:
                x = x[:, 0, :].float()
        else:
            x = x.float() # for CLIP ResNet visual model

        return x
        # return x.float() # for CLIP ResNet visual model
    
    def select_topk_token(self,feats,cls,k=16,add_cls=False):
        bs = feats.shape[0]

        sim_tkn = torch.nn.functional.cosine_similarity(cls,feats,dim=-1)
        _, idx = torch.sort(sim_tkn,dim=1)
        y_idx = idx[:,-k:].reshape(-1)
        if add_cls:
            y_idx += 1

        x_idx = torch.arange(bs).unsqueeze(0).reshape(-1,1).repeat(1,k).reshape(-1)
        selected = feats[x_idx,y_idx]
        return selected
    
    def forward_proj(self,image_feats):
        bs = image_feats.shape[0]
        x = self.cross_former(
            self.proj_prefix.unsqueeze(0).repeat(bs,1,1).to(image_feats.dtype),
                image_feats, 
                image_feats
            )
        x_casual = self.proj_dec(
                inputs_embeds = x,
                is_casual=True
            )
        x_attn = self.proj_dec(
            inputs_embeds = x,
            is_casual=False
        )
        x_dec = (x_casual[0]+x_attn[0])*0.5
        return x_dec
        pass

    def forward_token_proj(self,image_feats_conv):
        # x = self.token_proj.encoder(inputs_embeds=image_feats_conv)[0]
        # x = self.token_proj.decoder(inputs_embeds=x)[0]
        x = self.token_proj.decoder(inputs_embeds=image_feats_conv)[0]
        return x
    
    def encode_atr_text(self, atr_tokens):
        x = self.base_model.token_embedding(atr_tokens).type(self.base_model.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.text_pe.type(self.base_model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.base_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.base_model.ln_final(x).type(self.base_model.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), atr_tokens.argmax(dim=-1)] @ self.base_model.text_projection
        #x = x @ self.base_model.text_projection

        return x

    def encode_text(self, text):
        x = self.base_model.encode_text(text)
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    def forward(self, batch, cu_ep):
        ret = dict()

        logit_scale = self.logit_scale
        ret.update({'temperature': 1 / logit_scale})

        images = batch['images']
        bs = images.shape[0]
        caption_ids = batch['caption_ids']
        loss_sim=0
        if batch['pair_img'] is None:
            image_feats, text_feats = self.base_model(images, caption_ids)
            if isinstance(self.base_model.visual, VisionTransformer):
                i_feats = image_feats[:, 0, :].float()
            else:
                i_feats = image_feats.float()
        else:
            text_feats = self.base_model.encode_text(caption_ids)
            y_pair = batch['pair_img']
            if isinstance(self.base_model.visual, VisionTransformer):
                # with torch.no_grad():
                image_feats_conv = self.base_model.visual.conv_embedding(images.half())
            
                # y_pair_feats = self.base_model.encode_image(y_pair)
                y_pair_feats_conv = self.base_model.visual.conv_embedding(y_pair.half())
                
                image_feats = self.base_model.visual.forward_body(image_feats_conv)
                i_feats = (image_feats @ self.base_model.visual.proj)[:, 0, :].float()

                y_pair_feats = self.base_model.visual.forward_body(y_pair_feats_conv)
                y_feats = (y_pair_feats @ self.base_model.visual.proj)[:, 0, :].float()
            else:
                i_feats = self.base_model.encode_image(images.half()).float()
                y_feats = self.base_model.encode_image(y_pair.half()).float()
            #i_feats = image_feats[:, 0, :].float()

        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()
        if "match2" in self.current_task:
            atr_sent_embd = self.encode_text(self.tokenized_sent.to(i_feats.device))
            atr_a = batch['atr_a']
            atr_g = batch['atr_g']
            sim_pos_list = []
            sim_neg_list = []
            for idx_atr in range(len(self.size_list)):
                for idx_s in range(bs):
                    idx_pos,idx_neg = self.get_idx_pos_and_neg(idx_atr,atr_a[idx_s][idx_atr].tolist())
                    if "dot" in self.current_task:
                        sim_pos = self.atr_matching_head((atr_sent_embd[idx_pos] * i_feats[idx_s]).half())
                        sim_neg = self.atr_matching_head((atr_sent_embd[idx_neg] * i_feats[idx_s]).half())
                    elif "cat" in self.current_task:
                        sim_pos = self.atr_matching_head(torch.cat([atr_sent_embd[idx_pos], i_feats[idx_s]],dim=-1).half())
                        sim_neg = self.atr_matching_head(torch.cat([atr_sent_embd[idx_neg], i_feats[idx_s]],dim=-1).half())
                    sim_pos_list.append(sim_pos)
                    sim_neg_list.append(sim_neg)
                    idx_pos,idx_neg = self.get_idx_pos_and_neg(idx_atr,atr_g[idx_s][idx_atr].tolist())
                    if "dot" in self.current_task:
                        sim_pos = self.atr_matching_head((atr_sent_embd[idx_pos] * y_feats[idx_s]).half())
                        sim_neg = self.atr_matching_head((atr_sent_embd[idx_neg] * y_feats[idx_s]).half())
                    elif "cat" in self.current_task:
                        sim_pos = self.atr_matching_head(torch.cat([atr_sent_embd[idx_pos], y_feats[idx_s]],dim=-1).half())
                        sim_neg = self.atr_matching_head(torch.cat([atr_sent_embd[idx_neg], y_feats[idx_s]],dim=-1).half())
                    sim_pos_list.append(sim_pos)
                    sim_neg_list.append(sim_neg)

            sim_pos_list = torch.stack(sim_pos_list,dim=0)
            sim_neg_list = torch.stack(sim_neg_list,dim=0)
            label_pos = torch.ones(sim_pos_list.shape[0]).to(i_feats.device)
            label_neg = torch.zeros(sim_neg_list.shape[0]).to(i_feats.device)

            loss_match2 = F.cross_entropy(sim_pos_list,label_pos.long())+F.cross_entropy(sim_neg_list,label_neg.long())
            ret.update({'match2_loss':loss_match2*self.args.match_loss_weight})


        # if "match" in self.current_task:
        #     flag = random.random()
        #     if flag > 0.5:
        #         atr_p = batch['atr_a_pos']
        #         atr_n = batch['atr_a_neg']
        #         img_f = i_feats
        #         pass
        #     else:
        #         atr_p = batch['atr_g_pos']
        #         atr_n = batch['atr_g_neg']
        #         img_f = y_feats
        #         pass
        #     pos_text_list = []
        #     neg_text_list = []
        #     img_list = []
        #     s_size = atr_p.shape[1]
        #     for i_s in range(bs):
        #         pos_text_list.append(self.encode_atr_text(atr_p[i_s][:,:self.args.atr_length+1]))
        #         neg_text_list.append(self.encode_atr_text(atr_n[i_s][:,:self.args.atr_length+1]))
        #         img_list.append(img_f[i_s].unsqueeze(0).repeat(s_size,1))
            
        #     mul_pos_list = []
        #     mul_neg_list = []

        #     for i_s in range(bs):
        #         mul_pos_list.append(pos_text_list[i_s]*img_list[i_s])
        #         mul_neg_list.append(neg_text_list[i_s]*img_list[i_s])
            
        #     conf_pos = self.atr_matching_head(torch.cat(mul_pos_list,dim=0).half())
        #     conf_neg = self.atr_matching_head(torch.cat(mul_neg_list,dim=0).half())
        #     label_pos = torch.ones(conf_pos.shape[0]).to(i_feats.device)
        #     label_neg = torch.zeros(conf_pos.shape[0]).to(i_feats.device)
        #     loss_match = F.cross_entropy(conf_pos,label_pos.long())+F.cross_entropy(conf_neg,label_neg.long())
        #     ret.update({'match_loss':loss_match})
        #     pass
            
            #self.atr_matching_head = nn.Linear(self.embed_dim,2)

        # if "prot" in self.current_task and cu_ep >= 0:
        #     crops_for_assign = self.args.crops_for_assign
        #     feat_list = [i_feats,t_feats,y_feats]
        #     embds = torch.cat(feat_list[:len(crops_for_assign)])
        #     output = self.prototypes(embds)
        #     nmb_crops = [len(crops_for_assign)]
        #     prot_loss = 0
        #     for i, crop_id in enumerate(crops_for_assign):
        #         with torch.no_grad():
        #             out = output[bs * crop_id: bs * (crop_id + 1)]
        #             if self.args.queue_size>0:
        #                 if cu_ep >= self.args.queue_epoch:
        #                     out = torch.cat((torch.mm(
        #                         self.queue[i],
        #                         self.prototypes.weight.t()
        #                     ), out))
        #                 # fill the queue
        #                 self.queue[i, bs:] = self.queue[i, :-bs].clone()
        #                 self.queue[i, :bs] = embds[crop_id * bs: (crop_id + 1) * bs]

        #             q = sinkhorn_knopp(out,epsilon=self.args.prot_eps,sinkhorn_iterations=self.args.sink_iter)[-bs:]
        #         subloss = 0
        #         #print("out:",output)
        #         #print("q:",q)
        #         for v in np.delete(np.arange(np.sum(nmb_crops)), crop_id):
        #             x = output[bs * v: bs * (v + 1)] / self.args.prot_temp
        #             subloss -= torch.mean(torch.sum(q * F.log_softmax(x, dim=1), dim=1))

        #     prot_loss += subloss / (np.sum(nmb_crops) - 1)
        #     #print("prot_loss",prot_loss)
        #     ret.update({'prot_loss':prot_loss})

        # image_feats = self.base_model.visual.forward_body(image_feats_conv)
        #image_feats, text_feats = self.base_model(images, caption_ids)
        # print(isinstance(self.base_model.visual, VisionTransformer))
        # print(self.base_model.visual)
        # if isinstance(self.base_model.visual, VisionTransformer):
        #     i_feats = (image_feats @ self.base_model.visual.proj)[:, 0, :].float()
        # else:
        #     i_feats = image_feats.float() # for CLIP ResNet visual model
        # if "cap" in self.current_task:
        #     _,img_seq,_ = image_feats.shape
        #     _,txt_seq = caption_ids.shape
        #     caption_ids_ = torch.cat([caption_ids,torch.zeros((bs,img_seq-txt_seq)).long().to(caption_ids.device)],dim=1)
        #     output = self.caption_decoder(inputs_embeds=image_feats,labels=caption_ids_)
        #     ret.update({'cap_loss':output.loss})
        #     pass


        # if "exp" in self.current_task:
        #     sim = (i_feats / i_feats.norm(dim=1, keepdim=True)) @ (t_feats / t_feats.norm(dim=1, keepdim=True)).t()
        #     pid = batch['pids'].unsqueeze(0).reshape(-1,1)
        #     label = torch.eq(pid,pid.t()).float()
        #     _,index = torch.sort(sim*(1-label) - label,dim=1)
        #     idx = index[:,-1:]
        #     sim_pos = sim[torch.arange(bs),torch.arange(bs)]
        #     sim_neg = sim[torch.arange(bs),idx.reshape(bs)]
        #     exp_loss = torch.mean(((1-sim_pos).exp()-1)+sim_neg.exp())
        #     ret.update({'exp_loss':exp_loss})

        # if "exp2" in self.current_task:
        #     sim = (i_feats / i_feats.norm(dim=1, keepdim=True)) @ (t_feats / t_feats.norm(dim=1, keepdim=True)).t()
        #     pid = batch['pids'].unsqueeze(0).reshape(-1,1)
        #     label = torch.eq(pid,pid.t()).float()
        #     exp2_loss = torch.mean(torch.sum(label*((1-F.softmax(sim,dim=1)).exp()-1)+(1-label)*(F.softmax(sim,dim=1).exp()-1),dim=1))
        #     ret.update({'exp2_loss':exp2_loss})

        # if "sig" in self.current_task:
        #     sim = (i_feats / i_feats.norm(dim=1, keepdim=True)) @ (t_feats / t_feats.norm(dim=1, keepdim=True)).t()
        #     pid = batch['pids'].unsqueeze(0).reshape(-1,1)
        #     label = torch.eq(pid,pid.t()).float()
        #     sig_loss = torch.mean(-torch.sum(label * F.logsigmoid(sim) + (1. - label) * F.logsigmoid(-sim), axis=-1))
        #     ret.update({'sig_loss':sig_loss})

        # if "linear" in self.current_task:
        #     x_g_re = self.linear_proj(i_feats.clone().detach().half())
        #     linear_loss = nn.functional.mse_loss(x_g_re,y_feats.clone().detach().half(),reduce="mean")
        #     linearl1_loss = nn.functional.l1_loss(x_g_re,y_feats.clone().detach().half(),reduce="mean")
        #     ret.update({'linear_loss':linear_loss})
        #     ret.update({'linearl1_loss':linearl1_loss})
        
        # if "dist" in self.current_task:
        #     ret.update({'dist_loss':torch.dist(i_feats,t_feats)})

        # if "vaerec" in self.current_task or "vaekl" in self.current_task:
        #     x_g_re,kl_div,reconst_loss = self.cvae(y_feats.clone().detach().half(),i_feats.clone().detach().half())
        #     ret.update({'vaerec_loss':reconst_loss})
        #     ret.update({'vaekl_loss':kl_div})
        #     pass
        # if "token" in self.current_task:
        #     x_dec = self.forward_token_proj(image_feats_conv)
        #     loss_proj = torch.nn.functional.l1_loss(x_dec,y_pair_feats_conv.clone().detach()) + \
        #         torch.nn.functional.mse_loss(x_dec,y_pair_feats_conv.clone().detach())
            
        #     # token_feats = self.base_model.visual.forward_body(x_dec)
        #     # token_i_feats = (token_feats @ self.base_model.visual.proj)[:, 0, :].float()
        #     # loss_sim = (objectives.compute_sdm(token_i_feats, i_feats.clone().detach(), batch['pids'], logit_scale)+\
        #     #     objectives.compute_sdm(token_i_feats, y_feats.clone().detach(), batch['pids'], logit_scale))*0.5
        #     ret.update({'token_loss':loss_proj})
        #     # ret.update({'tokensdm_loss':loss_sim})
        #     pass
        
        # if "proj" in self.current_task:
        #     x_dec = self.forward_proj(image_feats_conv)
        #     x_dec = x_dec.reshape(bs*self.proj_token_num,-1)

        #     # Select top k token by cosine similarity
        #     # y_pair = batch['pair_img']
        #     # # y_pair_feats = self.base_model.encode_image(y_pair)
        #     # y_pair_feats_conv = self.base_model.visual.conv_embedding(y_pair.half())
        #     # y_pair_feats = self.base_model.visual.forward_body(y_pair_feats_conv)

        #     # i_y_feats = y_pair_feats[:, 0, :]
        #     # sim_tkn = torch.nn.functional.cosine_similarity(i_y_feats.unsqueeze(1),y_pair_feats[:,1:,:],dim=-1)
        #     # _, idx = torch.sort(sim_tkn,dim=1)
        #     # y_idx = idx[:,-self.proj_token_num:].reshape(-1)+1
        #     # x_idx = torch.arange(bs).unsqueeze(0).reshape(-1,1).repeat(1,self.proj_token_num).reshape(-1)
        #     # y_gt = y_pair_feats[x_idx,y_idx]
        #     #####
        #     y_gt = self.select_topk_token(y_pair_feats_conv[:,1:,:],y_pair_feats_conv[:, :1, :],self.proj_token_num)
        #     # add L1 and L2 Loss
            
        #     loss_proj = torch.nn.functional.l1_loss(x_dec,y_gt) + torch.nn.functional.mse_loss(x_dec,y_gt)

        #     x_add = torch.cat([y_gt.reshape(bs,self.proj_token_num,-1),image_feats_conv],dim=1).half()
        #     # print(image_feats_conv.shape)
        #     #print(y_gt.reshape(bs,self.proj_token_num,-1).shape)
        #     x_add = self.base_model.visual.forward_body(x_add)
        #     #i_add = self.proj_head(torch.cat([i_feats.to(i_y_feats.dtype),i_y_feats],dim=-1)).float()
        #     # x_add = torch.cat([image_feats[:, :1, :],y_gt.reshape(bs,self.proj_token_num,-1),image_feats[:, 1:, :]],dim=1)
        #     # x_add = self.cross_former_mm(x_add,x_add,x_add)
        #     x_add = x_add @ self.base_model.visual.proj
        #     #i_add = self.proj_head(x_add[:, 0, :]).float()
        #     i_feats_add = x_add[:, 0, :].float()

        #     ret.update({'itc_add_loss':objectives.compute_itc(i_feats_add, t_feats, logit_scale)})

        #     # if 'itc' in self.current_task:
        #     #     ret.update({'itc_add_loss':objectives.compute_itc(i_add, t_feats, logit_scale)})

        #     ret.update({'loss_proj':loss_proj})
        
        if 'id' in self.current_task:
            image_logits = self.classifier(i_feats.half()).float()
            text_logits = self.classifier(t_feats.half()).float()
            if batch['pair_img'] is not None:
                y_image_logits = self.classifier(y_feats.half()).float()
                ret.update({'id_loss':(objectives.compute_id(y_image_logits, text_logits, batch['pids'])+\
                                       objectives.compute_id(image_logits, text_logits, batch['pids']))*self.args.id_loss_weight})
            else:
                ret.update({'id_loss':objectives.compute_id(image_logits, text_logits, batch['pids'])*self.args.id_loss_weight})

            image_pred = torch.argmax(image_logits, dim=1)
            text_pred = torch.argmax(text_logits, dim=1)

            image_precision = (image_pred == batch['pids']).float().mean()
            text_precision = (text_pred == batch['pids']).float().mean()
            ret.update({'img_acc': image_precision})
            ret.update({'txt_acc': text_precision})
        
        # if 'fusion' in self.current_task or 'evafusion' in self.current_task:
        #     # specific information extractor
        #     image_feats_ff = image_feats[:,:,:self.dim]
        #     bs,seq_len,_ = image_feats.shape
        #     attn_mask = torch.LongTensor([[1]]).repeat((bs,seq_len)).to(image_feats.device)
        #     image_feats_ff_out = self.fusion_attn_ff(hidden_states=image_feats_ff,attention_mask = attn_mask[:, None, None, :])[0]
        #     #print(image_feats_ff_out.dtype,image_feats_ff.dtype)
        #     image_feats_ff_out = self.fusion_output_ff(hidden_states = image_feats_ff_out,input_tensor=image_feats_ff)
        #     # CLS feature of these information
        #     i_feats_ff = image_feats_ff[:, 0, :].float()
        #     i_feats_ff_out = image_feats_ff_out[:, 0, :].float()
        #     t_feats_ff = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()[:,:self.dim]
        #     if 'evafusion' in self.current_task:
        #         # cat processed feature and feat
        #         image_feats_all_out = self.fusion_attn_all(hidden_states=torch.cat([image_feats_ff,image_feats],dim=-1),attention_mask = attn_mask[:, None, None, :])[0]
        #         image_feats_all_out = self.fusion_output_all(hidden_states = image_feats_all_out,input_tensor = torch.cat([image_feats_ff,image_feats],dim=-1))
        #         # CLS feature of all image information
        #         i_feats_all_out = self.img_projection(image_feats_all_out[:, 0, :]).float()
            

        #     # ID loss and itc loss
        #     if 'itc' in self.current_task:
        #         ret.update({'itc_loss_ff':(objectives.compute_itc(i_feats_ff, t_feats_ff, logit_scale)+\
        #                                    objectives.compute_itc(i_feats_ff_out, t_feats_ff, logit_scale))*0.5})
        #         if 'evafusion' in self.current_task:
        #             # ret.update({'itc_loss_all': objectives.compute_itc(i_feats_all_out, t_feats, logit_scale)})
        #             ret.update({'itc_loss_all': (objectives.compute_itc(i_feats_all_out, t_feats, logit_scale) + \
        #                                          objectives.compute_itc(i_feats_all_out, i_feats, logit_scale) )*0.5})

        #     if 'id' in self.current_task:
        #         image_logits_ff = self.classifier_fusion(i_feats_ff.half()).float()
        #         text_logits_ff = self.classifier_fusion(t_feats_ff.half()).float()
        #         image_logits_ff_out = self.classifier_fusion(i_feats_ff_out.half()).float()
        #         ret.update({'id_loss_ff':
        #                     (objectives.compute_id(image_logits_ff, text_logits_ff, batch['pids']) + \
        #                      objectives.compute_id(image_logits_ff_out, text_logits_ff, batch['pids']))*0.5*self.args.id_loss_weight
        #             })

        #         image_pred_ff = torch.argmax(image_logits_ff, dim=1)
        #         image_pred_ff_out = torch.argmax(image_logits_ff_out, dim=1)
        #         text_pred_ff = torch.argmax(text_logits_ff, dim=1)

        #         image_precision_ff = (image_pred_ff == batch['pids']).float().mean()
        #         image_precision_ff_out = (image_pred_ff_out == batch['pids']).float().mean()
        #         text_precision_ff = (text_pred_ff == batch['pids']).float().mean()
        #         ret.update({'img_acc_ff': image_precision_ff})
        #         ret.update({'img_acc_ff_out': image_precision_ff_out})
        #         ret.update({'txt_acc_ff': text_precision_ff})

        #         if 'evafusion' in self.current_task:
        #             image_logits_all = self.classifier(i_feats_all_out.half()).float()
        #             ret.update({'id_loss_all':
        #                     objectives.compute_id(image_logits_all, text_logits, batch['pids'])*self.args.id_loss_weight
        #             })
        #             image_pred_all = torch.argmax(image_logits_all, dim=1)
        #             image_precision_all = (image_pred_all == batch['pids']).float().mean()
        #             ret.update({'img_acc_all': image_precision_all})
        #     pass
                
        if 'itc' in self.current_task and batch['pair_img'] is None:
            ret.update({'itc_loss':objectives.compute_itc(i_feats, t_feats, logit_scale)})

        if 'biitc' in self.current_task and batch['pair_img'] is not None:
            ret.update({'biitc_loss':(objectives.compute_itc(i_feats, t_feats, logit_scale)+\
                                        objectives.compute_itc(y_feats, t_feats, logit_scale))/2})
        
        if 'itc' in self.current_task and batch['pair_img'] is not None:
            ret.update({'itc_loss':(objectives.compute_itc(i_feats, t_feats, logit_scale)+\
                                    objectives.compute_itc(y_feats, t_feats, logit_scale)+\
                                        objectives.compute_itc(i_feats, y_feats, logit_scale))/3})

        if ('supkl' in self.current_task or 'supitc' in self.current_task) \
                and batch['pair_img'] is not None:
            
            epsilon = 1e-8
            labels = batch['pids'].reshape(-1,1)
            t_feats_norm = t_feats / t_feats.norm(dim=-1, keepdim=True)
            i_feats_norm = i_feats / i_feats.norm(dim=-1, keepdim=True)
            y_feats_norm = y_feats / y_feats.norm(dim=-1, keepdim=True)
            l_eye = labels.eq(labels.t()).float()
            #labels = labels / labels.sum(dim=1)
            if "queue" in self.current_task:
                labels_ = torch.cat([labels,labels,self.queue_id.clone().detach()],dim=0)
                l_s = torch.zeros((bs,bs+self.args.cont_queue_size)).to(labels.device)
                logits_t2i = logit_scale * t_feats_norm @torch.cat([i_feats_norm,y_feats_norm,self.queue_i.clone().detach()],dim=0).t()
                logits_i2t = logit_scale * i_feats_norm @torch.cat([t_feats_norm,y_feats_norm,self.queue_t.clone().detach()],dim=0).t()
            else:
                labels_ = torch.cat([labels,labels],dim=0)
                l_s = torch.zeros((bs,bs)).to(labels.device)
                logits_t2i = logit_scale * t_feats_norm @torch.cat([i_feats_norm,y_feats_norm],dim=0).t()
                logits_i2t = logit_scale * i_feats_norm @torch.cat([t_feats_norm,y_feats_norm],dim=0).t()
            
            l_eye = torch.cat([l_eye,l_s],dim=1)
            labels = labels.eq(labels_.t()).float()
            labels = labels + l_eye*0.15
            if 'supitc' in self.current_task:
                loss_t = (F.cross_entropy(logits_t2i, F.normalize(labels,dim=1,p=2))+F.cross_entropy(logits_t2i.t(), F.normalize(labels.t(),dim=1,p=2)))/2
                loss_i = (F.cross_entropy(logits_i2t, F.normalize(labels,dim=1,p=2))+F.cross_entropy(logits_i2t.t(), F.normalize(labels.t(),dim=1,p=2)))/2
                #print(loss_i)
                ret.update({'supitc_loss':(loss_i +  loss_t)/2})
            
            t2i_pred = F.softmax(logits_t2i, dim=1)
            i2t_pred = F.softmax(logits_i2t, dim=1)

            labels_distribute = F.normalize(labels,dim=1,p=1)
            # target_t2i = F.softmax(labels, dim=1)
            # target = F.log_softmax(labels, dim=1)

            if 'supkl' in self.current_task:
                loss_i2t = i2t_pred * (F.log_softmax(logits_i2t, dim=1) - torch.log(labels_distribute + epsilon))
                loss_t2i = t2i_pred * (F.log_softmax(logits_t2i, dim=1) - torch.log(labels_distribute + epsilon))
                # loss_t2i = kl_loss(logits_t2i, target)
                # loss_i2t = kl_loss(logits_i2t, target)
                ret.update({'supkl_loss':torch.mean(torch.sum(loss_i2t, dim=1)) + torch.mean(torch.sum(loss_t2i, dim=1))})
                if "queue" in self.current_task:
                    self.queue_t[bs:] = self.queue_t[:-bs].clone()
                    self.queue_t[:bs] = t_feats_norm.clone()
                    self.queue_y[bs:] = self.queue_y[:-bs].clone()
                    self.queue_y[:bs] = y_feats_norm.clone()
                    self.queue_i[bs:] = self.queue_i[:-bs].clone()
                    self.queue_i[:bs] = i_feats_norm.clone()
                    self.queue_id[bs:] = self.queue_id[:-bs].clone()
                    self.queue_id[:bs] = batch['pids'].reshape(-1,1).clone()

        if 'sdm' in self.current_task and batch['pair_img'] is None:
            ret.update({'sdm_loss':objectives.compute_sdm(i_feats, t_feats, batch['pids'], logit_scale)})
        
        if 'bisdm' in self.current_task and batch['pair_img'] is not None:
            ret.update({'bisdm_loss':(objectives.compute_sdm(i_feats, t_feats, batch['pids'], logit_scale)+\
                                        objectives.compute_sdm(y_feats, t_feats, batch['pids'], logit_scale))/2})

        if 'sdm' in self.current_task and batch['pair_img'] is not None:
            ret.update({'sdm_loss':(objectives.compute_sdm(i_feats, t_feats, batch['pids'], logit_scale)+\
                                    objectives.compute_sdm(y_feats, t_feats, batch['pids'], logit_scale)+\
                                        objectives.compute_sdm(i_feats, y_feats, batch['pids'], logit_scale))/3})
        
        if 'mlm' in self.current_task and batch['pair_img'] is None:
            mlm_ids = batch['mlm_ids']

            mlm_feats = self.base_model.encode_text(mlm_ids)

            x = self.cross_former(mlm_feats, image_feats, image_feats)

            x = self.mlm_head(x)  # [batch_size, text_len, num_colors]

            scores = x.float().reshape(-1, self.args.vocab_size)
            mlm_labels = batch['mlm_labels'].reshape(-1)
            ret.update({'mlm_loss': objectives.compute_mlm(scores, mlm_labels)*self.args.mlm_loss_weight})

            pred = scores.max(1)[1]
            mlm_label_idx = torch.nonzero(mlm_labels)
            acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()
            ret.update({'mlm_acc': acc})

        return ret


def build_model(args, num_classes=11003):
    model = IRRA(args, 11003)
    # covert model to fp16
    convert_weights(model)
    if "prot" in args.loss_names:
        model.prototypes.weight.data = model.prototypes.weight.data.float()
        if model.queue is not None:
            model.queue = model.queue.float()
    #print(model.fusion_output_ff.LayerNorm.weight.dtype)
    return model
