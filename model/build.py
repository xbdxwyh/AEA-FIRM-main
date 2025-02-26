from model import objectives
from .clip_model import (
    Transformer, 
    QuickGELU, 
    LayerNorm, 
    build_CLIP_from_openai_pretrained, 
    convert_weights, 
    VisionTransformer, 
)

import torch
import torch.nn as nn
from collections import OrderedDict
import random
import torch.nn.functional as F


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


class IRRA(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        #self.base_model, base_cfg, self.pe_enlarge = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim']

        self.logit_scale = torch.ones([]) * (1 / args.temperature) 
        self.proj_dec_cfg = None
        
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

        if 'id' in args.loss_names:
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)

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

    def encode_image(self, image):
        x = self.base_model.encode_image(image)
        if isinstance(self.base_model.visual, VisionTransformer):
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
