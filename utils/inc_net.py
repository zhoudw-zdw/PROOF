import copy
import logging
import torch
from torch import nn
from convs.linears import SimpleLinear, SplitCosineLinear, CosineLinear
import timm
import torch.nn.functional as F
from convs.projections import Proj_Pure_MLP, MultiHeadAttention
from utils.toolkit import get_attribute

def get_convnet(args, pretrained=False):

    backbone_name = args["convnet_type"].lower()
    algorithm_name = args["model_name"].lower()
    if 'clip' in backbone_name:
        print('Using CLIP model as the backbone')
        import open_clip
        if backbone_name == 'clip':
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion400m_e32')
            tokenizer = open_clip.get_tokenizer('ViT-B-16')
            model.out_dim = 512
            return model, preprocess, tokenizer
        elif backbone_name=='clip_laion2b':
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k')
            tokenizer = open_clip.get_tokenizer('ViT-B-16')
            model.out_dim = 512
            return model, preprocess, tokenizer
        elif backbone_name=='openai_clip':
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai')
            tokenizer = open_clip.get_tokenizer('ViT-B-16')
            model.out_dim = 512
            return model, preprocess, tokenizer
        else:
            raise NotImplementedError("Unknown type {}".format(backbone_name))
    
    else:
        raise NotImplementedError("Unknown type {}".format(backbone_name))


class BaseNet(nn.Module):
    def __init__(self, args, pretrained):
        super(BaseNet, self).__init__()

        self.convnet = get_convnet(args, pretrained)
        self.fc = None

    @property
    def feature_dim(self):
        return self.convnet.out_dim

    def extract_vector(self, x):
        return self.convnet(x)["features"]

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])
        """
        {
            'fmaps': [x_1, x_2, ..., x_n],
            'features': features
            'logits': logits
        }
        """
        out.update(x)
        return out

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self


class IncrementalNet(BaseNet):
    def __init__(self, args, pretrained, gradcam=False):
        super().__init__(args, pretrained)
        self.gradcam = gradcam
        if hasattr(self, "gradcam") and self.gradcam:
            self._gradcam_hooks = [None, None]
            self.set_gradcam_hook()

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print("alignweights,gamma=", gamma)
        self.fc.weight.data[-increment:, :] *= gamma

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)

        return fc

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])
        out.update(x)
        if hasattr(self, "gradcam") and self.gradcam:
            out["gradcam_gradients"] = self._gradcam_gradients
            out["gradcam_activations"] = self._gradcam_activations

        return out

    def unset_gradcam_hook(self):
        self._gradcam_hooks[0].remove()
        self._gradcam_hooks[1].remove()
        self._gradcam_hooks[0] = None
        self._gradcam_hooks[1] = None
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

    def set_gradcam_hook(self):
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

        def backward_hook(module, grad_input, grad_output):
            self._gradcam_gradients[0] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self._gradcam_activations[0] = output
            return None

        self._gradcam_hooks[0] = self.convnet.last_conv.register_backward_hook(
            backward_hook
        )
        self._gradcam_hooks[1] = self.convnet.last_conv.register_forward_hook(
            forward_hook
        )



class CosineIncrementalNet(BaseNet):
    def __init__(self, args, pretrained, nb_proxy=1):
        super().__init__(args, pretrained)
        self.nb_proxy = nb_proxy

    def update_fc(self, nb_classes, task_num):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            if task_num == 1:
                fc.fc1.weight.data = self.fc.weight.data
                fc.sigma.data = self.fc.sigma.data
            else:
                prev_out_features1 = self.fc.fc1.out_features
                fc.fc1.weight.data[:prev_out_features1] = self.fc.fc1.weight.data
                fc.fc1.weight.data[prev_out_features1:] = self.fc.fc2.weight.data
                fc.sigma.data = self.fc.sigma.data

        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        if self.fc is None:
            fc = CosineLinear(in_dim, out_dim, self.nb_proxy, to_reduce=True)
        else:
            prev_out_features = self.fc.out_features // self.nb_proxy
            # prev_out_features = self.fc.out_features
            fc = SplitCosineLinear(
                in_dim, prev_out_features, out_dim - prev_out_features, self.nb_proxy
            )

        return fc


class BiasLayer(nn.Module):
    def __init__(self):
        super(BiasLayer, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, requires_grad=True))
        self.beta = nn.Parameter(torch.zeros(1, requires_grad=True))

    def forward(self, x, low_range, high_range):
        ret_x = x.clone()
        ret_x[:, low_range:high_range] = (
            self.alpha * x[:, low_range:high_range] + self.beta
        )
        return ret_x

    def get_params(self):
        return (self.alpha.item(), self.beta.item())


class IncrementalNetWithBias(BaseNet):
    def __init__(self, args, pretrained, bias_correction=False):
        super().__init__(args, pretrained)

        # Bias layer
        self.bias_correction = bias_correction
        self.bias_layers = nn.ModuleList([])
        self.task_sizes = []

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])
        if self.bias_correction:
            logits = out["logits"]
            for i, layer in enumerate(self.bias_layers):
                logits = layer(
                    logits, sum(self.task_sizes[:i]), sum(self.task_sizes[: i + 1])
                )
            out["logits"] = logits

        out.update(x)

        return out

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        self.bias_layers.append(BiasLayer())

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)

        return fc

    def get_bias_params(self):
        params = []
        for layer in self.bias_layers:
            params.append(layer.get_params())

        return params

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True



class SimpleCosineIncrementalNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self.feature_dim, nb_classes).cuda()
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim).cuda()])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc


class SimpleVitNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)
        self.convnet, self.preprocess, self.tokenizer = get_convnet(args, pretrained)

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self.feature_dim, nb_classes).cuda()
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim).cuda()])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc

    def extract_vector(self, x):
        return self.convnet.encode_image(x)

    def encode_image(self, x):
        return self.convnet.encode_image(x)
    
    def encode_text(self, x):
        return self.convnet.encode_text(x)
        
    def forward(self, x):
        x = self.convnet.encode_image(x)
        out = self.fc(x)
        return out



class SimpleClipNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)

        self.convnet, self.preprocess, self.tokenizer = get_convnet(args, pretrained)
        self.class_name = 'SimpleClipNet'
        self.args = args


    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self.feature_dim, nb_classes).cuda()
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim).cuda()])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc

    def extract_vector(self, x):
        return self.convnet.encode_image(x)

    def encode_image(self, x):
        return self.convnet.encode_image(x)
    
    def encode_text(self, x):
        return self.convnet.encode_text(x)

    def forward(self, img, text):

        image_features, text_features, logit_scale=self.convnet(img, text)
        return image_features, text_features, logit_scale

    def re_initiate(self):
        print('re-initiate model')
        self.convnet, self.preprocess, self.tokenizer = get_convnet(self.args, True)


class Proof_Net(SimpleClipNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)
        self.projs_img = nn.ModuleList()
        self.projs_text = nn.ModuleList()
        self.args = args
        self._device = args["device"][0]
        self.projtype = get_attribute(self.args, 'projection_type', 'mlp')
        self.context_prompt_length_per_task = get_attribute(self.args, 'context_prompt_length_per_task', 3)
        
        self.sel_attn = MultiHeadAttention(1, self.feature_dim, self.feature_dim, self.feature_dim, dropout=0.1)
        self.img_prototypes = None

        self.context_prompts = nn.ParameterList()

    def update_prototype(self, nb_classes):
        if self.img_prototypes is not None:
            nb_output = len(self.img_prototypes)
            self.img_prototypes = torch.cat([copy.deepcopy(self.img_prototypes).to(self._device), torch.zeros(nb_classes - nb_output, self.feature_dim).to(self._device)]).to(self._device)
        else:
            self.img_prototypes = torch.zeros(nb_classes, self.feature_dim).to(self._device)
        print('update prototype, now we have {} prototypes'.format(self.img_prototypes.shape[0]))
    
    def update_context_prompt(self):
        for i in range(len(self.context_prompts)):
            self.context_prompts[i].requires_grad = False
        self.context_prompts.append(nn.Parameter(torch.randn(self.context_prompt_length_per_task, self.feature_dim).to(self._device)))
        print('update context prompt, now we have {} context prompts'.format(len(self.context_prompts) * self.context_prompt_length_per_task))
        self.context_prompts.to(self._device)
    
    def get_context_prompts(self):
        return torch.cat([item for item in self.context_prompts], dim=0)

    def encode_image(self, x, normalize: bool = False):
        x = x.to(self._device)
        basic_img_features = self.convnet.encode_image(x)
        img_features = [proj(basic_img_features) for proj in self.projs_img]
        img_features = torch.stack(img_features, dim=1)#[bs,num_proj,dim]
        image_feas = torch.sum(img_features, dim=1)#[bs,dim]
        return F.normalize(image_feas, dim=-1) if normalize else image_feas
        
    def encode_text(self, x, normalize: bool = False):
        x = x.to(self._device)
        basic_text_features = self.convnet.encode_text(x)
        text_features = [proj(basic_text_features) for proj in self.projs_text]
        text_features = torch.stack(text_features, dim=1)
        text_feas = torch.sum(text_features, dim=1) #[bs,dim]
        return F.normalize(text_feas, dim=-1) if normalize else text_feas
        
    def encode_prototpyes(self, normalize: bool = False):
        self.img_prototypes=self.img_prototypes.to(self._device)
        img_features = [proj(self.img_prototypes) for proj in self.projs_img]
        img_features=torch.stack(img_features, dim=1)#[nb_class,num_proj,dim]
        image_feas=torch.sum(img_features, dim=1)#[nb_class,dim]
        return F.normalize(image_feas, dim=-1) if normalize else image_feas

    def extend_task(self):
        self.projs_img.append(self.extend_item())
        self.projs_text.append(self.extend_item())

    def extend_item(self):
        if self.projtype=='pure_mlp':
            return Proj_Pure_MLP(self.feature_dim,self.feature_dim,self.feature_dim).to(self._device)
        else:
            raise NotImplementedError
    
    def forward(self, image, text):
        image_features = self.encode_image(image, normalize=True)#bs,dim
        text_features = self.encode_text(text, normalize=True)#bs,dim

        prototype_features = self.encode_prototpyes(normalize=True) #nb_class,dim
        context_prompts=self.get_context_prompts() # num_prompt, dim

        len_texts=text_features.shape[0]
        len_protos=prototype_features.shape[0]
        len_context_prompts=context_prompts.shape[0]
        # restack the features and pass them through the attention layer
        image_features = image_features.view(image_features.shape[0], -1, self.feature_dim)#bs,1,dim
        text_features = text_features.view(text_features.shape[0], self.feature_dim)#num_text,dim
        prototype_features = prototype_features.view(prototype_features.shape[0], self.feature_dim)#len_proto,dim
        context_prompts = context_prompts.view(context_prompts.shape[0], self.feature_dim)#len_con,dim
        # expand text features to be the same dim as image features
        text_features = text_features.expand(image_features.shape[0], text_features.shape[0], self.feature_dim)#bs,num_text,dim
        prototype_features = prototype_features.expand(image_features.shape[0], prototype_features.shape[0], self.feature_dim)#bs,len_proto,dim
        context_prompts = context_prompts.expand(image_features.shape[0], context_prompts.shape[0], self.feature_dim)#bs,len_con,dim
        # concat them together
        # features = torch.cat([image_features, text_features, prototype_features], dim=1) # bsize * (1+num_texts+num_protos) * dim
        features = torch.cat([image_features, text_features, prototype_features, context_prompts], dim=1) # bsize * (1+num_texts+num_protos+num_context) * dim
        # pass through the attention layer
        features = self.sel_attn(features, features, features)
        # split them back, image features are the first half, text features are the second half
        # image_features, text_features = torch.split(features, features.shape[1] // 2, dim=1)
        image_features = features[:, 0, :] # bsize * dim
        text_features = features[:, 1:len_texts+1, :] # bsize * num_texts * dim
        prototype_features = features[:, len_texts+1:len_texts+1+len_protos, :] # bsize * num_protos * dim 
        context_prompts = features[:, len_texts+1+len_protos:len_texts+1+len_protos+len_context_prompts, :] # bsize * num_context * dim
        # remove the 0-th dimension of text features to be num_texts * dim
        text_features = torch.mean(text_features, dim=0) # num_texts * dim
        prototype_features = torch.mean(prototype_features, dim=0) # num_protos * dim
        # squeeze
        image_features = image_features.view(image_features.shape[0], -1)
        text_features = text_features.view(text_features.shape[0], -1)
        prototype_features = prototype_features.view(prototype_features.shape[0], -1)
        return image_features, text_features, self.convnet.logit_scale.exp(), prototype_features
    
    def forward_transformer(self, image_features, text_features, transformer=False):
        prototype_features = self.encode_prototpyes(normalize=True)
        if transformer:
            context_prompts = self.get_context_prompts()
            len_texts = text_features.shape[0]
            len_protos = prototype_features.shape[0]
            len_context_prompts = context_prompts.shape[0]
            # restack the features and pass them through the attention layer
            image_features = image_features.view(image_features.shape[0], -1, self.feature_dim) #[bs, 1, dim]
            text_features = text_features.view(text_features.shape[0], self.feature_dim) #[total_classes, dim]
            prototype_features = prototype_features.view(prototype_features.shape[0], self.feature_dim) #[len_pro, dim]
            context_prompts = context_prompts.view(context_prompts.shape[0], self.feature_dim) #[len_con_pro, dim]
            # expand text features to be the same dim as image features
            text_features = text_features.expand(image_features.shape[0], text_features.shape[0], self.feature_dim) #[bs, total_classes, dim]
            prototype_features = prototype_features.expand(image_features.shape[0], prototype_features.shape[0], self.feature_dim) #[bs, len_pro, dim]
            context_prompts = context_prompts.expand(image_features.shape[0], context_prompts.shape[0], self.feature_dim) #[bs, len_con_pro, dim]
            # concat them together
            # features = torch.cat([image_features, text_features, prototype_features], dim=1) # bsize * (1+num_texts+num_protos) * dim
            features = torch.cat([image_features, text_features, prototype_features, context_prompts], dim=1) # bsize * (1+num_texts+num_protos+num_context) * dim
            # pass through the attention layer
            features = self.sel_attn(features, features, features)
            # split them back, image features are the first half, text features are the second half
            # image_features, text_features = torch.split(features, features.shape[1] // 2, dim=1)
            image_features = features[:, 0, :] # bsize * dim
            text_features = features[:, 1:len_texts+1, :] # bsize * num_texts * dim
            prototype_features = features[:, len_texts+1:len_texts+1+len_protos, :] # bsize * num_protos * dim 
            context_prompts = features[:, len_texts+1+len_protos:len_texts+1+len_protos+len_context_prompts, :] # bsize * num_context * dim
            # remove the 0-th dimension of text features to be num_texts * dim
            text_features = torch.mean(text_features, dim=0) # num_texts * dim
            prototype_features = torch.mean(prototype_features, dim=0) # num_protos * dim
            # squeeze
            image_features = image_features.view(image_features.shape[0], -1)
            text_features = text_features.view(text_features.shape[0], -1)
            prototype_features = prototype_features.view(prototype_features.shape[0], -1)
            return image_features, text_features, self.convnet.logit_scale.exp(), prototype_features
        else:
            return image_features, text_features, self.convnet.logit_scale.exp(), prototype_features
    
    
    def freeze_projection_weight_new(self):
        if len(self.projs_img)>1:
            for i in range(len(self.projs_img)):
                for param in self.projs_img[i].parameters():
                    param.requires_grad = False
                for param in self.projs_text[i].parameters():
                    param.requires_grad = True
            for param in self.projs_img[-1].parameters():
                param.requires_grad = True
        for param in self.sel_attn.parameters():
            param.requires_grad = True


