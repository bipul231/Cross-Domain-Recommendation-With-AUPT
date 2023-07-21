import torch
import torch.nn.functional as F

class LookupEmbedding(torch.nn.Module):

    def __init__(self, uid_all, iid_all, emb_dim):
        super().__init__()
        self.uid_embedding = torch.nn.Embedding(uid_all, emb_dim)
        self.iid_embedding = torch.nn.Embedding(iid_all + 1, emb_dim)

        #if self.uid_embedding.weight.size(1) != self.iid_embedding.weight.size(1):
            #self.iid_embedding = torch.nn.Embedding(iid_all + 1, self.uid_embedding.weight.size(1))

    def forward(self, x):

        #print("before gather uid_indices: ", x[:, 0].shape)
        #print("before gather iid_indices: ", x[:, 1].shape)
        uid_indices = x[:, 0]
        iid_indices = x[:, 1]
        uid_indices = torch.clamp(uid_indices, min=0, max=self.uid_embedding.num_embeddings-1)
        iid_indices = torch.clamp(iid_indices, min=0, max=self.iid_embedding.num_embeddings-1)
      
        #print("after clamp uid_indices: ", uid_indices.shape)
        #print("after clamp iid_indices: ", iid_indices.shape)
        uid_emb = torch.gather(self.uid_embedding.weight, 0, uid_indices.view(-1, 1)).squeeze(1)
        iid_emb = torch.gather(self.iid_embedding.weight, 0, iid_indices.view(-1, 1)).squeeze(1)

        if uid_emb.size() != iid_emb.size():
            print("uid_emb and iid_emb have different sizes")
        else:
            print("same size no issues")

        assert uid_emb.size(0) == iid_emb.size(0), f"Expected uid_emb and iid_emb to have the same number of rows but got {uid_emb.size(0)} and {iid_emb.size(0)}"
    

        #print("uid_emb: ", uid_emb.shape)
        #print("iid_emb: ", iid_emb.shape)
        #if uid_emb.size(1) > iid_emb.size(1) and iid_emb.size(1) != 1:

        if uid_emb.size(0) != iid_emb.size(0):
            diff = 0
        else:
            diff = uid_emb.size(0) - iid_emb.size(0)
    
        
        #if diff > 0:
            #uid_emb = uid_emb[:, :iid_emb.size(1), :]
        #elif diff < 0:
            #padding = torch.zeros(uid_emb.size(0), -diff, uid_emb.size(2))
            #uid_emb = torch.cat([uid_emb, padding], 1)
        #print(f'uid_emb shape: {uid_emb.shape}')
        #print(f'iid_emb shape: {iid_emb.shape}')
        emb = torch.stack([uid_emb, iid_emb], dim=1)
        #print("emb: ", emb.shape)
        return emb


#class LookupEmbedding(torch.nn.Module):

    #def __init__(self, uid_all, iid_all, emb_dim):
        #super().__init__()
        #self.uid_embedding = torch.nn.Embedding(uid_all, emb_dim)
        #self.iid_embedding = torch.nn.Embedding(iid_all + 1, emb_dim)

    #def forward(self, x):
        #print("Input shape: ", x.shape)
        #print("Input max index: ", x.max())
        #print("Number of uid embeddings: ", self.uid_embedding.num_embeddings)
        #print("Number of iid embeddings: ", self.iid_embedding.num_embeddings)
        #print("Input tensor range:", x.min(), x.max())
        #print("min index of input tensor:", torch.min(x))
        #print("max index of input tensor:", torch.max(x))
        #print("min index of uid embeddings:", torch.min(self.uid_embedding.weight))
        #print("max index of uid embeddings:", torch.max(self.uid_embedding.weight))
        #uid_indices = x[:, 0]
        #iid_indices = x[:, 1]
        #uid_indices = torch.clamp(uid_indices, min=0, max=self.uid_embedding.num_embeddings-1)
        #iid_indices = torch.clamp(iid_indices, min=0, max=self.iid_embedding.num_embeddings-1)
        #print(f'user indices: {uid_indices}')
        #print(f'item indices: {iid_indices}')
        #uid_emb = self.uid_embedding(uid_indices)
        #iid_emb = self.iid_embedding(iid_indices)
        #print(f'user embedding shape: {uid_emb.shape}')
        #print(f'item embedding shape: {iid_emb.shape}')
        ##uid_emb = self.uid_embedding(x[:, 0])
        ##iid_emb = self.iid_embedding(x[:, 1])
        #diff = uid_emb.size(1) - iid_emb.size(1)
        #if diff > 0:
            #uid_emb = uid_emb[:, :iid_emb.size(1), :]
        #elif diff < 0:
            #padding = torch.zeros(uid_emb.size(0), -diff, uid_emb.size(2))
            #uid_emb = torch.cat([uid_emb, padding], 1)
        #emb = torch.cat([uid_emb, iid_emb], dim=1)
        #return emb


class MetaNet(torch.nn.Module):
    def __init__(self, emb_dim, meta_dim):
        super().__init__()
        self.event_K = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.ReLU(),
                                           torch.nn.Linear(emb_dim, 1, False))
        self.event_softmax = torch.nn.Softmax(dim=1)
        self.decoder = torch.nn.Sequential(torch.nn.Linear(emb_dim, meta_dim), torch.nn.ReLU(),
                                           torch.nn.Linear(meta_dim, emb_dim * emb_dim))

    def forward(self, emb_fea, seq_index):
        mask = (seq_index == 0).float()
        event_K = self.event_K(emb_fea)
        t = event_K - torch.unsqueeze(mask, 2) * 1e8
        att = self.event_softmax(t)
        his_fea = torch.sum(att * emb_fea, 1)
        output = self.decoder(his_fea)
        return output.squeeze(1)


class GMFBase(torch.nn.Module):
    def __init__(self, uid_all, iid_all, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.embedding = LookupEmbedding(uid_all, iid_all, emb_dim)
        self.linear = torch.nn.Linear(emb_dim, 1, False)

    def forward(self, x):
        emb = self.embedding.forward(x)
        x = emb[:, 0] * emb[:, 1]
        x = self.linear(x)
        return x.squeeze(1)


class DNNBase(torch.nn.Module):
    def __init__(self, uid_all, iid_all, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.embedding = LookupEmbedding(uid_all, iid_all, emb_dim)
        self.linear = torch.nn.Linear(emb_dim, emb_dim)

    def forward(self, x):
        emb = self.embedding.forward(x)
        x = torch.sum(self.linear(emb[:, 0]) * emb[:, 1], 1)
        return x


class MFBasedModel(torch.nn.Module):
    def __init__(self, uid_all, iid_all, num_fields, emb_dim, meta_dim_0):
        super().__init__()
        self.num_fields = num_fields
        self.emb_dim = emb_dim
        self.src_model = LookupEmbedding(uid_all, iid_all, emb_dim)
        self.tgt_model = LookupEmbedding(uid_all, iid_all, emb_dim)
        self.aug_model = LookupEmbedding(uid_all, iid_all, emb_dim)
        self.meta_net = MetaNet(emb_dim, meta_dim_0)
        self.mapping = torch.nn.Linear(emb_dim, emb_dim, False)

    def forward(self, x, stage):
        if stage == 'train_src':
            emb = self.src_model.forward(x)
            x = torch.sum(emb[:, 0] * emb[:, 1], dim=-1)
            return x
        elif stage in ['train_tgt', 'test_tgt']:
            emb = self.tgt_model.forward(x)
            x = torch.sum(emb[:, 0] * emb[:, 1], dim=-1)
            return x
        elif stage in ['train_aug', 'test_aug']:
            emb = self.aug_model.forward(x)
            x = torch.sum(emb[:, 0] * emb[:, 1], dim=-1)
            return x
        elif stage in ['train_meta', 'test_meta']:
            iid_emb = self.tgt_model.iid_embedding(x[:, 1].unsqueeze(1))
            uid_emb_src = self.src_model.uid_embedding(x[:, 0].unsqueeze(1))
            ufea = self.src_model.iid_embedding(x[:, 2:])
            mapping = self.meta_net.forward(ufea, x[:, 2:]).view(-1, self.emb_dim, self.emb_dim)
            uid_emb = torch.bmm(uid_emb_src, mapping)
            emb = torch.cat([uid_emb, iid_emb], 1)
            output = torch.sum(emb[:, 0] * emb[:, 1], dim=-1)
            return output
        elif stage == 'train_map':
            src_emb = self.src_model.uid_embedding(x.unsqueeze(1)).squeeze()
            src_emb = self.mapping.forward(src_emb)
            tgt_emb = self.tgt_model.uid_embedding(x.unsqueeze(1)).squeeze()
            return src_emb, tgt_emb
        elif stage == 'test_map':
            uid_emb = self.mapping.forward(self.src_model.uid_embedding(x[:, 0].unsqueeze(1)).squeeze())
            emb = self.tgt_model.forward(x)
            n = emb[:, 0].shape[0]
            uid_emb = uid_emb[:n].reshape(n, -1)
            x = torch.sum(emb[:, 0] * emb[:, 1], dim=-1)
            return x


class GMFBasedModel(torch.nn.Module):
    def __init__(self, uid_all, iid_all, num_fields, emb_dim, meta_dim):
        super().__init__()
        self.num_fields = num_fields
        self.emb_dim = emb_dim
        self.src_model = GMFBase(uid_all, iid_all, emb_dim)
        self.tgt_model = GMFBase(uid_all, iid_all, emb_dim)
        self.aug_model = GMFBase(uid_all, iid_all, emb_dim)
        self.meta_net = MetaNet(emb_dim, meta_dim)
        self.mapping = torch.nn.Linear(emb_dim, emb_dim, False)

    def forward(self, x, stage):
        if stage == 'train_src':
            x = self.src_model.forward(x)
            return x
        elif stage in ['train_tgt', 'test_tgt']:
            x = self.tgt_model.forward(x)
            return x
        elif stage in ['train_aug', 'test_aug']:
            x = self.aug_model.forward(x)
            return x
        elif stage in ['test_meta', 'train_meta']:
            iid_emb = self.tgt_model.embedding.iid_embedding(x[:, 1].unsqueeze(1))
            uid_emb_src = self.src_model.embedding.uid_embedding(x[:, 0].unsqueeze(1))
            ufea = self.src_model.embedding.iid_embedding(x[:, 2:])
            mapping = self.meta_net.forward(ufea, x[:, 2:]).view(-1, self.emb_dim, self.emb_dim)
            uid_emb = torch.bmm(uid_emb_src, mapping)
            emb = torch.cat([uid_emb, iid_emb], 1)
            output = self.tgt_model.linear(emb[:, 0] * emb[:, 1])
            return output.squeeze(1)
        elif stage == 'train_map':
            src_emb = self.src_model.embedding.uid_embedding(x.unsqueeze(1)).squeeze()
            src_emb = self.mapping.forward(src_emb)
            tgt_emb = self.tgt_model.embedding.uid_embedding(x.unsqueeze(1)).squeeze()
            return src_emb, tgt_emb
        elif stage == 'test_map':
            uid_emb = self.mapping.forward(self.src_model.embedding.uid_embedding(x[:, 0].unsqueeze(1)))
            iid_emb = self.tgt_model.embedding.iid_embedding(x[:, 1].unsqueeze(1))
            emb = torch.cat([uid_emb, iid_emb], 1)
            x = self.tgt_model.linear(emb[:, 0] * emb[:, 1])
            return x.squeeze(1)


class DNNBasedModel(torch.nn.Module):
    def __init__(self, uid_all, iid_all, num_fields, emb_dim, meta_dim):
        super().__init__()
        self.num_fields = num_fields
        self.emb_dim = emb_dim
        self.src_model = DNNBase(uid_all, iid_all, emb_dim)
        self.tgt_model = DNNBase(uid_all, iid_all, emb_dim)
        self.aug_model = DNNBase(uid_all, iid_all, emb_dim)
        self.meta_net = MetaNet(emb_dim, meta_dim)
        self.mapping = torch.nn.Linear(emb_dim, emb_dim, False)

    def forward(self, x, stage):
        if stage == 'train_src':
            x = self.src_model.forward(x)
            return x
        elif stage in ['train_tgt', 'test_tgt']:
            x = self.tgt_model.forward(x)
            return x
        elif stage in ['train_aug', 'test_aug']:
            x = self.aug_model.forward(x)
            return x
        elif stage in ['test_meta', 'train_meta']:
            iid_emb = self.tgt_model.embedding.iid_embedding(x[:, 1].unsqueeze(1))
            uid_emb_src = self.src_model.linear(self.src_model.embedding.uid_embedding(x[:, 0].unsqueeze(1)))
            ufea = self.src_model.embedding.iid_embedding(x[:, 2:])
            mapping = self.meta_net.forward(ufea, x[:, 2:]).view(-1, self.emb_dim, self.emb_dim)
            uid_emb = torch.bmm(uid_emb_src, mapping)
            emb = torch.cat([uid_emb, iid_emb], 1)
            output = torch.sum(emb[:, 0] * emb[:, 1], 1)
            return output
        elif stage == 'train_map':
            src_emb = self.src_model.linear(self.src_model.embedding.uid_embedding(x.unsqueeze(1)).squeeze())
            src_emb = self.mapping.forward(src_emb)
            tgt_emb = self.tgt_model.linear(self.tgt_model.embedding.uid_embedding(x.unsqueeze(1)).squeeze())
            return src_emb, tgt_emb
        elif stage == 'test_map':
            uid_emb = self.mapping.forward(self.src_model.linear(self.src_model.embedding.uid_embedding(x[:, 0].unsqueeze(1))))
            iid_emb = self.tgt_model.embedding.iid_embedding(x[:, 1].unsqueeze(1))
            emb = torch.cat([uid_emb, iid_emb], 1)
            x = torch.sum(emb[:, 0] * emb[:, 1], 1)
            return x


