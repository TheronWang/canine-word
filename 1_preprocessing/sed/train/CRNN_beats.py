import torch.nn as nn
import torch
from .beats.beats_model import BEATs_model
from .RNN import BidirectionalGRU
from .CNN import CNN

class CRNN(nn.Module):
    def __init__(
        self,
        unfreeze_atst_layer=0,
        n_in_channel=1,
        nclass=10,
        activation="glu",
        dropout=0.5,
        rnn_type="BGRU",
        n_RNN_cell=128,
        n_layers_RNN=2,
        dropout_recurrent=0,
        embedding_size=768,
        model_init=None,
        beats_init=None,
        mode=None,
        **kwargs,
    ):
        super(CRNN, self).__init__()
        self.nclass = nclass
        self.n_RNN_cell = n_RNN_cell

        self.n_in_channel = n_in_channel
        n_in_cnn = n_in_channel
        self.cnn = CNN(
            n_in_channel=n_in_cnn, activation=activation, conv_dropout=dropout, **kwargs
        )

        if rnn_type == "BGRU":
            nb_in = self.cnn.nb_filters[-1]
            nb_in = nb_in * n_in_channel
            self.rnn = BidirectionalGRU(
                n_in=nb_in,
                n_hidden=n_RNN_cell,
                dropout=dropout_recurrent,
                num_layers=n_layers_RNN,
            )
        else:
            NotImplementedError("Only BGRU supported for CRNN for now")

        self.dropout = nn.Dropout(dropout)
        # TODO
        self.dense = nn.Linear(n_RNN_cell * 2, 1)
        self.sigmoid = nn.Sigmoid()

        # TODO
        self.dense_softmax = nn.Linear(n_RNN_cell * 2, 1)
        self.softmax = nn.Softmax(dim=-1)

        self.nb_in = nb_in
        self.cat_tf = torch.nn.Linear(nb_in+embedding_size, nb_in)
        
        self.init_beats(path=beats_init)
        self.init_model(model_init, mode=mode)
        
        self.unfreeze_atst_layer = unfreeze_atst_layer

    def init_beats(self, path=None):
        self.BEATs_model = BEATs_model(path)
        self.BEATs_model.eval()
    
    def init_model(self, path, mode=None):
        if path is None:
            self.dense = nn.Linear(self.n_RNN_cell * 2, self.nclass)
            self.dense_softmax = nn.Linear(self.n_RNN_cell * 2, self.nclass)
            self.cat_tf = torch.nn.Linear(self.nb_in + 768, self.nb_in)
            torch.nn.init.xavier_uniform_(self.dense.weight)
            torch.nn.init.xavier_uniform_(self.dense_softmax.weight)
            torch.nn.init.constant_(self.dense.bias, 0)
            torch.nn.init.constant_(self.dense_softmax.bias, 0)
            torch.nn.init.xavier_uniform_(self.cat_tf.weight)
            torch.nn.init.constant_(self.cat_tf.bias, 0)

            print('新参数')
        else:
            if mode == "teacher":
                print("Loading teacher from:", path)
                state_dict = torch.load(path, map_location="cpu")["sed_teacher"]
            else:
                print("Loading student from:", path)
                state_dict = torch.load(path, map_location="cpu")["sed_student"]
            self.load_state_dict(state_dict, strict=True)
            print("Model loaded")

    def forward(self, x, pretrain_x, pad_mask=None, embeddings=None):
        x = x.transpose(1, 2).unsqueeze(1)
        # conv features
        x = self.cnn(x)
        bs, chan, frames, freq = x.size()
        x = x.squeeze(-1)
        x = x.permute(0, 2, 1)  # [bs, frames, chan]
        
        # rnn features
        embeddings = self.BEATs_model(pretrain_x)
        torch.use_deterministic_algorithms(True, warn_only=True)
        embeddings = torch.nn.functional.adaptive_avg_pool1d(embeddings.transpose(-1, -2), 156).transpose(-1, -2)
        x = self.cat_tf(torch.cat((x, embeddings), -1))
        
        x = self.rnn(x)
        x = self.dropout(x)
        strong = self.dense(x)  # [bs, frames, nclass]
        strong = self.sigmoid(strong)
        sof = self.dense_softmax(x)  # [bs, frames, nclass]
        sof = self.softmax(sof)
        sof = torch.clamp(sof, min=1e-7, max=1)
        weak = (strong * sof).sum(1) / sof.sum(1)  # [bs, nclass]

        return strong.transpose(1, 2), weak

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super(CRNN, self).train(mode)

        # we comments this part because there are only two stage in the fine-tuning;
        # either eval the ATST-Frame or train the whole model
        # if mode:
        #     self.atst_frame.eval()
        #     if self.unfreeze_atst_layer > 0:
        #         self.atst_frame.atst.norm_frame.train()
        #         if self.unfreeze_atst_layer == 14:
        #             self.atst_frame.train()
        #         else:
        #             unfreeze_blocks = self.unfreeze_atst_layer - 1
        #             for i in range(unfreeze_blocks):
        #                 block_idx = 11 - i
        #                 self.atst_frame.atst.blocks[block_idx].train()
