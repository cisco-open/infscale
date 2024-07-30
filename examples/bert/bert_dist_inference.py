import sys, os, time, json, yaml
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.distributed.rpc import RRef
import torch.distributed.rpc as rpc
import threading
from typing import Iterator, List, Optional, Tuple, Union
from functools import reduce
from torch.nn.parameter import Parameter

from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from transformers.models.bert.modeling_bert import (
    BertEmbeddings,
    BertPreTrainedModel,
    BertPooler,
    BertEncoder,
    BertLayer,
    BertConfig,
)
from transformers.models.bert.modeling_bert import (
    BaseModelOutputWithPoolingAndCrossAttentions,
)
from transformers.modeling_utils import ModuleUtilsMixin

sys.path.append(".")
from inference_pipeline import (
    list2csvcell,
    RR_TransformerPipeline,
    TransformerShardBase,
)

dtype2int_map = {
    torch.float: 0,
    torch.double: 1,
    torch.int8: 2,
    torch.int16: 3,
    torch.int32: 4,
    torch.int64: 5,
}
int2dtype_map = [
    torch.float,
    torch.double,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
]


def encode_tensors(
    tensors: Union[List[torch.Tensor], Tuple[torch.Tensor]]
) -> torch.FloatTensor:
    tensor_num = len(tensors)
    ttype = torch.float

    # the first cell bears the number of tensors
    res = torch.tensor([tensor_num], dtype=ttype)
    for t in tensors:
        if isinstance(t, torch.Tensor):
            ts = list(t.shape)
            td = len(ts)
            tdtype = dtype2int_map[t.dtype]
            res = torch.cat((res, torch.tensor([tdtype, td] + ts, dtype=ttype)))
            res = torch.cat((res, t.to(ttype).view(-1)))
        else:
            res = torch.cat((res, torch.tensor([-1], dtype=ttype)))

    return res


def decode_tensors(
    encoded_data: torch.FloatTensor, logFile=sys.stderr
) -> List[torch.Tensor]:
    tensor_num = int(encoded_data[0])

    # decode tensor data from the second cell
    datap = 1
    res = []
    for i in range(tensor_num):
        tensor_dtype_index = int(encoded_data[datap])
        if tensor_dtype_index == -1:
            datap += 1
            res.append(None)
            continue
        tensor_dtype = int2dtype_map[tensor_dtype_index]

        tensor_dim = int(encoded_data[datap + 1])
        tensor_shape = encoded_data[datap + 2 : datap + (2 + tensor_dim)].int().tolist()
        datap += 2 + tensor_dim
        tensor_size = reduce(lambda x, y: x * y, tensor_shape)
        tensor = (
            encoded_data[datap : datap + tensor_size]
            .view(tensor_shape)
            .to(tensor_dtype)
        )
        datap += tensor_size
        res.append(tensor)

    return res


def bert_input_constructor(batch_size, seq_len, tokenizer, batch_num=1):
    fake_seq = ""
    for _ in range(seq_len - 2):  # ignore the two special tokens [CLS] and [SEP]
        fake_seq += tokenizer.pad_token
    inputs = tokenizer(
        [fake_seq] * (batch_size * batch_num),
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    inputs = dict(inputs)
    return inputs


#########################################################
#           Define Model Parallel Bert                  #
#########################################################

# Need to pass a tuple that contains inputs useful for *all* layers through every layer
# Input tuple definition as follows:
#   0: input_ids
#   1: token_type_ids
#   2: position_ids
#   3: input_embeds
#   4: hidden_states
#   5: pooled_output
#   6: attention_mask
#   7: head_mask
#   8: encoder_hidden_states
#   9: encoder_attention_mask
#   10: all_hidden_states
#   11: all_self_attentions
#   12: all_cross_attentions


class BertPreprocessLayer(nn.Module):
    def __init__(self, config, dtype=torch.float32, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = config
        self.dtype = dtype

    def parameter_rrefs(self):
        r"""
        Create one RRef for each parameter in the given local module, and return a
        list of RRefs.
        """
        return []

    def get_extended_attention_mask(
        self,
        attention_mask,
        input_shape: Tuple[int],
        device: torch.device = None,
        dtype: torch.float = None,
    ):
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (`Tuple[int]`):
                The shape of the input to the model.

        Returns:
            `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
        """
        if dtype is None:
            dtype = self.dtype

        if not (attention_mask.dim() == 2 and self.config.is_decoder):
            # show warning only if it won't be shown in `create_extended_attention_mask_for_decoder`
            if device is not None:
                pass
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                extended_attention_mask = (
                    ModuleUtilsMixin.create_extended_attention_mask_for_decoder(
                        input_shape, attention_mask, device
                    )
                )
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and the dtype's smallest value for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(
            dtype
        ).min
        return extended_attention_mask

    def forward(self, inputs):
        input_ids = inputs[0] if inputs[0] != None else None
        token_type_ids = inputs[1] if inputs[1] != None else None
        inputs_embeds = inputs[3] if inputs[3] != None else None
        attention_mask = inputs[6] if inputs[6] != None else None
        head_mask = inputs[7] if inputs[7] != None else None

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device="cpu")

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device="cpu")

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape
        )

        inputs[1] = token_type_ids
        inputs[6] = extended_attention_mask

        return inputs


class BertEmbeddingLayer(nn.Module):
    def __init__(self, config, embedding=None, device="cpu", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = config
        self.device = device

        if embedding == None:
            self.embedding_module = BertEmbeddings(config).to(device)
        else:
            self.embedding_module = embedding.to(device)

    def parameter_rrefs(self):
        r"""
        Create one RRef for each parameter in the given local module, and return a
        list of RRefs.
        """
        return [RRef(p.cpu()) for p in self.embedding_module.parameters()]

    def parameters(self, recurse: bool = True):
        return self.embedding_module.parameters(recurse)

    def to(self, device):
        self.device = device
        self.embedding_module = self.embedding_module.to(device)
        return self

    def forward(self, inputs):
        input_ids = inputs[0].to(self.device) if inputs[0] != None else None
        token_type_ids = inputs[1].to(self.device) if inputs[1] != None else None
        position_ids = inputs[2].to(self.device) if inputs[2] != None else None
        inputs_embeds = inputs[3].to(self.device) if inputs[3] != None else None

        embeddings = self.embedding_module.forward(
            input_ids, token_type_ids, position_ids, inputs_embeds
        )

        inputs[4] = embeddings
        return inputs


class BertPoolerLayer(nn.Module):
    def __init__(self, config, pooler=None, device="cpu", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if pooler == None:
            self.pooler = BertPooler(config).to(device)
        else:
            self.pooler = pooler.to(device)
        self.device = device

    def parameter_rrefs(self):
        r"""
        Create one RRef for each parameter in the given local module, and return a
        list of RRefs.
        """
        return [RRef(p.cpu()) for p in self.pooler.parameters()]

    def parameters(self, recurse: bool = True):
        return self.pooler.parameters(recurse)

    def forward(self, inputs) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        hidden_states = inputs[4].to(self.device) if inputs[4] != None else None
        pooled_output = self.pooler(hidden_states)
        inputs[5] = pooled_output
        return inputs

    def to(self, device):
        self.device = device
        self.pooler = self.pooler.to(device)
        return self


class BertEncoderLayer(nn.Module):
    def __init__(
        self, config, index, encoder=None, device="cpu", *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        # construct encoders
        self.config = config
        self.device = device
        if encoder == None:
            self.layer = BertLayer(config).to(self.device)
        else:
            self.layer = encoder
        self.index = index

    def parameter_rrefs(self):
        r"""
        Create one RRef for each parameter in the given local module, and return a
        list of RRefs.
        """
        return [RRef(p.cpu()) for p in self.layer.parameters()]

    def parameters(self, recurse: bool = True):
        return self.layer.parameters(recurse)

    def to(self, device):
        self.device = device
        self.layer = self.layer.to(device)
        return self

    def forward(self, inputs):
        hidden_states = inputs[4].to(self.device) if inputs[4] != None else None
        attention_mask = inputs[6].to(self.device) if inputs[6] != None else None
        head_mask = inputs[7] if inputs[7] != None else None
        layer_head_mask = (
            head_mask[self.index].to(self.device) if head_mask != None else None
        )
        encoder_hidden_states = inputs[8].to(self.device) if inputs[8] != None else None
        encoder_attention_mask = (
            inputs[9].to(self.device) if inputs[9] != None else None
        )
        all_hidden_states = inputs[10] if inputs[10] != None else None
        all_self_attentions = inputs[11] if inputs[11] != None else None
        all_cross_attentions = inputs[12] if inputs[12] != None else None

        if self.index == 0 and all_hidden_states != None:
            all_hidden_states = all_hidden_states + (inputs[4],)

        layer_outputs = self.layer(
            hidden_states,
            attention_mask,
            layer_head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            output_attentions=True,
        )

        hidden_states = layer_outputs[0]
        inputs[4] = hidden_states
        if all_hidden_states != None:
            all_hidden_states = all_hidden_states + (hidden_states.cpu(),)
            inputs[10] = all_hidden_states
        if all_self_attentions != None:
            all_self_attentions = all_self_attentions + (layer_outputs[1].cpu(),)
            inputs[11] = all_self_attentions
        if self.config.add_cross_attention and all_cross_attentions != None:
            all_cross_attentions = all_cross_attentions + (layer_outputs[2].cpu(),)
            inputs[12] = all_cross_attentions

        return inputs


class RRDistBertModel(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(
        self,
        config,
        split_size,
        workers,
        layers,
        partitions,
        shards,
        devices,
        *args,
        **kwargs,
    ):
        super().__init__(config)
        self.config = config
        self.split_size = split_size

        assert len(workers) >= len(shards)
        assert len(workers) <= len(devices)

        layer_partitions = []
        partitions = [0] + partitions + [len(layers)]
        for i in range(len(partitions) - 1):
            if len(layers[partitions[i] : partitions[i + 1]]) > 0:
                layer_partitions.append(layers[partitions[i] : partitions[i + 1]])

        self.shards_ref = [[] for i in range(len(layer_partitions))]
        # place shards according to configuration
        for i in range(len(shards)):
            shard_id = shards[i] - 1
            shard_layers = layer_partitions[shard_id]
            print("--------------------------------")
            print(
                "Starting {} with shard_id:{}".format(workers[i], shard_id), flush=True
            )
            rref = rpc.remote(
                workers[i],
                TransformerShardBase,
                args=(devices[i], shard_layers, i) + args,
                kwargs=kwargs,
            )
            self.shards_ref[shard_id].append(rref)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.Tensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length, always 0 because past_key_values is disabled
        past_key_values_length = 0

        if attention_mask is None:
            attention_mask = torch.ones(
                ((batch_size, seq_length + past_key_values_length)), device=device
            )

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, dtype=torch.float
        )

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            (
                encoder_batch_size,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask
            )
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        num_stages = len(self.shards_ref)
        spin_pointers = [0] * num_stages  # spin pointers
        out_futures = []
        for p in range(0, batch_size, self.split_size):
            # process one micro-batch at an iteration
            # split the input along the dimension
            input_ids_split = input_ids[p : p + self.split_size]
            token_type_ids_split = token_type_ids[p : p + self.split_size]
            position_ids_split = (
                position_ids[p : p + self.split_size]
                if position_ids is not None
                else None
            )
            inputs_embeds_split = (
                inputs_embeds[p : p + self.split_size]
                if inputs_embeds is not None
                else None
            )
            extended_attention_mask_split = extended_attention_mask[
                p : p + self.split_size
            ]
            head_mask_split = (
                head_mask[:][p : p + self.split_size] if head_mask is not None else None
            )
            encoder_hidden_states_split = (
                encoder_hidden_states[p : p + self.split_size]
                if encoder_hidden_states is not None
                else None
            )
            encoder_extended_attention_mask_split = (
                encoder_extended_attention_mask[p : p + self.split_size]
                if encoder_extended_attention_mask is not None
                else None
            )

            inputs = (
                input_ids_split,
                token_type_ids_split,
                position_ids_split,
                inputs_embeds_split,
                None,
                None,
                extended_attention_mask_split,
                head_mask_split,
                encoder_hidden_states_split,
                encoder_extended_attention_mask_split,
                None,
                None,
                None,
            )

            temp = encode_tensors(inputs)
            x_rref = RRef(temp)

            for i in range(len(self.shards_ref) - 1):
                x_rref = self.shards_ref[i][spin_pointers[i]].remote().forward(x_rref)
                spin_pointers[i] = (spin_pointers[i] + 1) % len(self.shards_ref[i])

            i = -1
            z_fut = self.shards_ref[i][spin_pointers[i]].rpc_async().forward(x_rref)
            spin_pointers[i] = (spin_pointers[i] + 1) % len(self.shards_ref[i])
            out_futures.append(z_fut)

        def merge_tupled_tensors(tlist: list):
            # tlist is a list of tuples of tensors that have the same number of cells
            out = tuple()
            for i in range(len(tlist[0])):
                out = out + (torch.cat([x[i] for x in tlist]),)
            return out

        output_list = torch.futures.wait_all(out_futures)
        output_list = [decode_tensors(t) for t in output_list]
        outputs = (
            torch.cat([x[4] for x in output_list]),  # last hidden states
            torch.cat([x[5] for x in output_list])
            if output_list[0][5] != None
            else None,  # pooled hidden states
            merge_tupled_tensors([x[10] for x in output_list])
            if output_list[0][10] != None
            else None,  # all hidden states
            merge_tupled_tensors([x[11] for x in output_list])
            if output_list[0][11] != None
            else None,  # attentions
            merge_tupled_tensors([x[12] for x in output_list])
            if output_list[0][12] != None
            else None,  # cross attentions
        )

        if not return_dict:
            return outputs

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=outputs[0],
            pooler_output=outputs[1],
            hidden_states=outputs[2],
            attentions=outputs[3],
            cross_attentions=outputs[4],
        )

    def parameter_list(self):
        res = []
        for i in range(len(self.shards_ref)):
            res += [
                p.to_here() for p in self.shards_ref[i][0].rpc_sync().parameter_rrefs()
            ]

        return res

    def verify_parameter_consistency(self):
        for i in range(len(self.shards_ref)):
            for j in range(1, len(self.shards_ref[i])):
                base = [
                    p.to_here()
                    for p in self.shards_ref[i][0].rpc_sync().parameter_rrefs()
                ]
                cmp = [
                    p.to_here()
                    for p in self.shards_ref[i][j].rpc_sync().parameter_rrefs()
                ]

                for k in range(len(base)):
                    if torch.all(base[k] != cmp[k]):
                        return False

        return True


#########################################################
#                   Run RPC Processes                   #
#########################################################

num_batches = 100
batch_size = 24
seq_len = 128


def run_leader(
    inputs,
    split_size,
    num_workers,
    partitions,
    shards,
    devices,
    pre_trained=False,
    logging=False,
):

    config = BertConfig()
    if pre_trained == True:
        net = BertModel.from_pretrained("bert-base-uncased")
    else:
        net = BertModel(config)
    net.eval()

    layers = [
        BertPreprocessLayer(config),
        BertEmbeddingLayer(config, embedding=net.embeddings),
        *[
            BertEncoderLayer(config, i, encoder=net.encoder.layer[i])
            for i in range(config.num_hidden_layers)
        ],
        BertPoolerLayer(config, pooler=net.pooler),
    ]

    device_list = devices

    if len(shards) == 0:
        # no partitioning
        model = net.to("cuda:0")
    else:
        model = RR_TransformerPipeline(
            config,
            split_size,
            ["worker{}".format(i + 1) for i in range(num_workers)],
            layers,
            partitions,
            shards,
            devices=device_list + device_list,
        )

        if not model.verify_parameter_consistency():
            print("Parameters not consistent!", flush=True)
            return

    file = open("./bert.csv", "a")
    original_stdout = sys.stdout
    sys.stdout = file

    print("{}".format(list2csvcell(shards)), end=", ", flush=True)
    tik = time.time()
    for i in range(num_batches):
        batch = dict()
        if len(shards) == 0:
            for k in inputs:
                batch[k] = inputs[k].to("cuda:0")
        else:
            batch = inputs

        outputs = model(**batch)

    tok = time.time()
    print(f"{split_size}, {tok - tik}, {(num_batches * batch_size) / (tok - tik)}")

    sys.stdout = original_stdout


def run_worker(
    rank,
    world_size,
    inputs,
    split_size,
    partitions,
    placement,
    devices,
    pre_trained=False,
    logging=False,
):
    os.environ["LEADER_ADDR"] = "localhost"
    os.environ["LEADER_PORT"] = "29500"

    # Higher timeout is added to accommodate for kernel compilation time in case of ROCm.
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=256, rpc_timeout=300)

    if rank == 0:
        rpc.init_rpc(
            "leader", rank=rank, world_size=world_size, rpc_backend_options=options
        )
        run_leader(
            inputs,
            split_size,
            num_workers=world_size - 1,
            partitions=partitions,
            shards=placement,
            devices=devices,
            pre_trained=pre_trained,
            logging=logging,
        )
    else:
        rpc.init_rpc(
            f"worker{rank}",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options,
        )
        pass

    # block until all rpcs finish
    rpc.shutdown()


if __name__ == "__main__":
    with open("bert_config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)
        file = open("./bert.log", "w")
        open("./bert.csv", "w")  # flush the csv file
        original_stdout = sys.stdout
        sys.stdout = file

        partitions = config["partitions"]
        devices = config["devices"]
        repeat_times = config["repeat_times"]
        placements = config["placements"]
        split_size = config["micro_batch_size"]
        pre_trained = (
            config["pre_trained"] == "True" if "pre_trained" in config else False
        )
        logging = config["logging"] == "True" if "logging" in config else False
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        for shards in placements:
            world_size = len(shards) + 1
            print("Placement:", shards)
            for i in range(repeat_times):
                # generate input
                inputs = bert_input_constructor(batch_size, seq_len, tokenizer)

                # run inference process
                tik = time.time()
                mp.spawn(
                    run_worker,
                    args=(
                        world_size,
                        inputs,
                        split_size,
                        partitions,
                        shards,
                        devices,
                        pre_trained,
                        logging,
                    ),
                    nprocs=world_size,
                    join=True,
                )
                tok = time.time()
                print(
                    f"size of micro-batches = {split_size}, end-to-end execution time = {tok - tik} s"
                )

        sys.stdout = original_stdout
