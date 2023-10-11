import sys
import threading
from functools import reduce
from typing import Callable, Iterator, List, Optional, Tuple, Union

import torch
import torch.distributed.rpc as rpc
import torch.nn as nn
from infscale import get_logger
from torch.distributed.rpc import RRef
from torch.nn.parameter import Parameter
from transformers import PreTrainedModel


class TransformerShardBase(nn.Module):
    def __init__(self, device, layers, *args, **kwargs):
        super().__init__()

        self.lock = threading.Lock()
        self.layers = [m.to(device) for m in layers]
        self.device = device

        self.shard_index = (
            kwargs["sid"] if "sid" in kwargs else None
        )  # index of the shard in the global set of shards
        self.partition_index = (
            kwargs["pid"] if "pid" in kwargs else None
        )  # index of the partition of layers beared by this shard

        logger_name = f"shard-sid{self.shard_index}-pid{self.partition_index}"
        self.logger = get_logger(logger_name)

    def forward(self, x_rref):
        x = x_rref.to_here()
        x = decode_tensors(x)
        with self.lock:
            for m in self.layers:
                x = m(x)

        for i in range(len(x)):
            if torch.is_tensor(x[i]):
                x[i] = x[i].to("cpu")

        x = encode_tensors(x)
        return x

    def parameter_rrefs(self):
        r"""
        Create one RRef for each parameter in the given local module, and return a
        list of RRefs.
        """
        res = []
        for l in self.layers:
            res += l.parameter_rrefs()
        return res

    def parameter_list(self):
        res = []
        for l in self.layers:
            res += l.parameter_list()
        return res


class RR_TransformerPipeline(PreTrainedModel):
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
        super().__init__(config, args, kwargs)
        self.config = config
        self.split_size = split_size
        self.output_handler = (
            kwargs["output_handler"] if "output_handler" in kwargs else None
        )

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
            partition_id = shards[i] - 1
            shard_layers = layer_partitions[partition_id]
            print("--------------------------------")
            print(
                "Starting {} with shard_id:{}".format(workers[i], partition_id),
                flush=True,
            )
            rref = rpc.remote(
                workers[i],
                TransformerShardBase,
                args=(devices[i], shard_layers, i) + args,
                kwargs=kwargs,
            )
            self.shards_ref[partition_id].append(rref)

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
        use_cache: Optional[bool] = None,
        output_handler: Optional[Callable] = None,
    ):
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

        num_stages = len(self.shards_ref)
        spin_pointers = [0] * num_stages  # spin pointers
        out_futures = []
        for p in range(0, batch_size, self.split_size):
            # process one micro-batch at an iteration
            # split the input along the dimension
            input_ids_split = input_ids[p : p + self.split_size]
            token_type_ids_split = (
                token_type_ids[p : p + self.split_size]
                if token_type_ids is not None
                else None
            )
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
            attention_mask_split = attention_mask[p : p + self.split_size]
            # TODO: How should we handle head_mask
            head_mask_split = (
                head_mask[:][p : p + self.split_size] if head_mask is not None else None
            )
            encoder_hidden_states_split = (
                encoder_hidden_states[p : p + self.split_size]
                if encoder_hidden_states is not None
                else None
            )
            encoder_attention_mask_split = (
                encoder_attention_mask[p : p + self.split_size]
                if encoder_attention_mask is not None
                else None
            )

            inputs = (
                input_ids_split,
                token_type_ids_split,
                position_ids_split,
                inputs_embeds_split,
                None,
                None,
                attention_mask_split,
                head_mask_split,
                encoder_hidden_states_split,
                encoder_attention_mask_split,
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

        output_list = torch.futures.wait_all(out_futures)
        output_list = [decode_tensors(t) for t in output_list]
        if output_handler is None and self.output_handler is None:
            # default handling process for generic transformer models that don't have task-specific heads
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
        elif output_handler != None:
            # customized handling process designed to incorporate flexible outputs from task-specific heads
            outputs = output_handler(output_list)
        else:
            # pre-defined customized handling process designed to incorporate flexible outputs from task-specific heads
            outputs = self.output_handler(output_list)

        return outputs

    def parameter_list(self):
        res = []
        for i in range(len(self.shards_ref)):
            res += [
                p.to_here() for p in self.shards_ref[i][0].rpc_sync().parameter_rrefs()
            ]

        return res

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return iter(self.parameter_list())

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

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs


###########################################################################################
# encode_tensor and decode_tensor are two functions used to handle the problem that PyTorch
# RPC can only transmit a single tensor while a list or a tuple of tensors need to be
# transmitted between shards in Transformer-based models.
#
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
###########################################################################################

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


def merge_tupled_tensors(tlist: list):
    # tlist is a list of tuples of tensors that have the same number of cells
    out = tuple()
    for i in range(len(tlist[0])):
        out = out + (torch.cat([x[i] for x in tlist]),)
    return out
