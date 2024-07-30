import sys, os, json, yaml
import time

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.distributed.rpc import RRef
import torch.distributed.rpc as rpc
from transformers import LlamaTokenizer, LlamaModel, LlamaConfig, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm,
    LlamaDecoderLayer as LDL,
    _make_causal_mask,
    _expand_mask,
)

sys.path.append(".")
from inference_pipeline import RR_TransformerPipeline


class LlamaEmbeddingLayer(nn.Module):
    def __init__(self, config, embedding=None, device="cpu", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = config
        self.device = device

        if embedding == None:
            self.embedding_module = nn.Embedding(
                config.vocab_size, config.hidden_size, self.padding_idx
            ).to(device)
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

    def _prepare_decoder_attention_mask(
        self, attention_mask, input_shape, inputs_embeds, past_key_values_length
    ):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            ).to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(self, inputs):
        input_ids = inputs[0].to(self.device) if inputs[0] != None else None
        attention_mask = inputs[6].to(self.device) if inputs[6] != None else None

        batch_size, seq_length = input_ids.shape
        embeddings = self.embedding_module.forward(input_ids)

        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length), dtype=torch.bool, device=embeddings.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), embeddings, 0
        )

        inputs[4] = embeddings
        inputs[6] = attention_mask
        return inputs


class LlamaNormLayer(nn.Module):
    def __init__(self, config, norm=None, device="cpu", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if norm == None:
            self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(
                device
            )
        else:
            self.norm = norm.to(device)
        self.device = device

    def parameter_rrefs(self):
        r"""
        Create one RRef for each parameter in the given local module, and return a
        list of RRefs.
        """
        return [RRef(p.cpu()) for p in self.norm.parameters()]

    def parameters(self, recurse: bool = True):
        return self.norm.parameters(recurse)

    def forward(self, inputs) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        hidden_states = inputs[4].to(self.device) if inputs[4] != None else None
        hidden_states = self.norm(hidden_states)
        inputs[4] = hidden_states
        return inputs

    def to(self, device):
        self.device = device
        self.norm = self.norm.to(device)
        return self


class LlamaDecoderLayer(nn.Module):
    def __init__(
        self, config, index, decoder=None, device="cpu", *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        # construct encoders
        self.config = config
        self.device = device
        if decoder == None:
            self.layer = LDL(config).to(self.device)
        else:
            self.layer = decoder.to(self.device)
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

        all_hidden_states = inputs[10] if inputs[10] != None else None
        all_self_attentions = inputs[11] if inputs[11] != None else None

        if self.index == 0 and all_hidden_states != None:
            all_hidden_states = all_hidden_states + (inputs[4],)

        layer_outputs = self.layer(
            hidden_states,
            attention_mask,
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

        return inputs


class LlamaLMHead(nn.Module):
    def __init__(self, config, lm_head=None, device="cpu", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # construct encoders
        self.config = config
        self.device = device
        if lm_head == None:
            self.lm_head = nn.Linear(
                config.hidden_size, config.vocab_size, bias=False
            ).to(self.device)
        else:
            self.lm_head = lm_head.to(device)

    def parameter_rrefs(self):
        r"""
        Create one RRef for each parameter in the given local module, and return a
        list of RRefs.
        """
        return [RRef(p.cpu()) for p in self.lm_head.parameters()]

    def to(self, device):
        self.device = device
        self.lm_head = self.lm_head.to(device)
        return self

    def forward(self, inputs):
        hidden_states = inputs[4].to(self.device) if inputs[4] != None else None
        logits = self.lm_head(hidden_states)

        return [None, logits, None, None, None]


def lm_head_output_handler(output_list):
    outputs = (
        torch.cat([x[0] for x in output_list])
        if output_list[0][0] != None
        else None,  # loss
        torch.cat([x[1] for x in output_list])
        if output_list[0][1] != None
        else None,  # logits
        torch.cat([x[2] for x in output_list])
        if output_list[0][2] != None
        else None,  # past key values
        torch.cat([x[3] for x in output_list])
        if output_list[0][3] != None
        else None,  # hidden states
        torch.cat([x[4] for x in output_list])
        if output_list[0][4] != None
        else None,  # attentions
    )

    return CausalLMOutputWithPast(
        loss=outputs[0],
        logits=outputs[1],
        past_key_values=outputs[2],
        hidden_states=outputs[3],
        attentions=outputs[4],
    )


num_batches = 10
batch_size = 16
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
    tokenizer = LlamaTokenizer.from_pretrained("openlm-research/open_llama_7b")
    # tokenizer.add_special_tokens({'pad_token': ' '})
    inputs = dict(
        tokenizer(inputs, return_tensors="pt", max_length=32, truncation=True)
    )
    # inputs = dict(tokenizer(inputs, return_tensors="pt"))
    print("Inputs:", inputs)

    config = LlamaConfig()
    if pre_trained == True:
        net = LlamaForCausalLM.from_pretrained("openlm-research/open_llama_7b")
    else:
        net = LlamaForCausalLM(config)
    net.eval()

    layers = [
        LlamaEmbeddingLayer(config, net.model.embed_tokens),
        *[
            LlamaDecoderLayer(config, i, net.model.layers[i])
            for i in range(config.num_hidden_layers)
        ],
        LlamaNormLayer(config, net.model.norm),
        LlamaLMHead(config, net.lm_head),
    ]

    device_list = devices

    model = RR_TransformerPipeline(
        config,
        split_size,
        ["worker{}".format(i + 1) for i in range(num_workers)],
        layers,
        partitions,
        shards,
        devices=device_list + device_list,
        output_handler=lm_head_output_handler,
    )
    if not model.verify_parameter_consistency():
        print("Parameters not consistent!", flush=True)
        return

    file = open("./llama.csv", "a")
    original_stdout = sys.stdout
    sys.stdout = file

    print("{}".format(shards), end=", ", flush=True)
    tik = time.time()
    for i in range(num_batches):
        outputs = model.generate(**inputs)

    tok = time.time()
    print(f"{split_size}, {tok - tik}, {(num_batches * batch_size) / (tok - tik)}")

    sys.stdout = original_stdout
    print("Raw Outputs:", outputs)
    print(
        "Outputs:",
        tokenizer.batch_decode(
            outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
        ),
    )
    # baseline_outputs = net.generate(**inputs)
    # print("Baseline Outputs:", tokenizer.batch_decode(baseline_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])


def run_worker(
    rank,
    world_size,
    inputs,
    split_size,
    partitions,
    placement,
    devices,
    pre_trained,
    logging,
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
    with open("llama_config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)
        file = open("./llama.log", "w")
        open("./llama.csv", "w")  # flush the csv file
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
        for shards in placements:
            world_size = len(shards) + 1
            print("placement:", shards)
            for i in range(repeat_times):
                # generate input
                prompts = [
                    "Q: What is the largest animal?\nA:",
                    "Q: What is the smallest animal?\nA:",
                    "Q: What is the prettiest animal?\nA:",
                    "Q: What is the most ugly animal?\nA:",
                    "Q: What is the fattest animal?\nA:",
                    "Q: What is the most skinny animal?\nA:",
                    "Q: What is the fastest animal?\nA:",
                    "Q: What is the slowest animal?\nA:",
                ]
                input_prompts = [prompts[0]] * batch_size
                print("Inputs prompts:", input_prompts)

                # run inference process
                tik = time.time()
                mp.spawn(
                    run_worker,
                    args=(
                        world_size,
                        input_prompts,
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
