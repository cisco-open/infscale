import torch
from torchsummary import summary
import json
from functools import partial
from deepspeed.profiling.flops_profiler import FlopsProfiler, num_to_string, number_to_string, macs_to_string, params_to_string, duration_to_string, flops_to_string
from deepspeed.profiling.flops_profiler import get_module_flops, get_module_duration, get_module_macs

def json_init(obj: dict, name: str):
    obj[name] = dict()
    return obj[name]

def json_add(obj: dict, name: str, val):
    obj[name] = val

def json_dump_model_aggregated_profile(output_obj, model, module_depth=-1, top_modules=1):
    """Prints the names of the top top_modules modules in terms of aggregated time, flops, and parameters at depth module_depth.

    Args:
        module_depth (int, optional): the depth of the modules to show. Defaults to -1 (the innermost modules).
        top_modules (int, optional): the number of top modules to show. Defaults to 1.
    """
    info = {}
    if not hasattr(model, "__flops__"):
        print("no __flops__ attribute in the model, call this function after start_profile and before end_profile")
        return

    def walk_module(module, curr_depth, info):
        if curr_depth not in info:
            info[curr_depth] = {}
        if module.__class__.__name__ not in info[curr_depth]:
            info[curr_depth][module.__class__.__name__] = [
                0,
                0,
                0,
            ]  # macs, params, time
        info[curr_depth][module.__class__.__name__][0] += get_module_macs(module)
        info[curr_depth][module.__class__.__name__][1] += module.__params__ + module.__expert_params__
        info[curr_depth][module.__class__.__name__][2] += get_module_duration(module)
        has_children = len(module._modules.items()) != 0
        if has_children:
            for child in module.children():
                walk_module(child, curr_depth + 1, info)

    walk_module(model, 0, info)

    depth = module_depth
    if module_depth == -1:
        depth = len(info) - 1

    print(f'Top {top_modules} modules in terms of params, MACs or fwd latency at different model depths:')

    for d in range(depth):
        num_items = min(top_modules, len(info[d]))

        sort_macs = {
            k: macs_to_string(v[0])
            for k, v in sorted(info[d].items(), key=lambda item: item[1][0], reverse=True)[:num_items]
        }
        sort_params = {
            k: params_to_string(v[1])
            for k, v in sorted(info[d].items(), key=lambda item: item[1][1], reverse=True)[:num_items]
        }
        sort_time = {
            k: duration_to_string(v[2])
            for k, v in sorted(info[d].items(), key=lambda item: item[1][2], reverse=True)[:num_items]
        }

        obj = json_init(output_obj, f"depth {d}")
        json_add(obj, "params", sort_params)
        json_add(obj, "MACs", sort_macs)
        json_add(obj, "fwd latency", sort_time)

def model2json(module: torch.nn.Module):
    """Translate the structure of a NN model to a json object"""
    json_obj = dict()
    extra = []
    extra_repr = module.extra_repr()

    if extra_repr:
        json_obj["extra"] = extra_repr
    if hasattr(module, "__input_shape__"):
        json_obj["input_shape"] = module.__input_shape__
    if hasattr(module, "__output_shape__"):
        json_obj["output_shape"] = module.__output_shape__

    if len(module._modules.items()) > 0:
        json_obj['children'] = dict()
        for key, module in module._modules.items():
            json_obj['children'][key] = model2json(module)

    json_obj["name"] = module._get_name()

    return json_obj

def json_dump_model_profile(model: torch.nn.Module, profiler: FlopsProfiler, profile_step=1, module_depth=-1, top_modules=1, detailed=True, output_file=None):
    """Prints the model graph with the measured profile attached to each module.
        Args:
            profile_step (int, optional): The global training step at which to profile. Note that warm up steps are needed for accurate time measurement.
            module_depth (int, optional): The depth of the model to which to print the aggregated module information. When set to -1, it prints information from the top to the innermost modules (the maximum depth).
            top_modules (int, optional): Limits the aggregated profile output to the number of top modules specified.
            detailed (bool, optional): Whether to print the detailed model profile.
            output_file (str, optional): Path to the output file. If None, the profiler prints to stdout.
    """
    if not profiler.started:
        return
    
    import sys
    import os.path
    f = None
    if output_file and output_file != "":
        dir_path = os.path.dirname(os.path.abspath(output_file))
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        f = open(output_file, "w")

    # create the top-level output object
    output = dict()

    total_flops = profiler.get_total_flops()
    total_macs = profiler.get_total_macs()
    total_duration = profiler.get_total_duration()
    total_params = profiler.get_total_params()
    expert_tensor_parallelism = None  # silence the linters
    total_model_expert_params = total_model_nonexpert_params = 0
    if profiler.ds_engine:
        total_model_nonexpert_params = profiler.model.__params__ * profiler.ds_engine.mp_world_size
        if profiler.ds_engine.has_moe_layers:
            expert_tensor_parallelism = profiler.is_expert_tensor_parallelism_enabled()
            total_model_expert_params = profiler.model.__model_expert_params__ * (profiler.ds_engine.mp_world_size
                                                                                if expert_tensor_parallelism else 1)

    profiler.flops = total_flops
    profiler.macs = total_macs
    profiler.params = total_params

    print("\n-------------------------- DeepSpeed Flops Profiler --------------------------")
    # beginning of Profile Summary
    print(f'Profile Summary at step {profile_step}:')
    psas_output = json_init(output, 'Profile Summary')

    if profiler.ds_engine:
        json_add(psas_output, 'world size', '{:<8}'.format(profiler.ds_engine.world_size))
        json_add(psas_output, 'data parallel size', '{:<8}'.format(profiler.ds_engine.dp_world_size))
        json_add(psas_output, 'model parallel size', '{:<8}'.format(profiler.ds_engine.mp_world_size))
        json_add(psas_output, 'batch size per GPU', '{:<8}'.format(profiler.ds_engine.train_micro_batch_size_per_gpu()))
        if profiler.ds_engine.has_moe_layers:
            json_add(psas_output, 'expert tensor parallelism enabled', '{:<8}'.format(expert_tensor_parallelism))

    json_add(psas_output, 'params per gpu', '{:<8}'.format(total_params))
    if total_model_expert_params > 0:
        json_add(psas_output, 'params of model', '{:<8}'.format(params_to_string(total_model_nonexpert_params + total_model_expert_params)))
        json_add(psas_output, 'non-expert params of model', '{:<8}'.format(params_to_string(total_model_nonexpert_params)))
        json_add(psas_output, 'expert params of model', '{:<8}'.format(params_to_string(total_model_expert_params)))
    else:
        json_add(psas_output, 'params of model', '{:<8}'.format(total_model_nonexpert_params))

    json_add(psas_output, 'fwd MACs per GPU', '{:<8}'.format(macs_to_string(total_macs)))
    json_add(psas_output, 'fwd flops per GPU', '{:<8}'.format(num_to_string(total_flops)))
    json_add(psas_output, 'fwd flops of model', '{:<8}'.format(num_to_string(total_flops * ((profiler.ds_engine.mp_world_size) if profiler.ds_engine else 1))))

    fwd_latency = profiler.get_total_duration()
    if profiler.ds_engine and profiler.ds_engine.wall_clock_breakdown():
        fwd_latency = profiler.ds_engine.timers('forward').elapsed(False) / 1000.0
    json_add(psas_output, 'fwd latency', '{:<8}'.format(duration_to_string(fwd_latency)))
    json_add(psas_output, 'fwd FLOPS per GPU', '{:<8}'.format(flops_to_string(total_flops / fwd_latency)))

    print("DS engine:", profiler.ds_engine)
    if profiler.ds_engine and profiler.ds_engine.wall_clock_breakdown():
        bwd_factor = 2 + profiler.recompute_fwd_factor
        bwd_latency = profiler.ds_engine.timers('backward').elapsed(False) / 1000.0
        step_latency = profiler.ds_engine.timers('step').elapsed(False) / 1000.0
        json_add(psas_output, 'bwd latency', '{:<8}'.format(duration_to_string(bwd_latency)))
        json_add(psas_output, 'bwd FLOPS per GPU', '{:<8}'.format(flops_to_string(flops_to_string(bwd_factor * total_flops / bwd_latency))))
        json_add(psas_output, 'fwd+bwd FLOPS per GPU', '{:<8}'.format(flops_to_string((bwd_factor + 1) * total_flops / (fwd_latency + bwd_latency))))

        json_add(psas_output, 'step latency', '{:<8}'.format(duration_to_string(step_latency)))

        iter_latency = fwd_latency + bwd_latency + step_latency
        json_add(psas_output, 'iter latency', '{:<8}'.format(duration_to_string(iter_latency)))
        json_add(psas_output, 'FLOPS per GPU', '{:<8}'.format(flops_to_string((bwd_factor + 1) * total_flops / iter_latency)))

        samples_per_iter = profiler.ds_engine.train_micro_batch_size_per_gpu() * profiler.ds_engine.world_size
        json_add(psas_output, 'samples/second', '{:<8.2f}'.format(samples_per_iter / iter_latency))

    # end of Profile Summary
    json.dump(psas_output, sys.stdout, indent=4)

    def profile_repr(module):
        params = module.__params__ + module.__expert_params__
        flops = get_module_flops(module)
        macs = get_module_macs(module)
        duration = get_module_duration(module)

        profile = dict()
        profile["params"] = params_to_string(params)
        profile["percentage of total params"] = "{:.2%}".format(params / total_params if total_params else 0)
        profile["MACs"] = macs_to_string(macs)
        profile["percentage of total MACs"] = "{:.2%}".format(0.0 if total_macs == 0 else macs / total_macs)
        profile["fwd latency"] = duration_to_string(duration)
        profile["percentage of total fwd latency"] = "{:.2%}".format(0.0 if total_duration == 0 else duration / total_duration)
        profile["fwd FLOPS"] = flops_to_string(0.0 if duration == 0 else flops / duration)

        return profile

    def add_extra_repr(module):
        flops_extra_repr = profile_repr.__get__(module)
        if module.extra_repr != flops_extra_repr:
            module.original_extra_repr = module.extra_repr
            module.extra_repr = flops_extra_repr
            assert module.extra_repr != module.original_extra_repr

    def del_extra_repr(module):
        if hasattr(module, "original_extra_repr"):
            module.extra_repr = module.original_extra_repr
            del module.original_extra_repr

    profiler.model.apply(add_extra_repr)

    print("\n----------------------------- Aggregated Profile per GPU -----------------------------")
    # beginning of aggregated profile per GPU
    appg_output = json_init(output, "Aggregated Profile per GPU")
    json_dump_model_aggregated_profile(appg_output, profiler.model, module_depth=module_depth, top_modules=top_modules)

    # end of aggregated profile per GPU
    json.dump(appg_output, sys.stdout, indent=4)

    if detailed:
        print("\n------------------------------ Detailed Profile per GPU ------------------------------")
        print(
            "Each module profile is listed after its name in the following order: \nparams, percentage of total params, MACs, percentage of total MACs, fwd latency, percentage of total fwd latency, fwd FLOPS"
        )
        print(
            "\nNote: 1. A module can have torch.nn.module or torch.nn.functional to compute logits (e.g. CrossEntropyLoss). They are not counted as submodules, thus not to be printed out. However they make up the difference between a parent's MACs (or latency) and the sum of its submodules'.\n2. Number of floating-point operations is a theoretical estimation, thus FLOPS computed using that could be larger than the maximum system throughput.\n3. The fwd latency listed in the top module's profile is directly captured at the module forward function in PyTorch, thus it's less than the fwd latency shown above which is captured in DeepSpeed.\n"
        )
        # beginning of detailed profile per GPU
        dppg_output = json_init(output, "Detailed Profile per GPU")
        dppg_output["root"] = model2json(model)
        # end of detailed profile per GPU
        print("Only shown in the json file")

    profiler.model.apply(del_extra_repr)

    print("------------------------------------------------------------------------------")

    if output_file:
        json.dump(output, f, indent=4)
        f.close()

def get_model_inference_profile(model,
                      input_shape=None,
                      args=[],
                      kwargs={},
                      print_profile=True,
                      detailed=True,
                      module_depth=-1,
                      top_modules=1,
                      warm_up=1,
                      as_string=True,
                      output_file=None,
                      ignore_modules=None,
                      mode='forward'):
    """Returns the total floating-point operations, MACs, and parameters of a model.
    Example:
    .. code-block:: python
        model = torchvision.models.alexnet()
        batch_size = 256
        flops, macs, params = get_model_profile(model=model, input_shape=(batch_size, 3, 224, 224)))
    Args:
        model ([torch.nn.Module]): the PyTorch model to be profiled.
        input_shape (tuple): input shape to the model. If specified, the model takes a tensor with this shape as the only positional argument.
        args (list): list of positional arguments to the model.
        kwargs (dict): dictionary of keyword arguments to the model.
        print_profile (bool, optional): whether to print the model profile. Defaults to True.
        detailed (bool, optional): whether to print the detailed model profile. Defaults to True.
        module_depth (int, optional): the depth into the nested modules. Defaults to -1 (the inner most modules).
        top_modules (int, optional): the number of top modules to print in the aggregated profile. Defaults to 3.
        warm_up (int, optional): the number of warm-up steps before measuring the latency of each module. Defaults to 1.
        as_string (bool, optional): whether to print the output as string. Defaults to True.
        output_file (str, optional): path to the output file. If None, the profiler prints to stdout.
        ignore_modules ([type], optional): the list of modules to ignore during profiling. Defaults to None.
    Returns:
        The number of floating-point operations, multiply-accumulate operations (MACs), and parameters in the model.
    """
    assert isinstance(model, torch.nn.Module), "model must be a PyTorch module"
    prof = FlopsProfiler(model)
    model.eval()

    if input_shape is not None:
        assert type(input_shape) is tuple, "input_shape must be a tuple"
        assert len(input_shape) >= 1, "input_shape must have at least one element"
        try:
            input = torch.ones(()).new_empty(
                (*input_shape, ),
                dtype=next(model.parameters()).dtype,
                device=next(model.parameters()).device,
            )
        except StopIteration:
            input = torch.ones(()).new_empty((*input_shape, ))

        args = [input]
    assert (len(args) > 0) or (len(kwargs) > 0), "args and/or kwargs must be specified if input_shape is None"
    for _ in range(warm_up):
        if kwargs:
            if mode == 'forward':
                _ = model(*args, **kwargs)
            if mode == 'generate':
                _ = model.generate(*args, **kwargs)
        else:
            if mode == 'forward':
                _ = model(*args)
            if mode == 'generate':
                _ = model.generate(*args)

    prof.start_profile(ignore_list=ignore_modules)

    # add input shape and output shape hook
    def register_io_shape_hook(module, ignore_list):
        if ignore_list and type(module) in ignore_list:
            return
        
        def io_shape_hook(module, input, output):
            print("input:", input)
            print("output", output)
            if len(input) > 0:
                module.__input_shape__ = list(input[0].size())
            if hasattr(output, '__iter__') or hasattr(output, '__getitem__'):
                module.__output_shape__ = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                module.__output_shape__ = list(output.size())

        if (
            not isinstance(module, torch.nn.Sequential)
            and not isinstance(module, torch.nn.ModuleList)
            and not (module == model)
        ):
            module.__io_shape_hook_handle__ = module.register_forward_hook(io_shape_hook)

    model.apply(partial(register_io_shape_hook, ignore_list=ignore_modules))

    if kwargs:
        if mode == 'forward':
            _ = model(*args, **kwargs)
        if mode == 'generate':
            _ = model.generate(*args, **kwargs)
    else:
        if mode == 'forward':
            _ = model(*args)
        if mode == 'generate':
            _ = model.generate(*args)

    flops = prof.get_total_flops()
    macs = prof.get_total_macs()
    params = prof.get_total_params()
    if print_profile:
        json_dump_model_profile(model, prof, profile_step=warm_up,
                                 module_depth=module_depth,
                                 top_modules=top_modules,
                                 detailed=detailed,
                                 output_file=output_file)

    def remove_io_shape_hook(module):
        if hasattr(module, "__io_shape_hook_handle__"):
            module.__io_shape_hook_handle__.remove()

    model.apply(remove_io_shape_hook)
    prof.end_profile()
    if as_string:
        return number_to_string(flops), macs_to_string(macs), params_to_string(params)

    return flops, macs, params