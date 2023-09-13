# ailia onnx optimizer
# (c) 2020-2023 AXELL CORPORATION

import argparse
import copy
import json
import os
import pathlib
import sys

import numpy
import onnx
import onnx.checker
import onnx.helper
import onnx.mapping
import onnx.numpy_helper
import onnx.shape_inference

SCRIPT_NAME = '+onnx_optimizer'
SCRIPT_VERSION = '+0.16'


class NodeInfo:
    def __init__(self, node):
        self.__node = node
        self.__input = set()
        self.__output = []
        for s in node.input:
            if len(s) == 0:
                continue
            self.__input.add(s)
        for s in node.output:
            self.__output.append(s)

    def clear_input(self, name):
        self.__input.remove(name)
        return len(self.__input)

    def check_resolve(self):
        return (len(self.__input) == 0)

    def ref_id(self):
        return self.__output[0]

    def ref_output(self):
        return self.__output

    def ref_input(self):
        return self.__input

    def ref_node(self):
        return self.__node


class ShapeAndTypeInfo:
    def __init__(self, model: onnx.ModelProto) -> None:
        self.__info = self.__shape_inference(model)

    def __shape_inference(self, model: onnx.ModelProto) -> dict:
        inf_model = onnx.shape_inference.infer_shapes(model)
        info = {}
        for e in model.graph.initializer:
            info[e.name] = {'type': e.data_type, 'shape': e.dims}
        for x in (inf_model.graph.input, inf_model.graph.output, inf_model.graph.value_info):
            for e in x:
                type = e.type.tensor_type.elem_type
                shape = []
                for v in e.type.tensor_type.shape.dim:
                    shape.append(v.dim_value)
                info[e.name] = {'type': type, 'shape': shape}
        return info

    def refresh_inference(self, model: onnx.ModelProto):
        self.__info = self.__shape_inference(model)

    def is_inferred(self, name: str) -> bool:
        if not name in self.__info:
            return False
        if len(self.__info[name]['shape']) == 0:
            return False
        if 0 in self.__info[name]['shape']:
            return False
        return True

    def get_shape(self, name: str) -> tuple:
        if not name in self.__info:
            return None
        return self.__info[name].get('shape')

    def get_dim(self, name: str) -> int:
        if not name in self.__info:
            return None
        return len(self.__info[name]['shape'])

    def get_type(self, name: str) -> int:
        return self.__info[name]['type']

    def listup_node_datatype(self) -> dict:
        node_datatype = {}
        for node, info in self.__info.items():
            node_datatype.setdefault(node, return_elem_datatype_str(info['type']))
        return node_datatype

    def register(self, name: str, data: numpy.array) -> None:
        self.__info[name] = {'type': convert_dtype_np2tensor(data.dtype), 'shape': data.shape}


def convert_dtype_np2tensor(nptype: numpy.dtype) -> int:
    if hasattr(onnx.helper, 'np_dtype_to_tensor_dtype'):
        return onnx.helper.np_dtype_to_tensor_dtype(nptype)
    return onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[nptype]


def convert_dtype_tensor2np(tensortype: int) -> numpy.dtype:
    if hasattr(onnx.helper, 'tensor_dtype_to_np_dtype'):
        return onnx.helper.tensor_dtype_to_np_dtype(tensortype)
    return onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[tensortype]


def onnx_topologically_sort(model):
    pool = {}
    in2nodes = {}  # input_name -> nodes
    candidate = []
    for e in model.graph.node:
        n = NodeInfo(e)
        id = n.ref_id()
        if e.op_type == 'Constant':
            candidate.append(n.ref_id())
        for i in n.ref_input():
            in2nodes.setdefault(i, [])
            in2nodes[i].append(n)
        pool[id] = n

    for x in (model.graph.initializer, model.graph.input):
        for e in x:
            name = e.name
            if not name in in2nodes:
                continue
            for n in in2nodes.pop(name):
                n.clear_input(name)
                if n.check_resolve():
                    candidate.append(n.ref_id())

    sorted = []
    while len(candidate) > 0:
        removed = []
        for id in candidate:
            n = pool.pop(id)
            sorted.append(n.ref_node())
            removed.extend(n.ref_output())

        candidate.clear()
        for name in removed:
            if not name in in2nodes:
                continue
            for n in in2nodes.pop(name):
                n.clear_input(name)
                if n.check_resolve():
                    candidate.append(n.ref_id())

    s_graph = onnx.helper.make_graph(sorted, model.graph.name, model.graph.input, model.graph.output, model.graph.initializer)
    s_model = onnx.helper.make_model(s_graph, producer_name=model.producer_name + SCRIPT_NAME, producer_version=model.producer_version + SCRIPT_VERSION, opset_imports=model.opset_import)

    return s_model


def convert_to_list(arg):
    ret = []
    for e in arg:
        ret.append(e)
    return ret


def adjust_graph_output(mod_node, replace, model):
    oname = {}
    for o in model.graph.output:
        if o.name in replace.keys():
            oname[replace[o.name]] = o.name
    for n in mod_node:
        for i in range(len(n.input)):
            k = n.input[i]
            if k in oname:
                n.input[i] = oname[k]
        for i in range(len(n.output)):
            k = n.output[i]
            if k in oname:
                n.output[i] = oname[k]


def yolov3_special_treatment(model):

    ver = model.opset_import[0].version
    if (not ver in (10, 11)):
        detail = 'unsupported model opset_version=({})'.format(ver)
        raise Exception(detail)

    mod_input = convert_to_list(model.graph.input)

    data = listup_initializer(model)

    # replace loop nodes & add input parameter
    mod_node = []
    loop_count = 0
    for n in model.graph.node:
        if (n.op_type == 'Loop'):
            loop_count += 1
            for i in range(1, len(n.input)):
                data[n.input[i]]['ref'] -= 1
            wn_name = 'usq/' + n.input[0]
            ni = [n.input[0]]
            no = [wn_name]
            nn = onnx.helper.make_node("Unsqueeze", ni, no, name=wn_name, axes=[0])
            mod_node.append(nn)
            ni = ['arange_base', 'arange_start', wn_name]
            no = [n.output[1]]
            nn = onnx.helper.make_node("Slice", ni, no, name=n.name)
            mod_node.append(nn)
        elif (n.op_type == 'NonMaxSuppression'):
            iou = n.input[3]
            threshold = n.input[4]
            # workaround for AILIA
            data[iou]['ref'] -= 1
            data[threshold]['ref'] -= 1
            mod_input.append(onnx.helper.make_tensor_value_info(iou, onnx.TensorProto.FLOAT, (1,)))
            mod_input.append(onnx.helper.make_tensor_value_info(threshold, onnx.TensorProto.FLOAT, (1,)))
            mod_node.append(n)
        else:
            mod_node.append(n)

    if loop_count == 0:
        return model

    print("    [i] report : replace {} Loop to Slice with yolov3_special_treatment.".format(loop_count))

    # append arange_base (0..512 / INT32 / 1D Tensor)
    arange_base = numpy.arange(0, 512, dtype=int)
    data['arange_base'] = {'ref': 1, 'id': 'arange_base', 'body': arange_base}
    # append arange_start (0 / INT64 / 1D Tensor)
    arange_start = numpy.zeros([1], dtype=numpy.int64)
    data['arange_start'] = {'ref': 1, 'id': 'arange_start', 'body': arange_start}

    mod_initializer, dummy = pack_initializer(data, model.graph.input)

    m_graph = onnx.helper.make_graph(mod_node, model.graph.name, mod_input, model.graph.output, mod_initializer)
    m_model = onnx.helper.make_model(m_graph, producer_name=model.producer_name, producer_version=model.producer_version, opset_imports=model.opset_import)

    return m_model


def onnx_extract_constant_to_initializer(model):
    mod_initializer = convert_to_list(model.graph.initializer)
    mod_node = []
    count = 0
    ex = set()  # exclude set
    for o in model.graph.output:
        ex.add(o.name)
    for n in model.graph.node:
        if (n.op_type != 'Constant'):
            mod_node.append(n)
            continue
        if n.output[0] in ex:
            mod_node.append(n)
            continue
        t = [a.t for a in n.attribute if (a.name == 'value')]
        t[0].name = n.output[0]
        mod_initializer.append(t[0])
        count += 1

    if (count == 0):
        # no constant node
        return model

    print("    [i] report : extract {} constant node to initializer.".format(count))

    # generate modified model
    m_graph = onnx.helper.make_graph(mod_node, model.graph.name, model.graph.input, model.graph.output, mod_initializer)
    m_model = onnx.helper.make_model(m_graph, producer_name=model.producer_name, producer_version=model.producer_version, opset_imports=model.opset_import)

    return m_model


def is_1in_eltwise(op_type):
    type_table = [
        'Abs',             # Y=abs(X)
        'Acos',            # Y=acos(X)
        'Acosh',           # Y=acosh(X)
        'Asin',            # Y=asin(X)
        'Asinh',           # Y=asinh(X)
        'Atan',            # Y=atan(X)
        'Atanh',           # Y=atanh(X)
        'Cast',            # Y=cast<type>(X)
        'Ceil',            # Y=ceil(X)
        'Clip',            # Y=max(min(X, max_value), min_value)
        'Cos',             # Y=cos(X)
        'Cosh',            # Y=cosh(X)
        'Elu',             # Y=(X<0) ? alpha*(exp(X)-1.0) : X
        'Erf',             # Y=erf(X)
        'Exp',             # Y=exp(X)
        'Floor',           # Y=floor(X)
        'HardSigmoid',     # Y=max(0, min(1, alpha * x + beta))
        'Identity',        # Y=X
        'IsInf',           # Y=(X==Inf) ? true : false
        'IsNaN',           # Y=(X==NaN) ? true : false
        'LeakyRelu',       # Y=(X<0) ? alpha*X : X
        'Log',             # Y=log(X)
        'Neg',             # Y=-X
        'Not',             # Y=X ? false : true
        'Reciprocal',      # Y=1/X
        'Round',           # Y=round(X)
        'Relu',            # Y=(X<0) ? 0 : X
        'Selu',            # Y=gamma * ((X<0) ? (alpha * exp(X) - alpha) : X)
        'Shrink',          # Y=(X<-lambd) ? X+bias : (X>lambd) ? X-bias : 0
        'Sigmoid',         # Y=1.0/(1.0 + exp(-X))
        'Sign',            # Y=(X<0) ? -1 : (X>0) ? 1 : 0
        'Sin',             # Y=sin(X)
        'Sinh',            # Y=sinh(X)
        'Sqrt',            # Y=sqrt(X)
        'Tan',             # Y=tan(X)
        'Tanh',            # Y=tanh(X)
        'ThresholdedRelu',  # Y=(X<alpha) ? 0 : X
    ]
    return (op_type in type_table)


def is_2in_eltwise(op_type):
    type_table = [
        'Add',
        'Mul',
        'Sub',
        'Div',
        'Mod',
        'Pow',
        'Min',
        'Max',
        'Equal',
        'Less',
        'Greater',
        'LessOrEqual',
        'GreaterOrEqual',
        'And',
        'Or',
        'Xor',
    ]
    return (op_type in type_table)

# https://github.com/onnx/onnx/blob/main/onnx/onnx.proto#L480


def return_elem_datatype_str(elem_type):
    DataType = [
        "UNDEFINED",
        "FLOAT",
        "UINT8",
        "INT8",
        "UINT16",
        "INT16",
        "INT32",
        "INT64",
        "STRING",
        "BOOL",
        "FLOAT16",
        "DOUBLE",
        "UINT32",
        "UINT64",
        "COMPLEX64",
        "COMPLEX128",
        "BFLOAT16"
    ]
    return DataType[elem_type]


def pick_attribute(node, attr_name):
    for a in node.attribute:
        if (a.name == attr_name):
            return a
    return None


def pick_constant_param(node_input, data):
    for i in node_input:
        if i in data:
            return data[i]
    return None


def listup_attribute(node):
    r = {}
    for a in node.attribute:
        r[a.name] = onnx.helper.get_attribute_value(a)
    return r


def check_duplicated_node(ref, trg, data, replace):
    if ref.op_type != trg.op_type:
        return False
    if len(ref.input) != len(trg.input):
        return False
    if len(ref.output) != len(trg.output):
        return False
    r_attr = listup_attribute(ref)
    t_attr = listup_attribute(trg)
    if len(r_attr) != len(t_attr):
        return False
    for k in r_attr.keys():
        if not k in t_attr:
            return False
        if not r_attr[k] == t_attr[k]:
            return False
    for i in range(len(ref.input)):
        rid = ref.input[i]
        tid = trg.input[i]
        if rid in replace:
            rid = replace[rid]
        if tid in replace:
            tid = replace[tid]
        if rid == tid:
            continue
        if (rid in data) and (tid in data):
            if numpy.array_equal(data[rid]['body'], data[tid]['body']):
                continue
        return False
    return True


def check_needless_transpose(perm):
    for x in range(len(perm)):
        if x != perm[x]:
            return False
    return True


def adjust_subgraph(subgraph, replace):
    if len(replace) == 0:
        return subgraph
    mod_node = []
    for n in subgraph.node:
        for i in range(len(n.input)):
            k = n.input[i]
            if k in replace:
                n.input[i] = replace[k]
        mod_node.append(n)
    return onnx.helper.make_graph(mod_node, subgraph.name, subgraph.input, subgraph.output, subgraph.initializer)


def adjust_node_with_subgraph(node, replace):
    if (node.op_type == "Loop") or (node.op_type == "Scan"):
        attr = listup_attribute(node)
        attr["body"] = adjust_subgraph(attr["body"], replace)
        node = onnx.helper.make_node(node.op_type, inputs=node.input, outputs=node.output, **attr)
    elif node.op_type == "If":
        attr = listup_attribute(node)
        attr["else_branch"] = adjust_subgraph(attr["else_branch"], replace)
        attr["then_branch"] = adjust_subgraph(attr["then_branch"], replace)
        node = onnx.helper.make_node(node.op_type, inputs=node.input, outputs=node.output, **attr)
    return node


def register_subgraph_ref(data, subgraph):
    for sn in subgraph.node:
        for si in sn.input:
            if si in data:
                data[si]['ref'] += 1
    return data


def listup_initializer(model):
    data = {}
    for e in model.graph.initializer:
        data[e.name] = {'ref': 0, 'id': e.name, 'body': onnx.numpy_helper.to_array(e)}
    for n in model.graph.node:
        for i in n.input:
            if i in data:
                data[i]['ref'] += 1
        if (n.op_type == 'Loop') or (n.op_type == 'Scan'):
            sg = onnx.helper.get_attribute_value(pick_attribute(n, 'body'))
            data = register_subgraph_ref(data, sg)
        if n.op_type == 'If':
            sg = onnx.helper.get_attribute_value(pick_attribute(n, 'else_branch'))
            data = register_subgraph_ref(data, sg)
            sg = onnx.helper.get_attribute_value(pick_attribute(n, 'then_branch'))
            data = register_subgraph_ref(data, sg)

    for o in model.graph.output:
        if o.name in data:
            data[o.name]['ref'] += 1

    return data


def pack_initializer(data, input):
    r_init = []
    r_input = []
    erased = {}
    for e in data.values():
        if e['ref'] < 1:
            erased[e['id']] = True
            continue
        body = e['body']
        r_init.append(onnx.numpy_helper.from_array(body, e['id']))
    for i in input:
        if i.name in erased:
            continue
        if i.name in data:
            d = data[i.name]['body']
            r_input.append(onnx.helper.make_tensor_value_info(i.name, convert_dtype_np2tensor(d.dtype), d.shape))
        else:
            r_input.append(i)
    return r_init, r_input


def check_data_size_change_on_eltwise(input_shape, const_shape):
    if len(const_shape) == 0:
        return False
    if (len(const_shape) == 1) and (const_shape[0] == 1):
        return False
    if not input_shape:
        return True
    if len(input_shape) != len(const_shape):
        return True
    for ie, ce in zip(input_shape, const_shape):
        if (ie == ce):
            continue
        if (ce == 1):
            continue
        if (ie == 1):
            return True
    return False


def onnx_move_transpose_rp(model_graph, model_output, data, shapes, opset_ver):

    # move transpose to upstream
    OL = 'o'
    IL = 'i'

    def is_candidate(op_type):
        if is_2in_eltwise(op_type):
            return True
        if is_1in_eltwise(op_type):
            return True
        support_op = ['Transpose', 'Pad', 'Resize', 'Concat', 'PRelu']
        if op_type in support_op:
            return True
        return False

    # list up replace node candidate
    cand = {}  # candidate
    cnct = {}  # conection
    output_id = '____graph_output____'
    # for output_node in model.output:
    for n in model_output:
        cnct.setdefault(n.name, [])
        cnct[n.name].append(output_id)
    for n in model_graph:
        for o in n.output:
            for i in n.input:
                cnct.setdefault(i, [])
                cnct[i].append(o)
        nmo = n.output[0]  # node main output
        if is_candidate(n.op_type):
            attr = listup_attribute(n)
            cand[nmo] = {'name': n.name, 'type': n.op_type, OL: nmo, IL: n.input, 'attr': attr}

    # 2in eltwise check input function
    def check_eltwise_input(inputs, perm):
        for e in inputs:
            if e in data:
                if len(data[e]['body'].shape) < len(perm):
                    return True
        for e in inputs:
            if e in data:
                continue
            dim = shapes.get_dim(e)
            if dim == 0:
                if (e in cand) and (cand[e]['type'] == 'Transpose'):
                    dim = len(cand[e]['attr']['perm'])
            if dim != len(perm):
                return False
        return True

    # verify candidate connection
    count = 0
    target = {}
    multi_ipt_trans = {}
    duplicate_trans = {}
    for n in cand.values():
        if not (n['type'] == 'Transpose'):
            continue
        wi = n
        wl = []
        if (len(cnct[wi[IL][0]]) > 1):
            if (wi[IL][0] in multi_ipt_trans):
                multi_ipt_trans[wi[IL][0]].append(wi)
                if len(cnct[wi[IL][0]]) == len(multi_ipt_trans[wi[IL][0]]):
                    flg = True
                    std_perm = multi_ipt_trans[wi[IL][0]][0]['attr']['perm']
                    for tr in multi_ipt_trans[wi[IL][0]]:
                        if not std_perm == tr['attr']['perm']:
                            flg = False
                            continue
                    if flg:
                        duplicate_trans[wi[IL][0]] = multi_ipt_trans[wi[IL][0]]
                        continue
                    else:
                        continue
                else:
                    continue
            else:
                multi_ipt_trans.setdefault(wi[IL][0], [])
                multi_ipt_trans[wi[IL][0]].append(wi)
                continue
        if not wi[IL][0] in cand:
            continue
        if (cand[wi[IL][0]]['type'] == 'Transpose'):
            continue

        wl.append(wi)
        wi = cand[wi[IL][0]]

        # reject unsupported input variations
        if is_2in_eltwise(wi['type']):
            perm = wl[0]['attr']['perm']
            if not check_eltwise_input(wi[IL], perm):
                continue
        elif is_1in_eltwise(wi['type']):
            if wi[IL][0] in data:
                continue
        elif (wi['type'] == 'Pad') or (wi['type'] == 'PRelu'):
           if wi[IL][0] in data:
                continue
           if (len(wi[IL]) > 1) and (not wi[IL][1] in data):
                continue
 
        wl.append(wi)
        target[wl[0][OL]] = copy.deepcopy(wl[1])
        target[wl[1][OL]] = copy.deepcopy(wl[0])
        count += 1

    # detect_duplicate transpose
    del_duplicate_transpose = {}
    upper_dup_transpose = {}
    bottom_dup_transpose_ipt = {}
    input2output = {}
    bottom_input2upper_output = {}
    for n in model_graph:
        id = n.output[0]
        # check upper node can move
        if id in duplicate_trans:
            if not (is_1in_eltwise(n.op_type)):
                continue
            attr = listup_attribute(n)
            upper_dup_transpose[id] = {'name': n.name, 'type': n.op_type, OL: id, IL: n.input, 'attr': attr}
            input2output.setdefault(id, [])
            for node in duplicate_trans[id]:
                del_duplicate_transpose[node[OL]] = node
                bottom_input2upper_output[node[OL]] = id
                input2output[id].append(node[OL])
                count += 1
            continue
        ipt2transpose = ''
        for idx, ipt in enumerate(n.input):
            if ipt in del_duplicate_transpose:
                ipt2transpose = ipt
                break
        if ipt2transpose == '':
            continue
        input2output.setdefault(ipt2transpose, n.output)
        attr = listup_attribute(n)
        bottom_dup_transpose_ipt[id] = [idx, {'name': n.name, 'type': n.op_type, OL: id, IL: n.input, 'attr': attr}, bottom_input2upper_output[ipt2transpose]]

    if (count == 0):
        # no available replace node
        return model_graph, False, 0

     # initializer updater function
    def update_initializer(db, shapes, id, body, sfx):
        if db[id]['ref'] == 1:
            db[id]['body'] = body
            return id
        db[id]['ref'] -= 1
        mod_id = id + sfx
        nxt_idx = 1
        while mod_id in db:
            if numpy.array_equal(db[mod_id]['body'], body):
                db[mod_id]['ref'] += 1
                return mod_id
            mod_id = id + f'{sfx}.{nxt_idx}'
            nxt_idx += 1
        db[mod_id] = {
            'ref': 1,
            'id': mod_id,
            'body': body,
        }
        shapes.register(mod_id, body)
        return mod_id

    def convert_perm_for_head_tail_pair(perm):
        dim = len(perm)
        perm = numpy.array(perm)
        perm = numpy.where( perm < 0, perm+dim, perm )
        return numpy.concatenate( [perm, perm+dim] )

    def apply_perm_head_tail_pair(trg, perm):
        dperm = convert_perm_for_head_tail_pair(perm)
        return trg[dperm]

    def adjust_concat_axis(axis, perm):
        dim = len(perm)
        if axis < 0:
            axis += dim
        perm = numpy.array(perm)
        perm = numpy.where( perm < 0, perm+dim, perm )
        for i,v in enumerate(perm):
            if v == axis:
                return i
        return axis

    def insert_additional_node(db, id, sfx, perm):
        mod_id = id + sfx
        nxt_idx = 1
        while mod_id in db:
            if numpy.array_equal(db[mod_id], perm):
                return mod_id, False
            mod_id = id + f'{sfx}.{nxt_idx}'
            nxt_idx += 1
        db[mod_id] = perm
        return mod_id, True

    def register_gather_indices(db, shapes, id, body):
        body = numpy.array(body)
        mod_id = id + '.gather'
        nxt_idx = 1
        while mod_id in db:
            if numpy.array_equal(db[mod_id], body):
                db[mod_id]['ref'] += 1
                return mod_id
            mod_id = id + f'{sfx}.{nxt_idx}'
            nxt_idx += 1
        db[mod_id] = {
            'ref': 1,
            'id': mod_id,
            'body': body,
        }
        shapes.register(mod_id, body)
        return mod_id

    mod_node = []
    blob_names = {}
    for n in model_graph:
        id = n.output[0]

        if not (id in target or id in del_duplicate_transpose or id in upper_dup_transpose or id in bottom_dup_transpose_ipt):
            mod_node.append(n)
            continue

        if id in upper_dup_transpose:  # move transpose
            tr_node = del_duplicate_transpose[input2output[id][0]]
            nn_ipt = []
            # new transpose node
            for ipt in n.input:
                if ipt in data:
                    nn_ipt.append(ipt)
                    perm = tr_node['attr']['perm']
                    const = pick_constant_param(wi[IL], data)
                    ipt = data[const['id']]
                    n = len(perm) - len(const['body'].shape)
                    if n > 0:
                        mod_shape = [1] * n + list(const['body'].shape)
                        const['body'] = const['body'].reshape(mod_shape)
                    const['body'] = const['body'].transpose(perm)
                    data[ipt]['body'] = const['body']
                    continue
                nn_ipt.append(ipt + '.tr.opt')

                ntranspose = onnx.helper.make_node('Transpose', [ipt], [ipt + '.tr.opt'], **tr_node['attr'])
                mod_node.append(ntranspose)
            nn = onnx.helper.make_node(n.op_type, nn_ipt, n.output, n.name, **listup_attribute(n))
            mod_node.append(nn)
            continue
        elif id in del_duplicate_transpose:
            continue
        elif id in bottom_dup_transpose_ipt:
            idx = bottom_dup_transpose_ipt[id][0]
            ipt = n.input
            ipt[idx] = bottom_dup_transpose_ipt[id][2]
            attr = listup_attribute(n)
            nn = onnx.helper.make_node(n.op_type, ipt, n.output, n.name, **attr)
            mod_node.append(nn)
            continue
        wi = target[id]
        if n.op_type == 'Transpose':
            nn = onnx.helper.make_node(wi['type'], wi[IL], n.output, **wi['attr'])
            mod_node.append(nn)
        elif is_1in_eltwise(n.op_type):
            # insert 'Transpose' before 1in eltwise
            elt = target[wi[OL]]
            nn = onnx.helper.make_node('Transpose', n.input, n.output, **wi['attr'])
            elt[IL][0] = id
            mod_node.append(nn)
        elif n.op_type == 'Pad':
            # insert 'Transpose' before 'Pad'
            pad = target[wi[OL]]
            perm = wi['attr']['perm']
            if 'pads' in pad['attr']:
                pads = wi['attr']['pads']
                wi['attr']['pads'] = apply_perm_head_tail_pair(pads, perm)
            else:
                pads_id = pad[IL][1]
                pads = data[pads_id]['body']
                m = apply_perm_head_tail_pair(pads, perm)
                pad[IL][1] = update_initializer(data, shapes, pads_id, m, '.tr.opt')
            nn = onnx.helper.make_node('Transpose', [n.input[0]], n.output, **wi['attr'])
            pad[IL][0] = id
            mod_node.append(nn)
        elif n.op_type == 'Resize':
            # insert 'Transpose'before 'Resize'
            rsz = target[wi[OL]]
            ct_mode = rsz['attr'].get('coordinate_transformation_mode', 'half_pixel')
            perm = wi['attr']['perm']
            for i,e in enumerate(n.input):
                if e == '':
                    # opset13 style absent
                    continue
                if (e in data) and (data[e]['body'].size == 0):
                    # opset11 style absent
                    continue
                if e in data:
                    if i == 0: # body
                        tmp = copy.deepcopy(data[e]['body']).transpose(perm)
                    elif (i == 1) and (opset_ver >= 11): # roi
                        if not ct_mode == 'tf_crop_and_resize':
                            continue
                        dperm = convert_perm_for_head_tail_pair(perm)
                        tmp = data[e]['body'][dperm]
                    else: # scale or size
                        tmp = data[e]['body'][perm]
                    rsz[IL][i] = update_initializer(data, shapes, e, tmp, '.tr.opt')
                else:
                    if i == 0: # body
                        nn = onnx.helper.make_node('Transpose', [e], [id], **wi['attr'])
                        rsz[IL][0] = id
                        mod_node.append(nn)
                    elif (i == 1) and (opset_ver >= 11): # roi
                        if not ct_mode == 'tf_crop_and_resize':
                            continue
                        dperm = convert_perm_for_head_tail_pair(perm)
                        mod_id, append = insert_additional_node(blob_names, e, '.tr.gather', dperm)
                        rsz[IL][i] = mod_id
                        if append:
                            indices = register_gather_indices(data, shapes, mod_id, dperm)
                            nn = onnx.helper.make_node('Gather', [e, indices], [mod_id])
                            mod_node.append(nn)
                    else: # scale or size
                        mod_id, append = insert_additional_node(blob_names, e, '.tr.opt', perm)
                        rsz[IL][i] = mod_id
                        if append:
                            indices = register_gather_indices(data, shapes, mod_id, perm)
                            nn = onnx.helper.make_node('Gather', [e, indices], [mod_id])
                            mod_node.append(nn)
        elif n.op_type == 'Concat':
            # insert 'Transpose' before 'Concat'
            ccat = target[wi[OL]]
            perm = wi['attr']['perm']
            ccat['attr']['axis'] = adjust_concat_axis(ccat['attr']['axis'], perm)
            single = True
            for i,e in enumerate(n.input):
                if e in data:
                    tmp = copy.deepcopy(data[e]['body'])
                    tmp = tmp.transpose(perm)
                    mod_id = update_initializer(data, shapes, e, tmp, '.tr.opt')
                    ccat[IL][i] = mod_id
                elif single:
                    nn = onnx.helper.make_node('Transpose', [e], [id], perm=perm)
                    ccat[IL][i] = id
                    mod_node.append(nn)
                    single = False
                else:
                    mod_id, append = insert_additional_node(blob_names, e, '.tr.opt', perm)
                    ccat[IL][i] = mod_id
                    if append:
                        nn = onnx.helper.make_node('Transpose', [e], [mod_id], perm=perm)
                        mod_node.append(nn)
        elif n.op_type == 'PRelu':
            # insert 'Transpose'before 'PRelu'
            prelu = target[wi[OL]]
            perm = wi['attr']['perm']
            slope_id = prelu[IL][1]
            slope = data[slope_id]['body']
            num = len(perm) - len(slope.shape)
            if num > 0:
                mod_shape = [1] * num + list(slope.shape)
                slope = slope.reshape(mod_shape)
            slope = slope.transpose(perm)
            prelu[IL][1] = update_initializer(data, shapes, slope_id, slope, '.tr.opt')
            nn = onnx.helper.make_node('Transpose', [n.input[0]], n.output, **wi['attr'])
            prelu[IL][0] = id
            mod_node.append(nn)
        elif is_2in_eltwise(n.op_type):
            # insert 'Transpose' before 2in eltwise
            elt = target[wi[OL]]
            perm = wi['attr']['perm']
            single = True
            for i,e in enumerate(n.input):
                if e in data:
                    body = copy.deepcopy(data[e]['body'])
                    if body.size == 1:
                        continue
                    num = len(perm) - len(body.shape)
                    if num > 0:
                        mod_shape = [1] * num + list(body.shape)
                        body = body.reshape(mod_shape)
                    body = body.transpose(perm)
                    mod_id = update_initializer(data, shapes, e, body, '.tr.opt')
                    elt[IL][i] = mod_id
                elif single:
                    nn = onnx.helper.make_node('Transpose', [e], [id], **wi['attr'])
                    elt[IL][i] = id
                    mod_node.append(nn)
                    single = False
                else:
                    mod_id, append = insert_additional_node(blob_names, e, '.tr.opt', perm)
                    elt[IL][i] = mod_id
                    if append:
                        nn = onnx.helper.make_node('Transpose', [e], [mod_id], **wi['attr'])
                        mod_node.append(nn)

    return mod_node, True, count


def onnx_move_transpose(model):
    changed = True
    count = 0
    # get model info
    opset_ver = model.opset_import[0].version
    data = listup_initializer(model)
    shapes = ShapeAndTypeInfo(model)
    model_nodes = list(model.graph.node)
    while changed:
        model_nodes, changed, num = onnx_move_transpose_rp(model_nodes, model.graph.output, data, shapes, opset_ver)
        count += num

    if count == 0:
        return model

    print("    [i] report : move {} times 'Transpose' with 'onnx_move_transpose'".format(count))

    mod_initializer, mod_input = pack_initializer(data, model.graph.input)
    m_graph = onnx.helper.make_graph(model_nodes, model.graph.name, mod_input, model.graph.output, mod_initializer)
    m_model = onnx.helper.make_model(m_graph, producer_name=model.producer_name, producer_version=model.producer_version, opset_imports=model.opset_import)

    return m_model


def onnx_summarize_transpose(model):
    # list up initializer
    data = listup_initializer(model)

    output_id = '____graph_output____'

    OL = 'o'
    IL = 'i'
    # list up transpose node and connection
    tr_order = [] # transpose exec order
    trns = {}  # transpose node
    cnct = {}  # conection
    for n in model.graph.node:
        for o in n.output:
            for i in n.input:
                cnct.setdefault(i, [])
                cnct[i].append(o)
        id = n.output[0]
        if n.op_type == 'Transpose':
            tr_order.append(id)
            attr = listup_attribute(n)
            perm = numpy.array(attr['perm'])
            perm = numpy.where( perm < 0, perm+perm.size, perm )
            trns[id] = {
                OL: id, 
                IL: n.input[0], 
                'perm': perm, 
                'modified': False,
                'removed': False,
                't_node': [], # connected transpose node
                'o_node': [], # connected other node
            }
    for n in model.graph.output:
        cnct.setdefault(n.name, [])
        cnct[n.name].append(output_id)

    # check transpose connection and update
    for id,tr in trns.items():
        perm = tr['perm']
        t_node = tr['t_node']
        o_node = tr['o_node']
        for e in cnct[id]:
            if e in trns:
                t_node.append(e)
                mod = trns[e]
                mod['modified'] = True
                mod[IL] = tr[IL]
                mod['perm'] = perm[ mod['perm'] ]
            else:
                o_node.append(e)

    remove = {}
    update = {}
    tr_i2n = {}
    # check no output / do noting transpose
    for id in tr_order:
        tr = trns[id]
        if len(tr['o_node']) == 0:
            # no output
            tr['removed'] = True
            remove[id] = tr[IL]
            continue
        if numpy.array_equal(tr['perm'], range(tr['perm'].size)):
            # do nothing
            tr['removed'] = True
            remove[id] = tr[IL]
            continue
        ii = tr[IL]
        tr_i2n.setdefault(ii, [])
        tr_i2n[ii].append(id)

    # check duplicated transpsoe
    for x in tr_i2n.values():
        for i,c in enumerate(x):
            dup_name = None
            cperm = trns[c]['perm']
            for t in x[i+1:]:
                tperm = trns[t]['perm']
                if numpy.array_equal(cperm, tperm):
                    dup_name = t
                    break
            if dup_name:
                trns[dup_name]['removed'] = True
                remove[dup_name] = c

    for id in tr_order:
        tr = trns[id]
        if tr['removed']:
            continue
        if tr['modified']:
            update[id] = tr

    if (len(remove) + len(update)) == 0:
        # no update
        return model

    if len(remove) > 0:
        print("    [i] report : remove {} 'Transpose' with 'onnx_summarize_transpose'".format(len(remove)))
    if len(update) > 0:
        print("    [i] report : modify {} 'Transpose' with 'onnx_summarize_transpose'".format(len(update)))

    mod_node = []
    for n in model.graph.node:
        id = n.output[0]
        if id in update:
            tr = update[id]
            n = onnx.helper.make_node('Transpose', [tr[IL]], [id], perm=tr['perm'])
        elif id in remove:
            continue
        for i,e in enumerate(n.input):
            while e in remove:
                e = remove[e]
            n.input[i] = e
        mod_node.append(n)            

    # adjust blob name related graph output
    adjust_graph_output(mod_node, remove, model)

    mod_initializer, mod_input = pack_initializer(data, model.graph.input)
    m_graph = onnx.helper.make_graph(mod_node, model.graph.name, mod_input, model.graph.output, mod_initializer)
    m_model = onnx.helper.make_model(m_graph, producer_name=model.producer_name, producer_version=model.producer_version, opset_imports=model.opset_import)

    return m_model


def onnx_treat_stylegan(model):
    # get model info
    data = listup_initializer(model)
    shapes = ShapeAndTypeInfo(model)

    OL = 'o'
    IL = 'i'
    # list up replace node candidate
    cand = {}  # candidate
    cnct = {}  # conection
    for n in model.graph.node:
        for o in n.output:
            for i in n.input:
                cnct.setdefault(i, [])
                cnct[i].append(o)
        nmo = n.output[0]  # node main output
        if (n.op_type == 'Reshape'):
            cand[nmo] = {'type': n.op_type, OL: nmo, IL: n.input}
        elif (n.op_type == 'Pad'):
            if not (len(n.input) == 3):
                continue
            if not (n.input[0] in cand and n.input[1] in data and n.input[2] in data):
                continue
            ut = cand[n.input[0]]
            if not (ut['type'] == 'Reshape'):
                continue
            attr = listup_attribute(n)
            if not(attr == {} or attr['mode'] == b'constant'):
                continue
            if not n.input[2] in data:
                continue
            if not data[n.input[2]]['body'] == [0]:
                continue
            cand[nmo] = {'type': n.op_type, OL: nmo, IL: n.input}
        elif (n.op_type == 'Conv'):
            if not (n.input[0] in cand and n.input[1] in data):
                continue
            if not (data[n.input[1]]['body'].shape == (1, 1, 4, 4)):
                continue
            ut = cand[n.input[0]]
            if not (ut['type'] == 'Pad'):
                continue
            cand[nmo] = {'type': n.op_type, OL: nmo, IL: n.input, 'attr': listup_attribute(n)}

    count = 0
    target = {}
    for n in cand.values():
        if not (n['type'] == 'Conv'):
            continue
        wi = n
        wl = []
        re_count = 0
        pad_count = 0
        conv_input_shape = shapes.get_shape(n[IL][0])
        while (wi[IL][0] in cand):
            if (len(cnct[wi[IL][0]]) > 1):
                break
            wl.append(wi)
            wi = cand[wi[IL][0]]
            if wi['type'] == 'Reshape':
                re_count += 1
            if wi['type'] == 'Pad':
                if (pad_count == 0):
                    if len(data[wi[IL][1]]['body']) < 8:
                        continue
                    if not (all(data[wi[IL][1]]['body'][0:4] == [0, 0, 2, 2]) and all(data[wi[IL][1]]['body'][4:] == [0, 0, 1, 1])):
                        continue
                pad_count += 1
        if not wi['type'] == 'Reshape' or re_count != 2 or pad_count != 2:
            continue
        if not shapes.is_inferred(wi[IL][0]):
            continue
        input_shape = shapes.get_shape(wi[IL][0])
        if len(input_shape) < 4 or len(conv_input_shape) < 4:
            continue
        if not ((input_shape[2] * 2 + 3 == conv_input_shape[2]) and (input_shape[3] * 2 + 3 == conv_input_shape[3])):
            continue
        input_num = wi[IL][0]
        wl.append(wi)
        count += 1
        for wi in wl:
            if (wi['type'] == 'Conv'):
                wi['reshape_shape'] = input_shape
                wi['reshape_input'] = input_num
            target[wi[OL]] = wi

    if count == 0:
        return model

    print("    [i] report : replace {} StyleGAN specific sequence to Reshape+ConvTranspose with 'onnx_treat_stylegan'".format(count))

    mod_node = []
    for n in model.graph.node:
        id = n.output[0]

        if not (id in target):
            mod_node.append(n)
            continue

        wi = target[id]
        if (wi['type'] == 'Reshape' or wi['type'] == 'Pad'):
            continue
        else:
            reshape_input = wi[IL][0] + '.reshape.opt'
            reshape_shape = numpy.array([-1, 1] + wi['reshape_shape'][2:]).astype(numpy.int64)
            convtr_input = wi[IL][0] + '.opt'
            data[reshape_input] = {'ref': 1, 'id': reshape_input, 'body': reshape_shape}
            nx = onnx.helper.make_node('Reshape', [wi['reshape_input'], reshape_input], [convtr_input])
            mod_node.append(nx)

            attr = {}
            attr['kernel_shape'] = [4, 4]
            attr['pads'] = [1, 1, 1, 1]
            attr['strides'] = [2, 2]
            data[wi[IL][1]]['body'] = numpy.flip(data[wi[IL][1]]['body'], axis=[2, 3])
            nx = onnx.helper.make_node('ConvTranspose', [convtr_input, wi[IL][1]], [wi[OL]], **attr)
            mod_node.append(nx)

    mod_initializer, mod_input = pack_initializer(data, model.graph.input)
    m_graph = onnx.helper.make_graph(mod_node, model.graph.name, mod_input, model.graph.output, mod_initializer)
    m_model = onnx.helper.make_model(m_graph, producer_name=model.producer_name, producer_version=model.producer_version, opset_imports=model.opset_import)

    return m_model


def onnx_convert_to_batchnorm(model):
    # list up initializers
    data = listup_initializer(model)

    OL = 'o'
    IL = 'i'
    # list up mul|add node
    node_mul = {}
    node_add = {}
    node_conv = {}
    cnct = {}
    for n in model.graph.node:
        for o in n.output:
            for i in n.input:
                cnct.setdefault(i, [])
                cnct[i].append(o)
        if n.op_type == 'Mul':
            node_mul[n.output[0]] = {OL: n.output[0], IL: n.input}
        elif n.op_type == 'Add':
            node_add[n.output[0]] = {OL: n.output[0], IL: n.input}
        elif n.op_type == 'Conv':
            node_conv[n.output[0]] = {OL: n.output[0], IL: n.input}

    # list up convert candidate
    cand = {}
    for na in node_add.values():
        ac = None
        nm = None
        for i in na[IL]:
            ac = data.get(i, ac)
            nm = node_mul.get(i, nm)
            if i in node_mul:
                node_mul.pop(i)
        if (ac == None) or (nm == None):
            continue
        if len(cnct[nm[OL]]) > 1:
            continue
        mc = None
        ac_data = ac['body']
        if len(ac_data.shape) < 3:
            continue
        if ac_data.shape[1] != ac_data.size:
            continue
        for i in nm[IL]:
            mc = data.get(i, mc)
            if mc == None:
                continue
            mc_data = mc['body']
            if ac_data.shape == mc_data.shape:
                break
            mc = None
        if mc == None:
            continue
        mi = nm[IL][0] if (nm[IL][0] != mc['id']) else nm[IL][1]
        cand[nm[OL]] = {'type': 'Mul', 'name': nm[OL]}
        cand[na[OL]] = {'type': 'Add', 'name': na[OL], 'ac': ac['id'], 'mc': mc['id'], 'mi': mi, 'C': ac_data.shape[1]}
    # independent and connect conv mul is target
    for nm in node_mul.values():
        nm_data = None
        cnv_data = None
        cnv = None
        for i in nm[IL]:
            nm_data = data.get(i, nm_data)
            cnv_data = node_conv.get(i, cnv_data)
        # check have connection to conv
        if (nm_data == None) or (cnv_data == None):
            continue
        nm_body = nm_data['body']
        if len(nm_body.shape) < 3:
            continue
        if nm_body.shape[1] != nm_body.size:
            continue
        mi = nm[IL][0] if (nm[IL][0] != nm_data['id']) else nm[IL][1]
        cand[nm[OL]] = {'type': 'Mul', 'name': nm[OL], 'mc': nm_data['id'], 'mi': mi, 'C': nm_body.shape[1]}
    count = int(len(cand) / 2)
    if (count == 0):
        # no available convertion
        return model

    print("    [i] report : fuse {} Mul+Add and (Conv->)Mul to BatchNormalization.".format(count))

    mod_node = []
    for n in model.graph.node:
        id = n.output[0]
        if (not id in cand):
            # keep node
            mod_node.append(n)
            continue

        if (n.op_type == 'Mul'):
            if len(cand[id]) < 3:
                # erase node
                continue
            else:
                ch = cand[id]['C']
                bias = 'opt.bias.size{}'.format(ch)
                bias_data = numpy.zeros((ch,), dtype=numpy.float32)
                data.setdefault(bias, {'ref': 0, 'id': bias, 'body': bias_data})
                data[bias]['ref'] += 1
        elif (n.op_type == 'Add'):
            bias = cand[id]['ac']
            ch = cand[id]['C']
            data[bias]['body'] = data[bias]['body'].reshape((ch,))
        x = cand[id]['mi']
        scale = cand[id]['mc']
        mean = 'opt.zero.size{}'.format(ch)
        var = 'opt.one.size{}'.format(ch)
        data[scale]['body'] = data[scale]['body'].reshape((ch,))
        mod_node.append(onnx.helper.make_node("BatchNormalization", [x, scale, bias, mean, var], [id], epsilon=0.0))
        mean_data = numpy.zeros((ch,), dtype=numpy.float32)
        data.setdefault(mean, {'ref': 0, 'id': mean, 'body': mean_data})
        data[mean]['ref'] += 1
        var_data = numpy.ones((ch,), dtype=numpy.float32)
        data.setdefault(var, {'ref': 0, 'id': var, 'body': var_data})
        data[var]['ref'] += 1

    mod_initializer, mod_input = pack_initializer(data, model.graph.input)
    m_graph = onnx.helper.make_graph(mod_node, model.graph.name, mod_input, model.graph.output, mod_initializer)
    m_model = onnx.helper.make_model(m_graph, producer_name=model.producer_name, producer_version=model.producer_version, opset_imports=model.opset_import)

    return m_model


def get_conv_output_channel(out_id, weight_id, data, shapes: ShapeAndTypeInfo):
    C = 0
    if shapes.is_inferred(out_id) and len(shapes.get_shape(out_id)) >= 3:
        C = shapes.get_shape(out_id)[1]
    if (C == 0) and (weight_id in data):
        C = data[weight_id]['body'].shape[1]
    return C


def onnx_fuse_bias_into_conv(model):
    # list up initializer + shape inference
    data = listup_initializer(model)
    shapes = ShapeAndTypeInfo(model)

    OL = 'o'
    IL = 'i'
    # list up conv|add|sub node + blob connection dictionary
    node_conv = {}
    node_bias = {}
    cnct_i2o = {}  # connection dict (key:input, value:list_of_outputs)
    for n in model.graph.node:
        for o in n.output:
            for i in n.input:
                cnct_i2o.setdefault(i, [])
                cnct_i2o[i].append(o)
        id = n.output[0]
        if n.op_type == 'Conv':
            node_conv[id] = {OL: id, IL: n.input}
        elif n.op_type in ['Add', 'Sub']:
            node_bias[id] = {OL: id, IL: n.input, 'op_type': n.op_type}

    # list up convert candidate and modify bias
    cand = {}
    replace = {}
    for nc in node_conv.values():
        C = get_conv_output_channel(nc[OL], nc[IL][1], data, shapes)
        if C == 0:
            continue
        id = nc[OL]
        if not id in cnct_i2o:
            continue
        bias_target = None
        bias_coef = None
        # track node connection
        while id in cnct_i2o:
            if len(cnct_i2o[id]) > 1:
                break
            nxt_id = cnct_i2o[id][0]
            if not nxt_id in node_bias:
                break
            for i in node_bias[nxt_id][IL]:
                bias_coef = data.get(i, bias_coef)
            id = nxt_id
            if bias_coef == None:
                continue
            if (bias_coef['body'].size == C) and (bias_coef['body'].shape[1] == C):
                bias_target = node_bias[nxt_id]
                break
            bias_coef = None
        if (bias_coef == None) or (bias_target == None):
            continue
        org_bias = None
        if len(nc[IL]) >= 3:
            if not nc[IL][2] in data:
                continue
            org_bias = data[nc[IL][2]]
            if org_bias['ref'] > 1:
                continue
        if org_bias:  # conv has original bias
            bias_data = org_bias['body']
            if bias_target['op_type'] == 'Add':
                org_bias['body'] = bias_data + bias_coef['body'].reshape((C,))
            else:
                org_bias['body'] = bias_data - bias_coef['body'].reshape((C,))
            bias_coef['ref'] -= 1
        else:  # conv has no original bias
            if bias_coef['ref'] > 1:
                continue
            bias_data = bias_coef['body']
            if bias_target['op_type'] == 'Add':
                bias_coef['body'] = bias_data.reshape((C,))
            else:
                bias_coef['body'] = numpy.zeros((C,)) - bias_data.reshape((C,))
            cand[nc[OL]] = {'type': 'Conv', OL: nc[OL], IL: [nc[IL][0], nc[IL][1], bias_coef['id']]}
        cand[bias_target[OL]] = {'type': 'Add', OL: bias_target[OL]}
        ro = bias_target[OL]
        il = copy.deepcopy(bias_target[IL])
        il.remove(bias_coef['id'])
        ri = il[0]
        replace[ro] = ri

    count = len(replace)
    if (count == 0):
        # no available fuse node
        return model

    print("    [i] report : fuse {} bias(Add or Sub) into Conv.".format(count))

    mod_node = []
    for n in model.graph.node:
        id = n.output[0]

        # update node input : for all node
        for i in range(len(n.input)):
            k = n.input[i]
            if (k in replace.keys()):
                n.input[i] = replace[k]

        if not id in cand:
            mod_node.append(n)
            continue

        if n.op_type == 'Add':
            # remove node
            continue

        # op_type should be 'Conv'
        x = cand[id]
        for i in range(len(x[IL])):
            k = x[IL][i]
            if (k in replace.keys()):
                x[IL][i] = replace[k]
        attr = listup_attribute(n)
        nx = onnx.helper.make_node('Conv', x[IL], [x[OL]], **attr)
        mod_node.append(nx)

    # adjust blob name related graph output
    adjust_graph_output(mod_node, replace, model)

    mod_initializer, mod_input = pack_initializer(data, model.graph.input)
    m_graph = onnx.helper.make_graph(mod_node, model.graph.name, mod_input, model.graph.output, mod_initializer)
    m_model = onnx.helper.make_model(m_graph, producer_name=model.producer_name, producer_version=model.producer_version, opset_imports=model.opset_import)

    return m_model


def onnx_fuse_bn_into_conv(model):
    # list up initializer
    data = listup_initializer(model)

    OL = 'o'
    IL = 'i'
    # list up conv|bn node
    node_conv = {}
    node_bn = {}
    cnct = {}
    for n in model.graph.node:
        for o in n.output:
            for i in n.input:
                cnct.setdefault(i, [])
                cnct[i].append(o)
        if n.op_type == 'Conv':
            node_conv[n.output[0]] = {'type': 'Conv', OL: n.output[0], IL: n.input}
        if n.op_type == 'ConvTranspose':
            node_conv[n.output[0]] = {'type': 'ConvTranspose', OL: n.output[0], IL: n.input}
        elif n.op_type == 'BatchNormalization':
            epsilon = pick_attribute(n, 'epsilon')
            ev = epsilon.f if epsilon else 1e-5
            node_bn[n.output[0]] = {OL: n.output[0], IL: n.input, 'epsilon': ev}

    # list up convert candidate and modify weight/bias
    cand = {}
    replace = {}
    count_c = 0
    for nb in node_bn.values():
        bn_in = nb[IL]
        if not bn_in[0] in node_conv:
            continue
        if not bn_in[1] in data:
            continue
        bn_scale = data[bn_in[1]]
        if not bn_in[2] in data:
            continue
        bn_bias = data[bn_in[2]]
        if not bn_in[3] in data:
            continue
        bn_mean = data[bn_in[3]]
        if not bn_in[4] in data:
            continue
        bn_var = data[bn_in[4]]
        if len(cnct[bn_in[0]]) > 1:
            continue
        nc = node_conv[bn_in[0]]
        cv_in = nc[IL]
        if not cv_in[1] in data:
            continue
        cv_weight = data[cv_in[1]]
        scale_data = bn_scale['body'] / numpy.sqrt(bn_var['body'] + nb['epsilon'])
        weight_data = cv_weight['body']
        if nc['type'] == 'Conv':
            ws_shape = [scale_data.size] + [1] * (weight_data.ndim - 1)
        elif nc['type'] == 'ConvTranspose':
            ws_shape = [1, scale_data.size] + [1] * (weight_data.ndim - 2)
        weight_data = weight_data * scale_data.reshape(ws_shape)
        if len(cv_in) >= 3:
            # convtranspose has bias
            if not cv_in[2] in data:
                continue
            cv_bias = data[cv_in[2]]
            bias_data = (cv_bias['body'] - bn_mean['body']) * scale_data + bn_bias['body']
            cv_bias['body'] = bias_data
        else:
            # convtranspose has no bias
            bias_data = bn_bias['body'] - bn_mean['body'] * scale_data
            bias_name = nc[OL] + '.opt.bias'
            data.setdefault(bias_name, {'ref': 0, 'id': bias_name, 'body': bias_data})
            data[bias_name]['ref'] += 1
            cand[nc[OL]] = {'type': nc['type'], OL: nc[OL], IL: [cv_in[0], cv_in[1], bias_name]}
        cv_weight['body'] = weight_data
        bn_scale['ref'] -= 1
        bn_bias['ref'] -= 1
        bn_mean['ref'] -= 1
        bn_var['ref'] -= 1
        cand[nb[OL]] = {'type': 'BatchNormalization', OL: nb[OL]}
        replace[nb[OL]] = nc[OL]
        if nc['type'] == 'Conv':
            count_c += 1

    count = len(replace)
    if (count == 0):
        # no available fuse node
        return model

    if count_c > 0:
        print("    [i] report : fuse {} BatchNormalization to Conv.".format(count_c))
    if count - count_c > 0:
        print("    [i] report : fuse {} BatchNormalization to ConvTranspose.".format(count - count_c))

    mod_node = []
    for n in model.graph.node:
        id = n.output[0]

        # update node input : for all node
        for i in range(len(n.input)):
            k = n.input[i]
            if (k in replace.keys()):
                n.input[i] = replace[k]

        if not id in cand:
            mod_node.append(n)
            continue

        if n.op_type == 'BatchNormalization':
            # remove node
            continue

        # op_type should be 'Conv' or 'ConvTranspose'
        x = cand[id]
        for i in range(len(x[IL])):
            k = x[IL][i]
            if (k in replace.keys()):
                x[IL][i] = replace[k]
        attr = listup_attribute(n)
        nx = onnx.helper.make_node(x['type'], x[IL], [x[OL]], **attr)
        mod_node.append(nx)

    # adjust blob name related graph output
    adjust_graph_output(mod_node, replace, model)

    mod_initializer, mod_input = pack_initializer(data, model.graph.input)
    m_graph = onnx.helper.make_graph(mod_node, model.graph.name, mod_input, model.graph.output, mod_initializer)
    m_model = onnx.helper.make_model(m_graph, producer_name=model.producer_name, producer_version=model.producer_version, opset_imports=model.opset_import)

    return m_model


def remove_needless_node(replace, data, node, ref_count, opset_ver):

    id = node.output[0]

    if (node.op_type == 'Identity'):
        replace[id] = node.input[0]
        return True

    if (node.op_type == 'Unsqueeze'):
        sid = node.input[0]
        if not (sid in data):
            return False
        body = data[sid]['body']
        if opset_ver >= 13:
            if not (node.input[1] in data):
                return False
            ax = data[node.input[1]]['body']
        else:
            ax = pick_attribute(node, 'axes').ints
        os = list(body.shape)
        dim = [0] * (len(os) + len(ax))
        for i in ax:
            if (i < 0):
                i += len(dim)
            dim[i] = 1
        for i in range(len(dim)):
            if (dim[i] == 0):
                dim[i] = os.pop(0)
        body = body.reshape(dim)
        data[id] = {'ref': ref_count, 'id': id, 'body': body}
        data[sid]['ref'] -= 1
        return True

    if (node.op_type == 'Transpose'):
        perm = pick_attribute(node, 'perm')
        if not (perm == None):
            perm = perm.ints
            if check_needless_transpose(perm):
                replace[id] = node.input[0]
                return True
        sid = node.input[0]
        if not (sid in data):
            return False
        d = data[node.input[0]]
        if (d['ref'] > 1) and (d['body'].size >= (512 * 1024)):
            return False
        if perm == None:
            # default : reverse axes
            ddim = d['body'].ndim
            perm = [i for i in range(ddim - 1, -1, -1)]
        data[id] = {'ref': ref_count, 'id': id, 'body': d['body'].transpose(perm)}
        return True

    if (node.op_type == 'Concat'):
        if (len(node.input) == 1):
            replace[id] = node.input[0]
            return True
        for i in node.input:
            if not (i in data):
                return False
        axis = pick_attribute(node, 'axis').i
        rb = [data[i]['body'] for i in node.input]
        data[id] = {'ref': ref_count, 'id': id, 'body': numpy.concatenate(rb, axis=axis)}
        for i in range(len(node.input)):
            data[node.input[i]]['ref'] -= 1
        return True

    if (node.op_type == 'Slice'):
        if opset_ver < 10:
            # Slice-1 is unsupported
            return False
        start_id = node.input[1]
        end_id = node.input[2]
        axes_id = node.input[3] if len(node.input) >= 4 else None
        step_id = node.input[4] if len(node.input) >= 5 else None
        cnv_flg = True
        for i in node.input:
            if not i in data:
                cnv_flg = False
                break
        if cnv_flg:
            ary = data[node.input[0]]['body']
            start = data[start_id]['body']
            end = data[end_id]['body']
            axes = numpy.arange(len(start), dtype=numpy.int64) if len(node.input) < 4 else data[axes_id]['body']
            step = numpy.ones((len(start),), dtype=numpy.int64) if len(node.input) < 5 else data[step_id]['body']
            data_dim = ary.ndim
            slc = [slice(None)] * data_dim
            for ax, s, e, t in zip(axes, start, end, step):
                slc[ax] = slice(s, e, t)
            data[id] = {'ref': ref_count, 'id': id, 'body': ary[tuple(slc)]}
            for i in node.input:
                data[i]['ref'] -= 1
            return True
        if step_id:
            if not step_id in data:
                return False
            for e in data[step_id]['body'].flat:
                if e != 1:
                    return False
        if not start_id in data:
            return False
        for e in data[start_id]['body'].flat:
            if e != 0:
                return False
        if not end_id in data:
            return False
        for e in data[end_id]['body'].flat:
            if e < (2**31 - 1):
                return False
        data[start_id]['ref'] -= 1
        data[end_id]['ref'] -= 1
        if step_id:
            data[step_id]['ref'] -= 1
        if axes_id and (axes_id in data):
            data[axes_id]['ref'] -= 1
        replace[id] = node.input[0]
        return True

    if (node.op_type == 'Cast'):
        sid = node.input[0]
        if not (sid in data):
            return False
        to = pick_attribute(node, 'to').i
        if (to != onnx.TensorProto.FLOAT) and (to != onnx.TensorProto.INT32) and (to != onnx.TensorProto.INT64):
            return False
        body = data[sid]['body'].astype(convert_dtype_tensor2np(to))
        data[id] = {'ref': ref_count, 'id': id, 'body': body}
        data[sid]['ref'] -= 1
        return True

    if (node.op_type == 'Add'):
        for i in node.input:
            if not i in data:
                return False
        a_id = node.input[0]
        b_id = node.input[1]
        if data[a_id]['body'].dtype != data[b_id]['body'].dtype:
            return False
        data[id] = {'ref': ref_count, 'id': id, 'body': numpy.add(data[a_id]['body'], data[b_id]['body'])}
        data[a_id]['ref'] -= 1
        data[b_id]['ref'] -= 1
        return True

    if (node.op_type == 'Sub'):
        for i in node.input:
            if not i in data:
                return False
        a_id = node.input[0]
        b_id = node.input[1]
        if data[a_id]['body'].dtype != data[b_id]['body'].dtype:
            return False
        data[id] = {'ref': ref_count, 'id': id, 'body': numpy.subtract(data[a_id]['body'], data[b_id]['body'])}
        data[a_id]['ref'] -= 1
        data[b_id]['ref'] -= 1
        return True

    if (node.op_type == 'Mul'):
        for i in node.input:
            if not i in data:
                return False
        a_id = node.input[0]
        b_id = node.input[1]
        if data[a_id]['body'].dtype != data[b_id]['body'].dtype:
            return False
        data[id] = {'ref': ref_count, 'id': id, 'body': numpy.multiply(data[a_id]['body'], data[b_id]['body'])}
        data[a_id]['ref'] -= 1
        data[b_id]['ref'] -= 1
        return True

    if (node.op_type == 'Div'):
        for i in node.input:
            if not i in data:
                return False
        a_id = node.input[0]
        b_id = node.input[1]
        if len(numpy.flatnonzero(data[b_id]['body'])) < len(data[b_id]['body'].flatten()):
            return False  # detect zero from denominator
        data[id] = {'ref': ref_count, 'id': id, 'body': numpy.divide(data[a_id]['body'], data[b_id]['body'])}
        data[a_id]['ref'] -= 1
        data[b_id]['ref'] -= 1
        return True

    if (node.op_type == 'Mod'):
        for i in node.input:
            if not i in data:
                return False
        a_id = node.input[0]
        b_id = node.input[1]
        if data[a_id]['body'].dtype != data[b_id]['body'].dtype:
            return False
        data[id] = {'ref': ref_count, 'id': id, 'body': numpy.fmod(data[a_id]['body'], data[b_id]['body'])}
        data[a_id]['ref'] -= 1
        data[b_id]['ref'] -= 1
        return True

    if (node.op_type == 'Sin'):
        if not node.input[0] in data:
            return False
        a_id = node.input[0]
        data[id] = {'ref': ref_count, 'id': id, 'body': numpy.sin(data[a_id]['body'])}
        data[a_id]['ref'] -= 1
        return True

    if (node.op_type == 'Cos'):
        if not node.input[0] in data:
            return False
        a_id = node.input[0]
        data[id] = {'ref': ref_count, 'id': id, 'body': numpy.cos(data[a_id]['body'])}
        data[a_id]['ref'] -= 1
        return True

    if (node.op_type == 'ReduceMean') or (node.op_type == 'ReduceMax') or (node.op_type == 'ReduceMin'):
        sid = node.input[0]
        if not sid in data:
            return False
        body = data[sid]['body']
        axes = pick_attribute(node, 'axes')
        if axes == None:
            axes = range(body.ndim)
        else:
            axes = axes.ints
        keepdims = pick_attribute(node, 'keepdims')
        if keepdims == None:
            keepdims = True
        else:
            keepdims = (keepdims.i != 0)
        if node.op_type == 'ReduceMean':
            body = numpy.mean(body, axis=tuple(axes), keepdims=keepdims)
        elif node.op_type == 'ReduceMax':
            body = numpy.maximum.reduce(body, axis=tuple(axes), keepdims=keepdims)
        elif node.op_type == 'ReduceMin':
            body = numpy.minimum.reduce(body, axis=tuple(axes), keepdims=keepdims)
        data[id] = {'ref': ref_count, 'id': id, 'body': body}
        data[sid]['ref'] -= 1
        return True

    if (node.op_type == 'Sqrt'):
        sid = node.input[0]
        if not sid in data:
            return False
        body = data[sid]['body']
        body = numpy.sqrt(body)
        data[id] = {'ref': ref_count, 'id': id, 'body': body}
        data[sid]['ref'] -= 1
        return True

    if (node.op_type == 'Split'):
        if not (node.input[0] in data):
            return False
        sp = len(node.output)
        attr = pick_attribute(node, 'axis')
        axis = attr.i if attr != None else 0
        attr = pick_attribute(node, 'split')
        src_dat = data[node.input[0]]['body']
        if src_dat.ndim <= axis:
            return False

        split = attr.ints if attr != None else [src_dat.shape[axis] // sp] * sp
        idx = 0
        dat_ = src_dat if axis == 0 else src_dat.swapaxes(0, axis)
        for cnt, sp_len in enumerate(split):
            sp_dat = dat_[idx:(idx + sp_len)]
            if axis != 0:
                sp_dat = sp_dat.swapaxes(0, axis)
            idx += sp_len
            add_id = node.input[0] + '__part' + str(cnt)
            data[add_id] = {'ref': 1, 'id': add_id, 'body': sp_dat}
            replace[node.output[cnt]] = add_id

        data[node.input[0]]['ref'] -= 1
        return True

    if (node.op_type == 'Squeeze'):
        if not (node.input[0] in data):
            return False
        rb = data[node.input[0]]['body']
        if opset_ver >= 13:
            if (len(node.input) < 2) or not (node.input[1] in data):
                return False
            ax = data[node.input[1]]['body']
        else:
            ax = pick_attribute(node, 'axes')
            if ax == None:
                return False
            else:
                ax = ax.ints
        if rb.ndim < max(ax):
            return False
        ax = tuple(ax)
        data[node.input[0]]['body'] = numpy.squeeze(rb, ax)
        replace[id] = node.input[0]
        return True

    if (node.op_type == 'Range'):
        for i in node.input:
            if not i in data:
                return False
        start = data[node.input[0]]['body']
        limit = data[node.input[1]]['body']
        delta = data[node.input[2]]['body']
        res = numpy.arange(start, limit, delta)
        data[id] = {'ref': ref_count, 'id': id, 'body': res}
        for i in node.input:
            data[i]['ref'] -= 1
        return True

    if (node.op_type == 'Tile'):
        if not (node.input[1] in data):
            return False
        rep = data[node.input[1]]['body']
        if node.input[0] in data:
            ind = data[node.input[0]]['body']
            data[id] = {'ref': ref_count, 'id': id, 'body': numpy.tile(ind, rep)}
            data[node.input[0]]['ref'] -= 1
            data[node.input[1]]['ref'] -= 1
            return True
        if numpy.array_equal(rep, numpy.ones_like(rep)):
            data[node.input[1]]['ref'] -= 1
            replace[id] = node.input[0]
            return True

    if (node.op_type == 'Reshape'):
        for i in node.input:
            if not i in data:
                return False
        db = data[node.input[0]]['body']
        shape = tuple(data[node.input[1]]['body'])
        data[id] = {'ref': ref_count, 'id': id, 'body': numpy.reshape(db, shape)}
        data[node.input[0]]['ref'] -= 1
        data[node.input[1]]['ref'] -= 1
        return True

    if (node.op_type == 'ConstantOfShape'):
        for i in node.input:
            if not i in data:
                return False
        shape = data[node.input[0]]['body']
        value = pick_attribute(node, 'value')
        value = onnx.numpy_helper.to_array(value.t) if value != None else numpy.array([0], dtype=numpy.int64)
        data[id] = {'ref': ref_count, 'id': id, 'body': numpy.full(shape, value)}
        data[node.input[0]]['ref'] -= 1
        return True

    if (node.op_type == 'Pad'):
        if opset_ver < 11:
            value = pick_attribute(node, 'pads').ints
        else:
            if not (node.input[1] in data):
                return False
            value = data[node.input[1]]['body']
        if sum(value) == 0:
            # replace blob
            replace[id] = node.input[0]
            # remove
            for i in node.input:
                if i in data:
                    data[i]['ref'] -= 1

            return True

    if (node.op_type == 'Reciprocal'):
        if not node.input[0] in data:
            return False
        value = numpy.reciprocal(data[node.input[0]]['body'])
        data[id] = {'ref': ref_count, 'id': id, 'body': value}
        data[node.input[0]]['ref'] -= 1
        return True

    if (node.op_type == 'ReduceSum'):
        for i in node.input:
            if not i in data:
                return False
        keepdims = pick_attribute(node, 'keepdims')
        if keepdims == None:
            keepdims = True
        else:
            keepdims = keepdims.i != 0
        if opset_ver >= 13:
            if len(node.input) == 1:
                value = numpy.sum(data[node.input[0]]['body'], axis=None, keepdims=keepdims)
            else:
                axes = data[node.input[1]]['body']
                value = numpy.sum(data[node.input[0]]['body'], axis=axes, keepdims=keepdims)
        else:
            axes = pick_attribute(node, 'axes')
            value = numpy.sum(data[node.input[0]]['body'], axis=tuple(axes.ints), keepdims=keepdims)
        data[id] = {'ref': ref_count, 'id': id, 'body': value}
        for i in node.input:
            data[i]['ref'] -= 1
        return True

    if (node.op_type == 'Greater'):
        for i in node.input:
            if not i in data:
                return False
        a = data[node.input[0]]['body']
        b = data[node.input[1]]['body']
        value = numpy.greater(a, b)
        data[id] = {'ref': ref_count, 'id': id, 'body': value}
        for i in node.input:
            data[i]['ref'] -= 1
        return True

    if (node.op_type == 'Max'):
        for i in node.input:
            if not i in data:
                return False
        max = data[node.input[0]]['body']
        for i in range(1, len(node.input)):
            max = numpy.maximum(max, data[node.input[1]]['body'])
        data[id] = {'ref': ref_count, 'id': id, 'body': max}
        for i in node.input:
            data[i]['ref'] -= 1
        return True

    if (node.op_type == 'Where'):
        for i in node.input:
            if not i in data:
                return False
        cond = data[node.input[0]]['body']
        a = data[node.input[1]]['body']
        b = data[node.input[2]]['body']
        value = numpy.where(cond, a, b)
        data[id] = {'ref': ref_count, 'id': id, 'body': value}
        for i in node.input:
            data[i]['ref'] -= 1
        return True

    if (node.op_type == 'Xor'):
        if node.input[0] in data:
            ipt = data[node.input[0]]['body']
            if ipt == False:
                data[node.input[0]]['ref'] -= 1
                replace[id] = node.input[1]
                return True
        if node.input[1] in data:
            ipt = data[node.input[1]]['body']
            if ipt == False:
                data[node.input[1]]['ref'] -= 1
                replace[id] = node.input[0]
                return True

    if (node.op_type == 'Expand'):
        if node.input[0] in data and node.input[1] in data:
            inputt = data[node.input[0]]['body']
            shapet = data[node.input[1]]['body']
            value = inputt * numpy.ones(shapet, dtype=inputt.dtype)
            if value.size >= (64 * 1024):
                # too large to convert
                return False
            data[id] = {'ref': ref_count, 'id': id, 'body': value}
            for i in node.input:
                data[i]['ref'] -= 1
            return True

    return False


def onnx_reduce_needless_node(model):

    opset_ver = model.opset_import[0].version

    # list up initializer
    data = listup_initializer(model)

    output_id = '____graph_output____'

    graph_input = {}
    for gi in model.graph.input:
        graph_input.setdefault(gi.name, True)

    cnct = {}  # node connection, key=input_id, val=[output_id]
    for n in model.graph.node:
        for o in n.output:
            for i in n.input:
                cnct.setdefault(i, [])
                cnct[i].append(o)
    for n in model.graph.output:
        cnct.setdefault(n.name, [])
        cnct[n.name].append(output_id)

    mod_node = []
    count = {}
    replace = {}
    for n in model.graph.node:
        for i in range(len(n.input)):
            k = n.input[i]
            if k in replace.keys():
                n.input[i] = replace[k]

        id = n.output[0]

        if id in cnct:
            io_direct_connect = False
            if (id in cnct) and (output_id in cnct[id]):
                for k in n.input:
                    if k in graph_input:
                        io_direct_connect = True

            if (not io_direct_connect) and (id in cnct) and remove_needless_node(replace, data, n, len(cnct[id]), opset_ver):
                count.setdefault(n.op_type, 0)
                count[n.op_type] += 1
                continue

        n = adjust_node_with_subgraph(n, replace)
        mod_node.append(n)

    if len(count) == 0:
        # no removable node
        return model

    # adjust blob name related graph output
    adjust_graph_output(mod_node, replace, model)

    for k in count.keys():
        print("    [i] report : remove {0} {1} node.".format(count[k], k))

    mod_initializer, mod_input = pack_initializer(data, model.graph.input)
    m_graph = onnx.helper.make_graph(mod_node, model.graph.name, mod_input, model.graph.output, mod_initializer)
    m_model = onnx.helper.make_model(m_graph, producer_name=model.producer_name, producer_version=model.producer_version, opset_imports=model.opset_import)

    return m_model


def onnx_convert_sum_to_scale(model):
    # list up initializer
    data = listup_initializer(model)

    OL = 'o'
    IL = 'i'

    # list up all Add
    node_add = {}
    cnct = {}  # node connection, key=input_id, val=[output_id]
    for n in model.graph.node:
        if n.op_type == 'Add':
            node_add[n.output[0]] = {OL: n.output[0], IL: n.input}
            if not n.input[0] in cnct.keys():
                cnct[n.input[0]] = []
            cnct[n.input[0]].append(n.output[0])

            if n.input[0] == n.input[1]:
                continue

            if not n.input[1] in cnct.keys():
                cnct[n.input[1]] = []
            cnct[n.input[1]].append(n.output[0])

    fix_node_add = dict(node_add.items())  # unifyed Add
    cnct = {}
    for n in fix_node_add.keys():
        nk = fix_node_add[n][IL]
        if not nk[0] in cnct.keys():
            cnct[nk[0]] = []
        cnct[nk[0]].append(fix_node_add[n][OL])

        if nk[0] == nk[1]:
            continue

        if not nk[1] in cnct.keys():
            cnct[nk[1]] = []
        cnct[nk[1]].append(fix_node_add[n][OL])

    # convert Add to Mul
    node_mul = {}
    node_init = {}
    while(True):
        ref_node_add = dict(fix_node_add.items())
        if len(ref_node_add.keys()) < 1:
            break

        for n in ref_node_add.keys():
            if ref_node_add[n][IL][0] != ref_node_add[n][IL][1]:
                continue
            if ref_node_add[n][IL][0] in ref_node_add.keys():
                continue

            # conbine consecutive Mul nodes as one Mul node
            m = n
            cnt = 1
            while(True):
                del fix_node_add[m]
                if not m in cnct.keys():
                    break
                if len(cnct[m]) > 1:
                    break

                m = cnct[m][0]
                cnt += 1

            mb_val = 2 ** cnt
            mb_id = 'opt' + m
            node_mul[m] = {OL: [m], IL: [ref_node_add[n][IL][0], mb_id]}
            node_init[mb_id] = {OL: [mb_id], IL: [mb_val]}
            break

        if n == list(ref_node_add.keys())[-1]:
            break

    if len(node_add) == len(fix_node_add):
        # no removable Add
        return model

    print("    [i] report : convert {} Add to {} Mul.".format(len(node_add) - len(fix_node_add.keys()), len(node_mul.keys())))

    mod_node = []
    for n in model.graph.node:
        id = n.output[0]

        if n.op_type == 'Add':
            # create unified Add
            if id in fix_node_add.keys():

                x = fix_node_add[id]
                nx = onnx.helper.make_node('Add', inputs=x[IL], outputs=[x[OL]], name=id)
                mod_node.append(nx)
                continue

            # create Mul
            elif id in node_mul.keys():
                x = node_mul[id]
                y = node_init[x[IL][1]]
                scale = numpy.array([y[IL][0]], dtype=numpy.float32)
                data[y[OL][0]] = {'body': scale, 'id': y[OL][0], 'ref': 1}
                nx = onnx.helper.make_node('Mul', inputs=x[IL], outputs=x[OL], name=id)
                mod_node.append(nx)
                continue

            else:
                continue

        mod_node.append(n)

    mod_initializer, mod_input = pack_initializer(data, model.graph.input)

    m_graph = onnx.helper.make_graph(mod_node, model.graph.name, mod_input, model.graph.output, mod_initializer)
    m_model = onnx.helper.make_model(m_graph, producer_name=model.producer_name, producer_version=model.producer_version, opset_imports=model.opset_import)

    return m_model


def onnx_fuse_mul_into_conv(model):
    # list up initializer
    data = listup_initializer(model)

    OL = 'o'
    IL = 'i'
    # list up conv|mul node
    node_conv = {}
    node_mul = {}
    cnct = {}
    for n in model.graph.node:
        for o in n.output:
            for i in n.input:
                cnct.setdefault(i, [])
                cnct[i].append(o)
        if n.op_type == 'Conv':
            node_conv[n.output[0]] = {'type': 'Conv', OL: n.output[0], IL: n.input}
        elif n.op_type == 'ConvTranspose':
            node_conv[n.output[0]] = {'type': 'ConvTranspose', OL: n.output[0], IL: n.input}
        elif n.op_type == 'Mul':
            node_mul[n.output[0]] = {OL: n.output[0], IL: n.input}

    # list up convert candidate and modify weight/bias
    cand = {}
    count_c = 0
    count_ct = 0
    for nm in node_mul.values():
        mul_in = nm[IL]
        if not mul_in[0] in node_conv:
            continue
        if not mul_in[1] in data:
            continue
        if not mul_in[1][:3] == 'opt':
            continue
        mul_scale = data[mul_in[1]]

        if len(cnct[mul_in[0]]) > 1:
            continue
        nc = node_conv[mul_in[0]]
        cv_in = nc[IL]
        if not cv_in[1] in data:
            continue
        cv_weight = data[cv_in[1]]
        scale_data = mul_scale['body']
        weight_data = cv_weight['body']
        ws_shape = [scale_data.size] + [1] * (weight_data.ndim - 1)
        weight_data = weight_data * scale_data.reshape(ws_shape)
        if len(cv_in) >= 3:
            # conv has bias
            if not cv_in[2] in data:
                continue
            cv_bias = data[cv_in[2]]
            bias_data = cv_bias['body'] * scale_data
            cv_bias['body'] = bias_data

        cv_weight['body'] = weight_data
        mul_scale['ref'] -= 1

        cand[nm[OL]] = {'type': 'Mul', OL: nm[OL]}
        cand[nc[OL]] = {'type': nc['type'], IL: nc[IL], OL: nm[OL]}
        if nc['type'] == 'Conv':
            count_c += 1
        else:
            count_ct += 1

    if (count_c + count_ct) == 0:
        # no available fuse node
        return model

    if count_c > 0:
        print("    [i] report : fuse {} Mul to Conv.".format(count_c))
    if count_ct > 0:
        print("    [i] report : fuse {} Mul to ConvTranspose.".format(count_ct))

    mod_node = []
    for n in model.graph.node:
        id = n.output[0]

        if not id in cand:
            mod_node.append(n)
            continue

        if n.op_type == 'Mul':
            # remove node
            continue

        # op_type should be 'Conv' or 'ConvTranspose'
        x = cand[id]
        attr = listup_attribute(n)
        nx = onnx.helper.make_node(x['type'], x[IL], [x[OL]], **attr)
        mod_node.append(nx)

    mod_initializer, mod_input = pack_initializer(data, model.graph.input)

    m_graph = onnx.helper.make_graph(mod_node, model.graph.name, mod_input, model.graph.output, mod_initializer)
    m_model = onnx.helper.make_model(m_graph, producer_name=model.producer_name, producer_version=model.producer_version, opset_imports=model.opset_import)

    return m_model


def onnx_fuse_pad_into_conv(model):
    op_ver = model.opset_import[0].version
    # list up initializer
    data = listup_initializer(model)

    OL = 'o'
    IL = 'i'
    # list up pad+conv node
    node_conv = {}
    node_pad = {}
    cnct = {}
    for n in model.graph.node:
        for o in n.output:
            for i in n.input:
                cnct.setdefault(i, [])
                cnct[i].append(o)
        if n.op_type == 'Conv':
            attr = listup_attribute(n)
            attr.setdefault('pads', None)
            attr.setdefault('auto_pad', b'NOTSET')
            if not (attr['auto_pad'] == b'NOTSET' or attr['auto_pad'] == b'VALID'):
                continue
            node_conv[n.output[0]] = {'type': 'Conv', OL: n.output[0], IL: n.input, 'pads': attr['pads']}
        elif n.op_type == 'Pad':
            attr = listup_attribute(n)
            if op_ver <= 10:
                if not len(n.input) == 1:
                    continue
                attr.setdefault('mode', b'constant')
                attr.setdefault('value', 0.0)
                pads_ = attr['pads']
                value_ = attr['value']
                mode_ = attr['mode']
            elif op_ver > 10:
                attr.setdefault('mode', b'constant')
                mode_ = attr['mode']
                if not len(n.input) in (2, 3):
                    continue
                if not n.input[1] in data:
                    continue
                pads_ = data[n.input[1]]['body']
                if (len(n.input) == 3) and (not n.input[2] == ''):
                    if not n.input[2] in data:
                        continue
                    value_ = data[n.input[2]]['body']
                else:
                    value_ = 0.0
            if not mode_ == b'constant':
                continue
            if not value_ == 0.0:
                continue
            node_pad[n.output[0]] = {OL: n.output[0], IL: n.input, 'mode': mode_, 'pads': pads_, 'value': value_}

    # list up convert candidate
    cand = {}
    count_c = 0
    count_ct = 0
    for nc in node_conv.values():
        conv_in = nc[IL]
        if not conv_in[0] in node_pad:
            continue
        if not len(cnct[conv_in[0]]) == 1:
            continue
        npad = node_pad[conv_in[0]]
        if nc['pads'] == None:
            cv_weight_rank = data[conv_in[1]]['body'].ndim
            nc['pads'] = list([0]) * cv_weight_rank
        cdim = len(nc['pads']) // 2
        pdim = len(npad['pads']) // 2
        if not len(numpy.nonzero(npad['pads'][0:pdim - cdim])[0]) == 0:
            continue
        if not len(numpy.nonzero(npad['pads'][pdim:2 * pdim - cdim])[0]) == 0:
            continue
        mod_pads = [0] * (cdim * 2)
        for i in range(cdim):
            mod_pads[i] = nc['pads'][i] + npad['pads'][(pdim - cdim) + i]
            mod_pads[cdim + i] = nc['pads'][cdim + i] + npad['pads'][pdim * 2 - cdim + i]
        if not len([i for i in mod_pads if i >= 0]) == len(mod_pads):
            continue

        cand[npad[OL]] = {'type': 'Pad', OL: npad[OL]}
        nc[IL][0] = npad[IL][0]
        cand[nc[OL]] = {'type': 'Conv', IL: nc[IL], OL: nc[OL], 'pads': mod_pads}
        if op_ver > 10:
            for i in npad[IL][1::]:
                if i == '':
                    continue
                data[i]['ref'] -= 1
        count_c += 1

    if (count_c + count_ct) == 0:
        # no available fuse node
        return model

    print("    [i] report : fuse {} Pad to Conv.".format(count_c))

    mod_node = []
    for n in model.graph.node:
        id = n.output[0]

        if not id in cand:
            mod_node.append(n)
            continue

        if n.op_type == 'Pad':
            # remove node
            continue

        # op_type should be 'Conv' or 'ConvTranspose'
        x = cand[id]
        attr = listup_attribute(n)
        attr['pads'] = x['pads']
        attr['auto_pad'] = b"NOTSET"
        nx = onnx.helper.make_node(x['type'], x[IL], [x[OL]], **attr)
        mod_node.append(nx)

    mod_initializer, mod_input = pack_initializer(data, model.graph.input)

    m_graph = onnx.helper.make_graph(mod_node, model.graph.name, mod_input, model.graph.output, mod_initializer)
    m_model = onnx.helper.make_model(m_graph, producer_name=model.producer_name, producer_version=model.producer_version, opset_imports=model.opset_import)
    return m_model


def onnx_reduce_squeeze_unsqueeze(model):

    opset_ver = model.opset_import[0].version
    if (opset_ver >= 13):
        # list up initializer
        data = listup_initializer(model)

    # 1st phase - list up candidate of erasable node block
    count_s = 0
    count_u = 0
    axes = {}  # squeeze/unsqueeze axes attribute
    cnct = {}  # node connection, key=input_id, val=[output_id]
    handling = {}  # node handling, key=output_id, val=int(-1:untouchable, 0/1:squeeze, 2/3:unsqueeze, 4/5:siso_eltwise)
    for n in model.graph.node:
        id = n.output[0]
        for o in n.output:
            for i in n.input:
                cnct.setdefault(i, [])
                cnct[i].append(o)
        if n.op_type == 'Squeeze':
            if (opset_ver >= 13):
                if (len(n.input) < 2) or (not n.input[1] in data):
                    handling[id] = -1
                    continue
                axes[id] = data[n.input[1]]['body']
            else:
                att = pick_attribute(n, 'axes')
                if att == None:  # Squeeze node has no attribute 'axes'
                    handling[id] = -1
                    continue
                axes[id] = att.ints
            handling[id] = 0
            count_s += 1
        elif n.op_type == 'Unsqueeze':
            if (opset_ver >= 13):
                if not n.input[1] in data:
                    handling[id] = -1
                    continue
                axes[id] = data[n.input[1]]['body']
            else:
                axes[id] = pick_attribute(n, 'axes').ints
            handling[id] = 2
            count_u += 1
        elif is_1in_eltwise(n.op_type):
            handling[id] = 4  # single in/single out element width
        else:
            handling[id] = -1  # untoutchable

    if (count_s == 0) or (count_u == 0):
        # no removable squeeze/unsqueeze
        return model

    # 2nd phase - effect untouchable to donwstream
    for i in model.graph.input:
        id = i.name
        # skip unused input
        if not id in cnct:
            continue
        for nxt in cnct[id]:
            if (not nxt in handling):
                continue
            if (handling[nxt] < 4):
                continue
            handling[nxt] = -1
    for n in model.graph.node:
        id = n.output[0]
        if (handling[id] >= 0):
            continue
        for o in n.output:
            if (not o in cnct):
                continue
            for nxt in cnct[o]:
                if (not nxt in handling):
                    continue
                if (handling[nxt] < 4):
                    continue
                handling[nxt] = -1

    # 3rd phase - remove needless squeeze/unsqueeze
    count = 0
    mod_node = []
    replace = {}
    oname = []
    for o in model.graph.output:
        oname.append(o.name)
    for n in model.graph.node:
        id = n.output[0]

        # update node input : for all node
        for i in range(len(n.input)):
            k = n.input[i]
            if (k in replace.keys()):
                n.input[i] = replace[k]

        if (id in oname):
            mod_node.append(n)
            continue

        if (handling[id] == 0) or (handling[id] == 2):
            cand_o = handling[id]
            axes_o = axes[id]
            wsrc = cnct[id]
            wdst = {}
            while (len(wsrc) > 0):
                k = wsrc[0]
                if k in oname:
                    break  # reach end of graph
                type = handling[k]
                if (type < 0):
                    break  # untouchable
                if (cand_o == 0) and (0 <= type <= 1):
                    break  # re-squeeze
                if (cand_o == 2) and (2 <= type <= 3):
                    break  # re-unsqueeze
                if (type < 4) and (axes_o != axes[k]):
                    break  # unmatch axes
                del wsrc[0]
                if ((type % 2) == 0):
                    type += 1
                wdst[k] = type
                if (type > 3):  # siso eltwise
                    wsrc.extend(cnct[k])

            if len(wsrc) > 0:
                # unremovable
                mod_node.append(n)
                continue

            # removable
            count += 1
            replace[id] = n.input[0]
            for k in wdst:
                handling[k] = wdst[k]
            if (opset_ver >= 13):
                data[n.input[1]]['ref'] -= 1

        elif (handling[id] == 1) or (handling[id] == 3):
            # removing closer
            replace[id] = n.input[0]

        else:
            # other
            mod_node.append(n)

    if count == 0:
        # no removable squeeze/unsqueeze
        return model

    print("    [i] report : remove {} squeeze/unsqueeze node pair.".format(count))

    # generate modified model
    m_graph = onnx.helper.make_graph(mod_node, model.graph.name, model.graph.input, model.graph.output, model.graph.initializer)
    m_model = onnx.helper.make_model(m_graph, producer_name=model.producer_name, producer_version=model.producer_version, opset_imports=model.opset_import)

    return m_model


def onnx_replace_to_reducel2(model):
    # list up initializer
    data = listup_initializer(model)

    OL = 'o'
    IL = 'i'
    # list up replace node candidate
    cand = {}  # candidate
    cnct = {}  # conection
    for n in model.graph.node:
        for o in n.output:
            for i in n.input:
                cnct.setdefault(i, [])
                cnct[i].append(o)
        nmo = n.output[0]  # node main output
        if (n.op_type == 'Mul'):
            if (len(n.input) != 2):
                continue
            if (n.input[0] != n.input[1]):
                continue
            if (n.input[0] in cand):
                continue
            cand[nmo] = {'type': 'Mul', OL: nmo, IL: n.input}
        elif (n.op_type == 'Pow'):
            if not (n.input[1] in data):
                continue
            pow = data[n.input[1]]['body']
            if pow.size != 1:
                continue
            if pow.ndim == 0:
                pow = numpy.asarray([pow], dtype=pow.dtype)
            if abs(pow[0] - 2.0) > 1e-7:
                continue
            cand[nmo] = {'type': 'Pow', OL: nmo, IL: n.input}
        elif (n.op_type == 'ReduceSum'):
            if not (n.input[0] in cand):
                continue
            ut = cand[n.input[0]]['type']
            if not ((ut == 'Mul') or (ut == 'Pow')):
                continue
            cand[nmo] = {'type': 'ReduceSum', OL: nmo, IL: n.input, 'attr': listup_attribute(n)}
        elif (n.op_type == 'Max'):
            if not (n.input[1] in data):
                continue
            if not (n.input[0] in cand):
                continue
            ut = cand[n.input[0]]['type']
            if (ut != 'ReduceSum'):
                continue
            cand[nmo] = {'type': 'Max', OL: nmo, IL: n.input}
        elif (n.op_type == 'Add'):
            if not (n.input[1] in data):
                continue
            if not (n.input[0] in cand):
                continue
            epsilon = data[n.input[1]]['body']
            if epsilon > 1.0e-8:
                continue
            ut = cand[n.input[0]]['type']
            if not (ut == 'ReduceSum'):
                continue
            cand[nmo] = {'type': 'Add', OL: nmo, IL: n.input}
        elif (n.op_type == 'Sqrt'):
            if not (n.input[0] in cand):
                continue
            ut = cand[n.input[0]]['type']
            if not ((ut == 'ReduceSum') or (ut == 'Max') or (ut == 'Add')):
                continue
            cand[nmo] = {'type': 'Sqrt', OL: nmo, IL: n.input}

    # verify candidate connection
    count = 0
    target = {}
    for n in cand.values():
        if (n['type'] != 'Sqrt'):
            continue
        wi = n
        wl = []
        attr = None
        node_max = None
        epsilon_node = None
        while (wi[IL][0] in cand):
            if (len(cnct[wi[IL][0]]) > 1):
                break
            if (wi['type'] == 'ReduceSum'):
                attr = wi['attr']
            if (wi['type'] == 'Max'):
                node_max = wi
            if (wi['type'] == 'Add'):
                epsilon_node = wi
            wl.append(wi)
            wi = cand[wi[IL][0]]
        if not ((wi['type'] == 'Mul') or (wi['type'] == 'Pow')):
            continue
        wl.append(wi)
        wl[0]['attr'] = attr
        wl[0][IL] = [wi[IL][0]]
        if node_max:
            node_max[OL], wl[0][OL] = wl[0][OL], node_max[OL]
            node_max[IL][0] = wl[0][OL]
        if epsilon_node:
            epsilon_node[OL], wl[0][OL] = wl[0][OL], epsilon_node[OL]
            epsilon_node[IL][0] = wl[0][OL]
        wl[0]['type'] = 'ReduceL2'
        count += 1
        for wi in wl:
            target[wi[OL]] = wi

    if (count == 0):
        # no available replace node
        return model

    print("    [i] report : replace {} 'Mul+ReduceSum+Sqrt' to ReduceL2.".format(count))

    mod_node = []
    for n in model.graph.node:
        id = n.output[0]

        if not (id in target):
            mod_node.append(n)
            continue

        wi = target[id]
        if (wi['type'] == 'ReduceL2'):
            nx = onnx.helper.make_node('ReduceL2', wi[IL], [wi[OL]], **wi['attr'])
            mod_node.append(nx)
        elif (wi['type'] == 'Max'):
            val = data[wi[IL][1]]
            if val['ref'] == 1:
                val['body'] = numpy.sqrt(val['body'])
            else:
                val['ref'] -= 1
                mod_id = 'opt.' + wi[IL][1]
                wi[IL][1] = mod_id
                data[mod_id] = {'ref': 1, 'id': mod_id, 'body': numpy.sqrt(val['body'])}
            nx = onnx.helper.make_node('Max', wi[IL], [wi[OL]])
            mod_node.append(nx)
        elif (wi['type'] == 'Pow'):
            val = data[wi[IL][1]]
            val['ref'] -= 1
        elif (wi['type'] == 'Add'):
            val = data[wi[IL][1]]
            if val['ref'] == 1:
                val['body'] = numpy.sqrt(val['body'])
            else:
                val['ref'] -= 1
                mod_id = 'opt.' + wi[IL][1]
                wi[IL][1] = mod_id
                data[mod_id] = {'ref': 1, 'id': mod_id, 'body': numpy.sqrt(val['body'])}
            nx = onnx.helper.make_node('Add', wi[IL], [wi[OL]])
            mod_node.append(nx)

    mod_initializer, mod_input = pack_initializer(data, model.graph.input)
    m_graph = onnx.helper.make_graph(mod_node, model.graph.name, mod_input, model.graph.output, mod_initializer)
    m_model = onnx.helper.make_model(m_graph, producer_name=model.producer_name, producer_version=model.producer_version, opset_imports=model.opset_import)

    return m_model


def onnx_replace_to_lpnorm(model):
    # list up initializer
    data = listup_initializer(model)
    # shape inference
    info = ShapeAndTypeInfo(model)

    OL = 'o'
    IL = 'i'
    # list up replace node candidate
    cand = {}  # candidate
    cnct = {}  # conection
    for n in model.graph.node:
        for o in n.output:
            for i in n.input:
                cnct.setdefault(i, [])
                cnct[i].append(o)
        nmo = n.output[0]  # node main output
        if (n.op_type == 'ReduceL2'):
            if n.input[0] in data:
                continue
            attr = listup_attribute(n)
            axes = attr['axes']
            shape = info.get_shape(n.input[0])
            # pass if
            #   input tensor rank is unknown
            # or
            #   input tensor shape is unclear and axes select multiple axis
            if (shape == None) or ((0 in shape) and len(axes) > 1):
                continue
            # change negative axes to positive
            adjust_axes = numpy.array([(len(shape) + s) if s < 0 else s for s in axes])
            adjust_axes.sort()
            tmp = adjust_axes - adjust_axes.min()
            if len(tmp) - 1 != tmp[-1]:
                continue
            attr['axes'] = adjust_axes
            cand[nmo] = {'type': 'ReduceL2', OL: nmo, IL: n.input, 'attr': attr}
        elif (n.op_type == 'Add'):
            if not (n.input[1] in data):
                continue
            if not (n.input[0] in cand):
                continue
            epsilon = data[n.input[1]]['body']
            if epsilon > 1.0e-4:
                continue
            ut = cand[n.input[0]]['type']
            if not (ut == 'ReduceL2'):
                continue
            cand[nmo] = {'type': 'Add', OL: nmo, IL: n.input}
        elif (n.op_type == 'Clip'):
            if not (n.input[0] in cand):
                continue
            ut = cand[n.input[0]]['type']
            if not (ut == 'ReduceL2'):
                continue
            cand[nmo] = {'type': 'Clip', OL: nmo, IL: n.input}
        elif (n.op_type == 'Div'):
            if not (n.input[1] in cand):
                continue
            ut = cand[n.input[1]]['type']
            if not ((ut == 'Add') or (ut == 'Expand') or (ut == 'Clip')):
                continue
            cand[nmo] = {'type': 'Div', OL: nmo, IL: n.input}
        elif (n.op_type == 'Reshape'):
            if not (n.input[1] in data):
                continue
            if not (n.input[0] in cand):
                continue
            ut = cand[n.input[0]]['type']
            if not (ut == 'Div'):
                continue
            cand[nmo] = {'type': 'Reshape', OL: nmo, IL: n.input}
        elif (n.op_type == 'Max'):
            if not (n.input[0] in cand):
                continue
            if not (n.input[1] in data):
                continue
            ut = cand[n.input[0]]['type']
            if not (ut == 'ReduceL2'):
                continue
            cand[nmo] = {'type': 'Max', OL: nmo, IL: n.input}
        elif (n.op_type == 'Reciprocal'):
            if not (n.input[0] in cand):
                continue
            ut = cand[n.input[0]]['type']
            if not (ut == 'Max'):
                continue
            cand[nmo] = {'type': 'Reciprocal', OL: nmo, IL: n.input}
        elif (n.op_type == 'Mul'):
            if (len(n.input) != 2):
                continue
            if not (n.input[0] != n.input[1]):
                continue
            if not (n.input[1] in cand):
                continue
            ut = cand[n.input[1]]['type']
            if not ((ut == 'Reshape') or (ut == 'Reciprocal') or (ut == 'Div')):
                continue
            cand[nmo] = {'type': 'Mul', OL: nmo, IL: n.input}
    # verify candidate connection
    count = 0
    target = {}
    for n in cand.values():
        if not (n['type'] == 'Mul' or n['type'] == 'Div'):
            continue
        wi = n
        wl = []
        attr = None
        tar_ipt = None
        valid = False
        ipt = wi[IL][1]
        while (ipt in cand):
            if wi['type'] == 'Mul':
                if (len(cnct[wi[IL][1]]) > 1):
                    break
                if tar_ipt == None:
                    tar_ipt = wi[IL][0]
            elif wi['type'] == 'Div':
                if (len(cnct[wi[IL][1]]) > 1):
                    break
                if tar_ipt == None:
                    tar_ipt = wi[IL][0]
            else:
                if (len(cnct[wi[IL][0]]) > 1):
                    break
            wl.append(wi)
            if wi['type'] == 'Div' or wi['type'] == 'Mul':
                wi = cand[wi[IL][1]]
            else:
                wi = cand[wi[IL][0]]
            if wi['type'] == 'Div' or wi['type'] == 'Mul':
                ipt = wi[IL][1]
            else:
                ipt = wi[IL][0]
        if wi['type'] == 'ReduceL2':
            if (tar_ipt != wi[IL][0]):
                continue
            shape = numpy.array(info.get_shape(tar_ipt))
            valid = True
            attr = wi['attr']
        if not valid:
            continue
        wl.append(wi)
        wl[0]['attr'] = attr
        wl[0][IL] = [wi[IL][0]]
        wl[0]['type'] = 'LpNormalization'
        count += 1
        for wi in wl:
            target[wi[OL]] = wi
    if (count == 0):
        # no available replace node
        return model

    print("    [i] report : replace {} 'ReduceL2+eltwise' to LpNormalization".format(count))

    mod_node = []
    for n in model.graph.node:
        id = n.output[0]

        if not (id in target):
            mod_node.append(n)
            continue

        wi = target[id]
        if (wi['type'] == 'LpNormalization'):
            if len(wi['attr']['axes']) > 1:
                id = 'opt.reshape1.' + wi[IL][0]
                reshape1_output = 'opt.reshape1.output.' + wi[IL][0]
                shape = numpy.array(info.get_shape(wi[IL][0]), dtype=numpy.int64)
                axes = wi['attr']['axes']
                reshape1_shape = []
                for i in range(len(shape)):
                    if i in axes:
                        if -1 in reshape1_shape:
                            continue
                        reshape1_shape.append(-1)
                    else:
                        reshape1_shape.append(shape[i])
                data[id] = {'ref': 1, 'id': id, 'body': numpy.array(reshape1_shape, dtype=numpy.int64)}
                il = wi[IL]
                il.append(id)
                ol = []
                ol.append(reshape1_output)
                nx = onnx.helper.make_node('Reshape', il, ol)
                mod_node.append(nx)

                id = 'opt.lpnorm.' + wi[IL][0]
                lpnorm_output = 'opt.lpnorm.output.' + wi[IL][0]
                il = []
                il.append(reshape1_output)
                ol = []
                ol.append(lpnorm_output)
                attr = {'axis': min(wi['attr']['axes']), 'p': 2}
                nx = onnx.helper.make_node('LpNormalization', il, ol, **attr)
                mod_node.append(nx)

                id = 'opt.reshape2.' + wi[IL][0]
                data[id] = {'ref': 1, 'id': id, 'body': shape}
                il = [lpnorm_output, id]
                nx = onnx.helper.make_node('Reshape', il, [wi[OL]])
                mod_node.append(nx)
            else:
                attr = {'axis': wi['attr']['axes'][0], 'p': 2}
                nx = onnx.helper.make_node('LpNormalization', wi[IL], [wi[OL]], **attr)
                mod_node.append(nx)
        elif (wi['type'] == 'Add') or (wi['type'] == 'Reshape') or (wi['type'] == 'Max'):
            val = data[wi[IL][1]]
            val['ref'] -= 1
        elif (wi['type'] == 'Div'):
            val = data[wi[IL][0]]
            val['ref'] -= 1

    mod_initializer, mod_input = pack_initializer(data, model.graph.input)
    m_graph = onnx.helper.make_graph(mod_node, model.graph.name, mod_input, model.graph.output, mod_initializer)
    m_model = onnx.helper.make_model(m_graph, producer_name=model.producer_name, producer_version=model.producer_version, opset_imports=model.opset_import)

    return m_model


def onnx_replace_to_size(model):
    OL = 'o'
    IL = 'i'
    # list up replace node candidate
    cand = {}  # candidate
    cnct = {}  # conection
    for n in model.graph.node:
        for o in n.output:
            for i in n.input:
                cnct.setdefault(i, [])
                cnct[i].append(o)
        nmo = n.output[0]  # node main output
        if (n.op_type == 'Shape'):
            cand[nmo] = {'type': 'Shape', OL: nmo, IL: n.input}
        elif (n.op_type == 'ReduceProd'):
            if not (n.input[0] in cand):
                continue
            ut = cand[n.input[0]]['type']
            if not (ut == 'Shape'):
                continue
            cand[nmo] = {'type': 'ReduceProd', OL: nmo, IL: n.input}
    # verify candidate connection
    count = 0
    target = {}
    for n in cand.values():
        if (n['type'] != 'ReduceProd'):
            continue
        nm = n
        nr = cand[n[IL][0]]
        if (nr['type'] != 'Shape'):
            continue
        if (len(cnct[n[IL][0]]) > 1):
            continue
        nm['type'] = 'Size'
        nm[IL][0] = nr[IL][0]
        target[nr[OL]] = nr
        target[nm[OL]] = nm
        count += 1
    if (count == 0):
        # no available replace node
        return model

    print("    [i] report : replace {} 'Shape+ReduceProt' to Size.".format(count))
    mod_node = []
    for n in model.graph.node:
        id = n.output[0]

        if not (id in target):
            mod_node.append(n)
            continue

        wi = target[id]
        if (wi['type'] == 'Size'):
            nx = onnx.helper.make_node('Size', wi[IL], [wi[OL]])
            mod_node.append(nx)

    m_graph = onnx.helper.make_graph(mod_node, model.graph.name, model.graph.input, model.graph.output, model.graph.initializer)
    m_model = onnx.helper.make_model(m_graph, producer_name=model.producer_name, producer_version=model.producer_version, opset_imports=model.opset_import)

    return m_model


def onnx_replace_randomlike(model):
    # shape inference
    info = ShapeAndTypeInfo(model)

    OL = 'o'
    IL = 'i'
    # list up replace node candidate
    cand = {}  # candidate
    for n in model.graph.node:
        nmo = n.output[0]  # node main output
        if (n.op_type == 'RandomNormalLike'):
            cand[nmo] = {'type': n.op_type, OL: nmo, IL: n.input, 'attr': listup_attribute(n)}
        elif (n.op_type == 'RandomUniformLike'):
            cand[nmo] = {'type': n.op_type, OL: nmo, IL: n.input, 'attr': listup_attribute(n)}

    count = 0
    target = {}
    for n in cand.values():
        if not info.is_inferred(n[IL][0]):
            continue
        shape = info.get_shape(n[IL][0])
        n['attr']['shape'] = shape
        target[n[OL]] = n
        count += 1

    if count == 0:
        return model

    print("    [i] report : replace {} 'Random*Like' to Random*".format(count))

    mod_node = []
    for n in model.graph.node:
        id = n.output[0]

        if not id in target:
            mod_node.append(n)
            continue

        wi = target[id]
        if ('Random' in wi['type']):
            if 'Uniform' in wi['type']:
                nx = onnx.helper.make_node('RandomUniform', [], [wi[OL]], **wi['attr'])
            else:
                nx = onnx.helper.make_node('RandomNormal', [], [wi[OL]], **wi['attr'])
            mod_node.append(nx)

    m_graph = onnx.helper.make_graph(mod_node, model.graph.name, model.graph.input, model.graph.output, model.graph.initializer)
    m_model = onnx.helper.make_model(m_graph, producer_name=model.producer_name, producer_version=model.producer_version, opset_imports=model.opset_import)

    return m_model


def onnx_reduce_needless_arithmetic(model):
    OL = 'o'
    IL = 'i'
    TP = 'type'
    ID = 'id'
    IIN = 'iin'  # node input (initializer)
    DIN = 'din'  # node input (not an initializer)

    # list up candidate node
    node_cand = {}
    cnct = {}
    cand_list = ['Add', 'Sub', 'Mul', 'Div', 'Pow']
    only_2nd = ['Sub', 'Div', 'Pow']  # requires 2nd input constant
    need_zero = ['Add', 'Sub']
    need_one = ['Mul', 'Div', 'Pow']
    for n in model.graph.node:
        for o in n.output:
            for i in n.input:
                cnct.setdefault(i, [])
                cnct[i].append(o)
        if not n.op_type in cand_list:
            continue
        node_cand[n.output[0]] = {TP: n.op_type, OL: n.output[0], IL: n.input}

    # list up initializer
    data = listup_initializer(model)
    initializers, inputs = pack_initializer(data, model.graph.input)

    # exclude unsupported nodes from candidates
    node_exclude = []
    for nck in node_cand.keys():
        nc = node_cand[nck]
        ninit = []
        nnode = []
        for nci in nc[IL]:
            if nci in data.keys():
                ninit.append(nci)
            else:
                nnode.append(nci)
        if (len(ninit) == 1) and (len(nnode) == 1):
            nc[DIN] = nnode[0]
            nc[IIN] = ninit[0]
        else:
            node_exclude.append(nck)

    for nck in node_cand.keys():
        if nck in node_exclude:
            continue
        nc = node_cand[nck]
        if (nc[TP] in only_2nd) and (nc[IL][1] != nc[IIN]):
            node_exclude.append(nck)
            continue
        coef = data[nc[IIN]]['body']
        if (nc[TP] in need_one) and (coef != 1.).any():
            node_exclude.append(nck)
            continue
        if (nc[TP] in need_zero) and (coef != 0.0).any():
            node_exclude.append(nck)
            continue

    for nck in node_exclude:
        del node_cand[nck]

    replace = {}
    rem = []
    # inference shape of original model
    shapes = ShapeAndTypeInfo(model)
    for nck in node_cand.keys():
        # get inderred shape of each input
        nc = node_cand[nck]

        d_shape = shapes.get_shape(nc[DIN]) if shapes.is_inferred(nc[DIN]) else None
        c_shape = shapes.get_shape(nc[IIN])

        if check_data_size_change_on_eltwise(d_shape, c_shape):
            continue

        tag = nc[DIN]
        replace[nck] = replace[tag] if tag in replace else tag
        rem.append(nc[IIN])
        data[nc[IIN]]['ref'] -= 1

    count = len(replace.keys())
    if not count:
        return model

    mod_node = []
    for n in model.graph.node:
        id = n.output[0]
        if id in replace.keys():
            continue
        if id in rem:
            continue

        # update node input : for all node
        for i in range(len(n.input)):
            k = n.input[i]
            if k in replace.keys():
                n.input[i] = replace[k]

        mod_node.append(n)

    # adjust blob name related graph output
    adjust_graph_output(mod_node, replace, model)

    mod_initializer, mod_input = pack_initializer(data, model.graph.input)
    m_graph = onnx.helper.make_graph(mod_node, model.graph.name, mod_input, model.graph.output, mod_initializer)
    m_model = onnx.helper.make_model(m_graph, producer_name=model.producer_name, producer_version=model.producer_version, opset_imports=model.opset_import)
    print("    [i] report : reduce {} arithmetic node.".format(count))

    return m_model


def onnx_convert_node_to_const(model):
    MAX_ITR = 100
    OL = 'o'
    IL = 'i'
    TP = 'type'
    ID = 'id'

    # list up candidate node
    node_cand = {}
    cnct = {}
    cand_list = ['Equal', 'Where', 'Concat', 'Gather', 'Slice', 'Cast', 'Add', 'Sub', 'Mul', 'Div', 'Floor', 'Ceil', 'Sqrt', 'Shape', 'ConstantOfShape', 'NonZero', 'Transpose', 'Squeeze', 'Unsqueeze', 'Expand', 'Reshape', 'ReduceMean', 'ReduceProd', 'Range', 'Tile', 'Gemm', 'ScatterND', 'Not']
    for n in model.graph.node:
        for o in n.output:
            for i in n.input:
                cnct.setdefault(i, [])
                cnct[i].append(o)
        if not n.op_type in cand_list:
            continue

        if n.op_type in ('Unsqueeze', 'Squeeze'):  # list of int
            dat = pick_attribute(n, 'axes')
            if dat == None:
                continue
            else:
                dat = dat.ints
        elif n.op_type in ('Cast'):  # int
            dat = pick_attribute(n, 'to').i
        elif n.op_type in ('Concat'):  # int
            dat = pick_attribute(n, 'axis').i
        elif n.op_type == 'Gather':  # int (optional)
            dat = pick_attribute(n, 'axis')
            dat = dat.i if dat != None else 0
        elif n.op_type == 'Transpose':  # list of int (optional)
            dat = pick_attribute(n, 'perm')
            dat = dat.ints if dat != None else 'rev'
        elif n.op_type == 'ConstantOfShape':  # tensor:int64
            dat_ = pick_attribute(n, 'value')
            dat = onnx.numpy_helper.to_array(dat_.t) if dat_ != None else numpy.array([0], dtype=numpy.int64)
        elif (n.op_type == 'ReduceMean') or (n.op_type == 'ReduceProd'):
            ax = pick_attribute(n, 'axes')
            kd = pick_attribute(n, 'keepdims')
            ax = None if (ax == None) else tuple(ax.ints)
            kd = True if (kd == None) else (kd.i != 0)
            dat = {'axes': ax, 'keepdims': kd}
        elif n.op_type == 'Gemm':
            al = pick_attribute(n, 'alpha')
            bt = pick_attribute(n, 'beta')
            tA = pick_attribute(n, 'transA')
            tB = pick_attribute(n, 'transB')
            al = al.f if al != None else 1.0
            bt = bt.f if al != None else 1.0
            tA = tA.t if tA != None else 0
            tB = tB.i if tB != None else 0
            dat = {'alpha': al, 'beta': bt, 'transA': tA, 'transB': tB}
        elif (n.op_type == 'ScatterND'):
            red = pick_attribute(n, 'reduction')
            dat = {'reduction': red}
        else:
            dat = None
        node_cand[n.output[0]] = {TP: n.op_type, OL: n.output[0], IL: n.input, 'attribute': dat}

    if len(node_cand) == 0:
        return model

    count = 0
    rem_count = 0

    # list up initializer
    data = listup_initializer(model)
    mod_initializer, mod_input = pack_initializer(data, model.graph.input)

    # setup input blobs shape dict
    shapes = ShapeAndTypeInfo(model)

    for itr_ in range(MAX_ITR):
        cand = {}
        replace = {}
        rem = {}

        # inference blob shape dict
        shapes.refresh_inference(model)

        for k in node_cand.keys():
            nc = node_cand[k]
            if not k in cnct.keys():
                continue
            nc['ref'] = len(cnct[k])
            ci = []
            if (nc[TP] != 'Shape'):
                const_flag = False
                for ni in nc[IL]:
                    consd = data.get(ni, None)
                    consc = cand.get(ni, None)
                    if (consd == None) and (consc == None):
                        const_flag = True
                    else:
                        ci.append(consd) if consd != None else ci.append(consc)
                if const_flag:
                    continue

            if nc[TP] == 'Shape':
                ni = nc[IL][0]
                if not shapes.is_inferred(ni):
                    continue
                inferred_shape = shapes.get_shape(ni)
                rep_body = numpy.asarray(inferred_shape, dtype=numpy.int64)
                if ni in data:
                    ci.append(data[ni])
                elif ni in cand:
                    ci.append(cand[ni])

            elif nc[TP] == 'Equal':
                if len(ci) < 2:
                    continue
                rep_body = numpy.equal(ci[0]['body'], ci[1]['body'])

            elif nc[TP] == 'Where':
                if len(ci) < 3:
                    continue
                rep_body = numpy.where(ci[0]['body'], ci[1]['body'], ci[2]['body'])

            elif nc[TP] == 'Gather':
                if len(ci) < 2:
                    continue
                dat = ci[0]['body']
                idc = ci[1]['body']
                axes = nc['attribute']
                rep_body = numpy.take(dat, idc, axis=axes)

            elif nc[TP] == 'Concat':
                if len(ci) < 1:
                    continue
                rb = [ci[i]['body'] for i in range(len(ci))]
                rep_body = numpy.concatenate(rb, axis=nc['attribute'])

            elif nc[TP] == 'Slice':
                if len(ci) < 3:
                    continue
                dat_ = ci[0]['body']
                st = ci[1]['body']
                en = ci[2]['body']
                axes = numpy.arange(len(st), dtype=numpy.int64) if len(ci) < 4 else ci[3]['body']
                steps = numpy.ones((len(st),), dtype=numpy.int64) if len(ci) < 5 else ci[4]['body']
                dat_dim = dat_.ndim
                slc = [slice(None)] * dat_.ndim
                for ax, s, e, t in zip(axes, st, en, steps):
                    slc[ax] = slice(s, e, t)
                rep_body = dat_[tuple(slc)]

            elif nc[TP] == 'Cast':
                if len(ci) != 1:
                    continue
                dt = ci[0]['body'].dtype
                if nc['attribute'] == 1:
                    dt = numpy.float32
                elif nc['attribute'] == 7:
                    dt = numpy.int64
                rep_body = numpy.asarray(ci[0]['body'], dtype=dt)

            elif nc[TP] == 'Add':
                if len(ci) != 2:
                    continue
                if ci[0]['body'].dtype != ci[1]['body'].dtype:
                    continue
                rep_body = numpy.add(ci[0]['body'], ci[1]['body'])

            elif nc[TP] == 'Sub':
                if len(ci) != 2:
                    continue
                if ci[0]['body'].dtype != ci[1]['body'].dtype:
                    continue
                rep_body = numpy.subtract(ci[0]['body'], ci[1]['body'])

            elif nc[TP] == 'Mul':
                if len(ci) != 2:
                    continue
                if ci[0]['body'].dtype != ci[1]['body'].dtype:
                    continue
                rep_body = numpy.multiply(ci[0]['body'], ci[1]['body'])

            elif nc[TP] == 'Div':
                if len(ci) != 2:
                    continue
                x1 = ci[0]['body']
                x2 = ci[1]['body']
                if len(numpy.flatnonzero(x2)) < len(x2.flatten()):
                    continue  # detect zero from denominator
                rep_body = numpy.divide(x1, x2)

            elif nc[TP] == 'Floor':
                if len(ci) != 1:
                    continue
                if ci[0]['body'].dtype != numpy.float32:
                    continue
                rep_body = numpy.floor(ci[0]['body'])

            elif nc[TP] == 'Ceil':
                if len(ci) != 1:
                    continue
                if ci[0]['body'].dtype != numpy.float32:
                    continue
                rep_body = numpy.ceil(ci[0]['body'])

            elif nc[TP] == 'Sqrt':
                if len(ci) != 1:
                    continue
                if ci[0]['body'].dtype != numpy.float32:
                    continue
                rep_body = numpy.sqrt(ci[0]['body'])

            elif nc[TP] == 'ConstantOfShape':
                if len(ci) != 1:
                    continue
                if nc['attribute'] == None:
                    continue
                rep_body = nc['attribute'] * numpy.ones(ci[0]['body'], dtype=nc['attribute'].dtype)
                if rep_body.size >= (64 * 1024):
                    # too large to convert
                    if not shapes.is_inferred(nc[OL]):
                        shapes.register(nc[OL], rep_body)
                    continue

            elif nc[TP] == 'NonZero':
                if len(ci) != 1:
                    continue
                rep_body = numpy.array(numpy.nonzero(ci[0]['body']))

            elif nc[TP] == 'Transpose':
                dat = ci[0]['body']
                perm = nc['attribute']
                if (ci[0]['ref'] > 1) and (dat.size >= (512 * 1024)):
                    continue
                if len(ci) != 1:
                    continue
                if len(perm) != dat.ndim:
                    continue
                if nc['attribute'] == 'rev':
                    perm = [i for i in range(dat.ndim, -1)]
                rep_body = numpy.transpose(dat, perm)

            elif nc[TP] == 'Squeeze':
                if len(ci) != 1:
                    continue
                rb = ci[0]['body']
                ax = tuple(nc['attribute'])
                if rb.ndim < max(ax):
                    continue
                brf = False
                for i in ax:
                    if rb.shape[i] != 1:
                        brf = True
                if brf:
                    continue
                rep_body = numpy.squeeze(ci[0]['body'], ax)

            elif nc[TP] == 'Unsqueeze':
                if len(ci) != 1:
                    continue
                dat = ci[0]['body']
                axes = nc['attribute']
                if len(axes) > 1:
                    continue
                if axes[0] > dat.ndim:
                    continue
                rep_body = numpy.expand_dims(dat, axes[0])

            elif nc[TP] == 'Expand':
                if len(ci) != 2:
                    continue
                rep_body = ci[0]['body'] * numpy.ones(ci[1]['body'], dtype=ci[0]['body'].dtype)
                if rep_body.size >= (64 * 1024):
                    # too large to convert
                    if not shapes.is_inferred(nc[OL]):
                        shapes.register(nc[OL], rep_body)
                    continue

            elif nc[TP] == 'Reshape':
                if len(ci) != 2:
                    continue
                dat = ci[0]['body']
                shape = ci[1]['body']
                rep_body = numpy.reshape(dat, shape)

            elif nc[TP] == 'ReduceMean':
                if len(ci) != 1:
                    continue
                dat = ci[0]['body']
                axes = nc['attribute']['axes']
                keepdims = nc['attribute']['keepdims']
                rep_body = numpy.mean(dat, axis=axes, keepdims=keepdims)

            elif nc[TP] == 'ReduceProd':
                if len(ci) != 1:
                    continue
                dat = ci[0]['body']
                axes = nc['attribute']['axes']
                keepdims = nc['attribute']['keepdims']
                rep_body = numpy.prod(dat, axis=axes, keepdims=keepdims)

            elif nc[TP] == 'Range':
                if len(ci) != 3:
                    continue
                start = ci[0]['body']
                limit = ci[1]['body']
                delta = ci[2]['body']
                rep_body = numpy.arange(start, limit, delta)

            elif nc[TP] == 'Tile':
                if len(ci) != 2:
                    continue
                ind = ci[0]['body']
                rep = ci[1]['body']
                rep_body = numpy.tile(ind, rep)

            elif nc[TP] == 'Gemm':
                if len(ci) < 2:
                    continue
                alpha = nc['attribute']['alpha']
                beta = nc['attribute']['beta']
                dataa = ci[0]['body'].transpose() if nc['attribute']['transA'] else ci[0]['body']
                datab = ci[1]['body'].transpose() if nc['attribute']['transB'] else ci[1]['body']
                datac = ci[2]['body'] if len(ci) > 2 else 0
                rep_body = numpy.matmul(alpha * dataa, beta * datab) + datac

            elif nc[TP] == 'ScatterND':
                if len(ci) != 3:
                    continue
                ipt = ci[0]['body']
                ind = ci[1]['body']
                upd = ci[2]['body']
                red = nc['attribute']['reduction']
                f = None
                if red is not None:
                    if red == "add":
                        f = numpy.add
                    elif red == "mul":
                        f = numpy.multiply
                    # for opset18
                    elif red == "max":
                        f = numpy.maximum
                    elif red == "min":
                        f = numpy.minimum
                val = numpy.copy(ipt)
                upd_ind = ind.shape[:-1]
                for idx in numpy.ndindex(upd_ind):
                    val[tuple(ind[idx])] = upd[idx] if f is None else f(val[tuple(ind[idx])], upd[idx])
                rep_body = val

            elif nc[TP] == 'Not':
                if len(ci) != 1:
                    continue
                x = ci[0]['body']
                rep_body = numpy.logical_not(x)

            for ni in nc[IL]:
                ni = replace[ni] if ni in replace else ni
                cnct[ni].remove(nc[OL])
                if len(cnct[ni]) == 0:
                    del cnct[ni]

            tag = nc[OL]
            opt_tag = 'opt_' + tag
            cand[tag] = {TP: 'Init', IL: [], ID: opt_tag, 'body': rep_body, 'ref': len(cnct[tag])}
            replace[tag] = opt_tag
            shapes.register(tag, rep_body)
            if tag in cnct:
                cnct[opt_tag] = cnct.pop(tag)

            for cons in ci:
                cons['ref'] -= 1
                if cons['ref'] == 0:
                    if 'type' in cons.keys():
                        keys = [k for k, v in cand.items() if v == cons]
                        del cand[keys[0]]
                        rem[keys[0]] = cons

        rem_count += len(rem)
        count += len(cand)
        if len(cand) == 0:
            break

        if itr_ == 0:
            print("    now analysing the model using shape inference :")
            print("      ", end='')

        for kk in rem.keys():
            del node_cand[kk]

        for kk in cand.keys():
            del node_cand[kk]

        mod_node = []
        for n in model.graph.node:
            id = n.output[0]
            if id in rem:
                continue

            # update node input : for all node
            for i in range(len(n.input)):
                k = n.input[i]
                if k in replace.keys():
                    n.input[i] = replace[k]
                if n.output[0] in node_cand:
                    node_cand[n.output[0]][IL] = n.input
            if not id in cand:
                mod_node.append(n)
                continue

            x = cand[id]
            for i in range(len(x[IL])):
                k = x[IL][i]
                if k in replace.keys():
                    x[IL][i] = replace[k]

            init_value = numpy.array(x['body'], dtype=x['body'].dtype)
            data[x[ID]] = {'body': init_value, 'id': x[ID], 'ref': x['ref']}

        mod_initializer, mod_input = pack_initializer(data, mod_input)
        m_graph = onnx.helper.make_graph(mod_node, model.graph.name, mod_input, model.graph.output, mod_initializer)
        m_model = onnx.helper.make_model(m_graph, producer_name=model.producer_name, producer_version=model.producer_version, opset_imports=model.opset_import)
        model = m_model
        print(".", end='', flush=True)

    if count > 0:
        mod_initializer, mod_input = pack_initializer(data, mod_input)
        m_graph = onnx.helper.make_graph(mod_node, model.graph.name, mod_input, model.graph.output, mod_initializer)
        model = onnx.helper.make_model(m_graph, producer_name=model.producer_name, producer_version=model.producer_version, opset_imports=model.opset_import)
        print("")
        print("    [i] report : convert {} node to initializer. remove {} node.".format(count, rem_count))

    return model


def onnx_remove_duplicated_node(model):

    # list up initializer
    data = listup_initializer(model)

    # setup node connection (input to output) dict + node dict
    cnct = {}
    node = {}
    oname = []
    for o in model.graph.output:
        oname.append(o.name)
    src_names = []
    for i in model.graph.input:
        src_names.append(i.name)
    src_names.extend(data.keys())
    for n in model.graph.node:
        id = n.output[0]
        src_names.append(id)
        for o in n.output:
            for i in n.input:
                cnct.setdefault(i, [])
                if o in cnct[i]:
                    continue
                cnct[i].append(o)
        node[id] = n

    replace = {}
    count = {}
    # pickup duplicated node
    for id in src_names:
        ol = cnct.get(id, [])
        if len(ol) < 2:
            continue
        rm = []
        for i in range(0, len(ol)):
            if i in rm:
                continue
            if ol[i] in replace:
                continue
            if not ol[i] in node:
                continue
            ref = node[ol[i]]
            for j in range(i + 1, len(ol)):
                if not ol[j] in node:
                    continue
                trg = node[ol[j]]
                if not check_duplicated_node(ref, trg, data, replace):
                    continue
                rm.append(j)
                for oi in range(len(ref.output)):
                    kn = ref.output[oi]
                    rn = trg.output[oi]
                    while kn in replace:
                        kn = replace[kn]
                    replace[rn] = kn
                    if not rn in cnct:
                        continue
                    for x in cnct[rn]:
                        if x in cnct[kn]:
                            continue
                        cnct[kn].append(x)
                count.setdefault(trg.op_type, 0)
                count[trg.op_type] += 1

    if len(replace) == 0:
        # no removable node
        return model

    # remove duplicated node
    mod_node = []
    for n in model.graph.node:
        id = n.output[0]

        # update node input : for all node
        for i in range(len(n.input)):
            k = n.input[i]
            if (k in replace.keys()):
                n.input[i] = replace[k]

        n = adjust_node_with_subgraph(n, replace)

        if not id in replace:
            # unremovable
            mod_node.append(n)
            continue

        # removable
        for i in n.input:
            if not i in data:
                continue
            data[i]['ref'] -= 1

    for k in count.keys():
        print("    [i] report : remove {0} duplicated {1} node.".format(count[k], k))

    adjust_graph_output(mod_node, replace, model)

    mod_initializer, mod_input = pack_initializer(data, model.graph.input)
    m_graph = onnx.helper.make_graph(mod_node, model.graph.name, mod_input, model.graph.output, mod_initializer)
    m_model = onnx.helper.make_model(m_graph, producer_name=model.producer_name, producer_version=model.producer_version, opset_imports=model.opset_import)

    return m_model


def onnx_manual_optimization(model, json_path):

    # expected json data format
    # ----
    # {
    #     "remove_node" : [ # list of target node 1st output blob names
    #         "foo",
    #         "bar",
    #         ...
    #     ],
    #     "replace_blob" : { # dictionary of target blob names
    #         "old_blob_name": "modified_blob_name",
    #         ...
    #     },
    #     "insert_node" : {
    #       "trigger": {
    #         "input": ["", ...],
    #         "output": ["", ...],
    #         "op_type": "",
    #         "attribute": {
    #           ...
    #         }
    #       },
    #       ...
    #     },
    #     "initializer": [
    #       {
    #         "name": "",
    #         "value": [x, y, z, ...]
    #         "dtype": "",
    #       },
    #       ...
    #     ]
    # }
    with open(json_path) as f:
        order = json.load(f)

    remove = order.get("remove_node", [])
    replace = order.get("replace_blob", {})
    insert = order.get("insert_node", {})

    # list up initializer
    data = listup_initializer(model)

    if "initializer" in order:
        initializer = order['initializer']
        for i in initializer:
            data[i["name"]] = {'ref': 0, 'id': i["name"], 'body': numpy.array(i["value"], dtype=i["dtype"])}

    available = []
    available.extend(data.keys())
    for e in model.graph.input:
        available.append(e.name)

    count = 0
    add_count = 0
    mod_node = []
    for n in model.graph.node:
        id = n.output[0]

        for i in range(len(n.input)):
            k = n.input[i]
            # empty string indicate omitted input - skip processing
            if len(k) == 0:
                continue
            if k in replace:
                if k in data:
                    data[k]['ref'] -= 1
                k = replace[k]
                if k in data:
                    data[k]['ref'] += 1
                n.input[i] = k
            if not k in available:
                remove.append(id)
                break

        if id in remove:
            count += 1
            for i in n.input:
                if k in data:
                    data[k]['ref'] -= 1
            continue

        for o in n.output:
            available.append(o)

        mod_node.append(n)

        insertion_target = copy.deepcopy(n.output)
        for o in insertion_target:
            if not o in insert:
                continue
            insert_node = insert[o]
            attribute = insert_node.get("attribute", {})
            mod_node.append(onnx.helper.make_node(op_type=insert_node["op_type"], inputs=insert_node["input"], outputs=insert_node["output"], **attribute))
            for input in insert_node["input"]:
                if input in data:
                    data[input]['ref'] += 1
            for output in insert_node["output"]:
                insertion_target.append(output)
                available.append(output)
            add_count += 1

    if count == 0:
        return model

    print("    [i] report : remove {} nodes with \'{}\'.".format(count, json_path.name))
    print("    [i] report : insert {} nodes with \'{}\'.".format(add_count, json_path.name))

    mod_initializer, mod_input = pack_initializer(data, model.graph.input)
    m_graph = onnx.helper.make_graph(mod_node, model.graph.name, mod_input, model.graph.output, mod_initializer)
    m_model = onnx.helper.make_model(m_graph, producer_name=model.producer_name, producer_version=model.producer_version, opset_imports=model.opset_import)

    return m_model


def onnx_convert_pow_div_to_mul(model):
    data = listup_initializer(model)

    count = 0
    mod_node = []
    for n in model.graph.node:
        if not (n.op_type == 'Pow' or n.op_type == 'Div'):
            mod_node.append(n)
            continue
        a_id = n.input[0]
        b_id = n.input[1]
        if not b_id in data:
            mod_node.append(n)
            continue
        value = data[b_id]['body']

        if n.op_type == 'Pow':
            if not data[b_id]['body'] == 2:
                mod_node.append(n)
                continue
            data[b_id]['ref'] -= 1
            mod_node.append(onnx.helper.make_node(op_type='Mul', inputs=[a_id, a_id], outputs=n.output))
            count += 1
        if n.op_type == 'Div':
            if 'int' in str(value.dtype):
                mod_node.append(n)
                continue
            data[b_id]['ref'] -= 1
            b_id = b_id + '.opt'
            if b_id in data:
                data[b_id]['ref'] += 1
                mod_node.append(onnx.helper.make_node(op_type='Mul', inputs=[a_id, b_id], outputs=n.output))
            else:
                data[b_id] = {'ref': 1, 'id': b_id, 'body': numpy.reciprocal(value)}
                mod_node.append(onnx.helper.make_node(op_type='Mul', inputs=[a_id, b_id], outputs=n.output))
            count += 1

    if count == 0:
        return model

    print("    [i] report : convert {} pow or div node to mul with 'onnx_convert_pow_div_to_mul'".format(count))

    mod_initializer, mod_input = pack_initializer(data, model.graph.input)
    m_graph = onnx.helper.make_graph(mod_node, model.graph.name, mod_input, model.graph.output, mod_initializer)
    m_model = onnx.helper.make_model(m_graph, producer_name=model.producer_name, producer_version=model.producer_version, opset_imports=model.opset_import)

    return m_model


def onnx_remove_needless_reshape(model):
    # list up initializer
    data = listup_initializer(model)
    # shape inference
    info = ShapeAndTypeInfo(model)
    count = 0

    # make reshape output list
    reshape_output = []
    reshape_input = []   # reshape input blob list
    reshape_node_o = {}  # output to input list dict
    reshape_out_n = {}
    reshape_shape = {}
    for n in model.graph.node:
        if n.op_type == 'Reshape':
            for i in n.output:
                reshape_output.append(i)
                reshape_node_o[i] = n.input
                reshape_out_n.setdefault(i, 0)
            for i in n.input:
                reshape_input.append(i)
                if i in reshape_out_n:
                    reshape_out_n[i] += 1
            if n.input[1] in data:
                reshape_shape[n.output[0]] = data[n.input[1]]['body']
        else:
            for o in n.input:
                if o in reshape_out_n:
                    reshape_out_n[o] += 1
    # pickup consecutive reshapes
    remove = []
    replace = {}
    for i in reshape_output:
        # detect seriese reshape
        if i in reshape_input and reshape_out_n[i] == 1:
            replace[i] = reshape_node_o[i][0]
            remove.append(i)
            continue
        # detect meanless reshape
        if not i in reshape_shape:
            continue
        if not info.is_inferred(reshape_node_o[i][0]):
            continue
        node_shape = list(info.get_shape(reshape_node_o[i][0]))
        node_reshape = list(reshape_shape[i])
        if 0 in node_reshape:
            for idx, val in enumerate(node_reshape):
                if val != 0:
                    continue
                node_reshape[idx] = node_reshape[idx]
        if -1 in node_reshape:
            x = numpy.prod(node_shape) // numpy.prod(node_reshape) * -1
            node_reshape[node_reshape.index(-1)] = x
        if node_shape == node_reshape:
            replace[i] = reshape_node_o[i][0]
            remove.append(i)
    mod_node = []
    for n in model.graph.node:
        id = n.output[0]

        # replace blob
        for i in range(len(n.input)):
            k = n.input[i]
            if k in replace:
                if k in data:
                    data[k]['ref'] -= 1
                while k in replace:
                    k = replace[k]
                if k in data:
                    data[k]['ref'] += 1
                n.input[i] = k
                count += 1

        # remove
        if id in remove:
            for i in n.input:
                if i in data:
                    data[i]['ref'] -= 1
            continue

        mod_node.append(n)

    if count == 0:
        return model

    print("    [i] report : remove {} reshape nodes with \'onnx_remove_needless_reshape\'.".format(count))

    mod_initializer, mod_input = pack_initializer(data, model.graph.input)
    m_graph = onnx.helper.make_graph(mod_node, model.graph.name, mod_input, model.graph.output, mod_initializer)
    m_model = onnx.helper.make_model(m_graph, producer_name=model.producer_name, producer_version=model.producer_version, opset_imports=model.opset_import)

    return m_model


def onnx_remove_needless_cast(model):

    info = ShapeAndTypeInfo(model)

    # list up cast node
    node_cast = {}
    cnct = {}  # input to output connection
    for n in model.graph.node:
        for o in n.output:
            for i in n.input:
                cnct.setdefault(i, [])
                cnct[i].append(o)
        id = n.output[0]  # node main output
        if (n.op_type == 'Cast'):
            node_cast[id] = {'o': n.output, 'i': n.input}

    count = 0
    mod_node = []
    replace = {}
    for n in model.graph.node:
        id = n.output[0]

        # update node input : for all node
        for i, k in enumerate(n.input):
            if k in replace:
                n.input[i] = replace[k]

        if n.op_type != 'Cast':
            n = adjust_node_with_subgraph(n, replace)
            mod_node.append(n)
            continue

        to = pick_attribute(n, 'to').i
        input_type = info.get_type(n.input[0])

        remove = False
        if (id in cnct) and (len(cnct[id]) == 1) and (cnct[id][0] in node_cast):
            # consecutive cast
            remove = True
        elif (to == input_type):
            # input and output datatype are same
            remove = True

        if not remove:
            mod_node.append(n)
            continue

        count += 1
        replace[id] = n.input[0]

    if (count == 0):
        return model

    # adjust blob name related graph output
    adjust_graph_output(mod_node, replace, model)

    print(f'    [i] report : remove {count} cast node with \'onnx_remove_needless_cast\'.')

    m_graph = onnx.helper.make_graph(mod_node, model.graph.name, model.graph.input, model.graph.output, model.graph.initializer)
    m_model = onnx.helper.make_model(m_graph, producer_name=model.producer_name, producer_version=model.producer_version, opset_imports=model.opset_import)

    return m_model


def onnx_replace_to_meanvarnorm(model):

    # list up initializer
    data = listup_initializer(model)

    # list up replace node candidate
    cand = {}  # candidate
    cnct = {}  # input to output connection
    ops = ['Sub', 'Pow', 'Add', 'Sqrt', 'Div', 'Mul']
    for n in model.graph.node:
        for o in n.output:
            for i in n.input:
                cnct.setdefault(i, [])
                cnct[i].append(o)
        id = n.output[0]  # node main output
        if (n.op_type == 'ReduceMean'):
            kd = pick_attribute(n, 'keepdims')
            kd = 1 if (kd == None) else kd.i
            ax = pick_attribute(n, 'axes')
            ax = None if (ax == None) else tuple(ax.ints)
            if kd == 0:
                continue
            cand[id] = {'type': n.op_type, 'o': n.output, 'i': n.input, 'ax': ax}
        elif (n.op_type in ops):
            cand[id] = {'type': n.op_type, 'o': n.output, 'i': n.input}

    # check node connection
    insert = {}
    remove = []
    for dst_name in cand.keys():
        # dst_node = meanvarnorm( src_node )
        #
        # dst_node = div( node0, node1 )
        # node0 = sub( src_node, node2 )
        # node2 = reducemean( src_node )
        # node1 = sqrt( node3 )
        # node3 = add( node4, epsilon )
        # node4 = reducemean( node5 )
        # node5 = pow( node0, 2 )

        unref = []

        dst_node = cand[dst_name]
        if dst_node['type'] != 'Div':
            continue
        name0 = dst_node['i'][0]
        name1 = dst_node['i'][1]

        if not name0 in cand:
            continue
        node0 = cand[name0]
        if node0['type'] != 'Sub':
            continue
        name2 = node0['i'][1]
        src_name = node0['i'][0]

        if not name2 in cand:
            continue
        node2 = cand[name2]
        if node2['type'] != 'ReduceMean':
            continue
        if node2['i'][0] != src_name:
            continue

        if not name1 in cand:
            continue
        node1 = cand[name1]
        if node1['type'] == 'Pow':
            pp_name = node1['i'][1]
            if not pp_name in data:
                continue
            pp = data[pp_name]['body']
            if pp.size != 1:
                continue
            if pp.ndim == 0:
                pp = numpy.asarray([pp], dtype=pp.dtype)
            if abs(pp[0] - 0.5) > 1e-7:
                continue
            unref.append(pp_name)
        elif node1['type'] != 'Sqrt':
            continue
        name3 = node1['i'][0]

        if not name3 in cand:
            continue
        node3 = cand[name3]
        epsilon = 0.0
        if node3['type'] == 'Add':
            ap_name = node3['i'][1]
            if not ap_name in data:
                continue
            ap = data[ap_name]['body']
            if ap.size != 1:
                continue
            epsilon = ap.item()
            unref.append(ap_name)
            name4 = node3['i'][0]
        else:
            node3 = None
            name4 = name3

        if epsilon > 1.0e-5:
            continue

        if not name4 in cand:
            continue
        node4 = cand[name4]
        if node4['type'] != 'ReduceMean':
            continue
        name5 = node4['i'][0]

        if node2['ax'] != node4['ax']:
            continue
        axes = node2['ax']

        if not name5 in cand:
            continue
        node5 = cand[name5]
        if node5['type'] == 'Mul':
            if node5['i'][0] != node5['i'][1]:
                continue
        elif node5['type'] == 'Pow':
            pp_name = node5['i'][1]
            if not pp_name in data:
                continue
            pp = data[pp_name]['body']
            if pp.size != 1:
                continue
            if pp.ndim == 0:
                pp = numpy.asarray([pp], dtype=pp.dtype)
            if abs(pp[0] - 2.0) > 1e-7:
                continue
            unref.append(pp_name)
        else:
            continue
        if node5['i'][0] != name0:
            continue

        if len(cnct[name2]) != 1:
            continue
        if len(cnct[name0]) != 2:
            continue
        if len(cnct[name5]) != 1:
            continue
        if len(cnct[name4]) != 1:
            continue
        if node3 and (len(cnct[name3]) != 1):
            continue
        if len(cnct[name1]) != 1:
            continue

        for e in unref:
            data[e]['ref'] -= 1

        remove.append(dst_name)
        remove.append(name0)
        remove.append(name1)
        remove.append(name2)
        if node3 != None:
            remove.append(name3)
        remove.append(name4)
        remove.append(name5)

        insert[dst_name] = {'op_type': 'MeanVarianceNormalization', 'input': [src_name], 'output': [dst_name], 'attribute': {'axes': axes}}

    if len(insert) == 0:
        return model

    print(f'    [i] report : replace {len(insert)} node group to MeanVarianceNormalization.')

    mod_node = []
    for n in model.graph.node:
        id = n.output[0]
        if not id in remove:
            mod_node.append(n)
        if id in insert:
            op_type = insert[id]['op_type']
            input = insert[id]['input']
            output = insert[id]['output']
            attribute = insert[id].get('attribute', {})
            mod_node.append(onnx.helper.make_node(op_type, input, output, **attribute))

    mod_initializer, mod_input = pack_initializer(data, model.graph.input)
    m_graph = onnx.helper.make_graph(mod_node, model.graph.name, mod_input, model.graph.output, mod_initializer)
    m_model = onnx.helper.make_model(m_graph, producer_name=model.producer_name, producer_version=model.producer_version, opset_imports=model.opset_import)

    return m_model


def onnx_replace_transpose_to_reshape(model):
    OL = 'o'
    IL = 'i'
    info = ShapeAndTypeInfo(model)
    data = listup_initializer(model)

    cand_transpose = []
    for node in model.graph.node:
        if not node.op_type == 'Transpose':
            continue
        attr = listup_attribute(node)
        nmo = node.output[0]
        cand_transpose.append({OL: nmo, IL: node.input, 'attr': attr})

    target = {}
    count = 0
    for tr in cand_transpose:
        # check transpose can replace reshape
        if not (info.is_inferred(tr[OL]) and info.is_inferred(tr[IL][0])):
            continue
        # calc input/output stride
        input_shape = info.get_shape(tr[IL][0])
        output_shape = info.get_shape(tr[OL])
        perm = tr['attr']['perm']
        input_stride = [1] * len(input_shape)
        output_stride = [1] * len(output_shape)
        for i in reversed(range(len(input_shape))):
            if i == 0:
                break
            input_stride[i - 1] = input_stride[i] * input_shape[i]
            output_stride[i - 1] = output_stride[i] * output_shape[i]
        continuous = True
        for p_idx, p in enumerate(perm):
            if input_shape[p] == 1:
                continue
            if output_stride[p_idx] != input_stride[p]:
                continuous = False
                break
        if not continuous:
            continue
        count += 1
        tr['shape'] = output_shape
        target[tr[OL]] = tr

    if count == 0:
        return model

    print("    [i] report : replace {} Transpose to Reshape.".format(count))

    mod_node = []
    for n in model.graph.node:
        id = n.output[0]
        if not (id in target):
            mod_node.append(n)
            continue

        wi = target[id]
        shape = wi['shape']
        reshape_input = 'opt.reshape_' + wi[IL][0]
        data[reshape_input] = {'ref': 1, 'id': reshape_input, 'body': numpy.array(shape).astype(numpy.int64)}
        nx = onnx.helper.make_node('Reshape', [wi[IL][0], reshape_input], [wi[OL]])
        mod_node.append(nx)

    mod_initializer, mod_input = pack_initializer(data, model.graph.input)
    m_graph = onnx.helper.make_graph(mod_node, model.graph.name, mod_input, model.graph.output, mod_initializer)
    m_model = onnx.helper.make_model(m_graph, producer_name=model.producer_name, producer_version=model.producer_version, opset_imports=model.opset_import)

    return m_model


def onnx_replace_expand_to_reshape(model):
    OL = 'o'
    IL = 'i'
    info = ShapeAndTypeInfo(model)
    data = listup_initializer(model)

    target = {}
    for node in model.graph.node:
        if not node.op_type == 'Expand':
            continue
        if not node.input[1] in data:
            continue
        ss = info.get_shape(node.input[0])
        if ss == None:
            continue
        ds = data[node.input[1]]['body']
        slen = int(numpy.prod(ss))
        dlen = int(numpy.prod(ds))
        if slen == dlen:
            id = node.output[0]
            target[id] = {OL: id, IL: node.input}

    count = len(target)
    if count == 0:
        return model

    print("    [i] report : replace {} Expand to Reshape.".format(count))

    mod_node = []
    for n in model.graph.node:
        id = n.output[0]
        if not (id in target):
            mod_node.append(n)
            continue

        wi = target[id]
        nx = onnx.helper.make_node('Reshape', wi[IL], [wi[OL]])
        mod_node.append(nx)

    mod_initializer, mod_input = pack_initializer(data, model.graph.input)
    m_graph = onnx.helper.make_graph(mod_node, model.graph.name, mod_input, model.graph.output, mod_initializer)
    m_model = onnx.helper.make_model(m_graph, producer_name=model.producer_name, producer_version=model.producer_version, opset_imports=model.opset_import)

    return m_model


def listup_subgraph_external_ref(subgraph):
    rv = []  # return value - external references
    ex = set()  # exclude set - subgraph's local blob
    for e in subgraph.input:
        ex.add(e.name)
    for e in subgraph.initializer:
        ex.add(e.name)
    for n in subgraph.node:
        for e in n.output:
            ex.add(e)
        i = [e for e in n.input]
        if (n.op_type == 'If'):
            sg = onnx.helper.get_attribute_value(pick_attribute(n, 'else_branch'))
            xi = listup_subgraph_external_ref(sg)
            i.extend(xi)
            sg = onnx.helper.get_attribute_value(pick_attribute(n, 'then_branch'))
            xi = listup_subgraph_external_ref(sg)
            i.extend(xi)
        elif (n.op_type == 'Loop') or (n.op_type == 'Scan'):
            sg = onnx.helper.get_attribute_value(pick_attribute(n, 'body'))
            xi = listup_subgraph_external_ref(sg)
            i.extend(xi)
        i = set(i)  # remove duplicate member
        for e in i:
            if e in ex:
                continue
            rv.append(e)
    return rv


def onnx_remove_node_no_output(model):
    data = listup_initializer(model)
    gin = set()
    for e in model.graph.input:
        gin.add(e.name)

    # 1st step - setup node connections
    o2n = {}  # output -> node_id dict
    n2i = {}  # node_id -> input dict
    nids = set()  # node_id set
    for n in model.graph.node:
        id = n.output[0]
        i = [e for e in n.input]
        nids.add(id)
        if (n.op_type == 'If'):
            sg = onnx.helper.get_attribute_value(pick_attribute(n, 'else_branch'))
            xi = listup_subgraph_external_ref(sg)
            i.extend(xi)
            sg = onnx.helper.get_attribute_value(pick_attribute(n, 'then_branch'))
            xi = listup_subgraph_external_ref(sg)
            i.extend(xi)
        elif (n.op_type == 'Loop') or (n.op_type == 'Scan'):
            sg = onnx.helper.get_attribute_value(pick_attribute(n, 'body'))
            xi = listup_subgraph_external_ref(sg)
            i.extend(xi)
        for e in n.output:
            o2n[e] = id
        n2i[id] = []
        for e in set(i):
            if e == '':
                continue
            if e in data:
                continue
            if e in gin:
                continue
            n2i[id].append(e)

    # 2nd step - check working node (connection from output)
    target = []
    for e in model.graph.output:
        target.append(e.name)
    for e in target:
        if not e in o2n:
            continue
        id = o2n[e]
        if not id in nids:
            continue
        nids.remove(id)
        target.extend(n2i[id])

    if len(nids) == 0:
        return model

    print("    [i] report : remove {} node with 'onnx_remove_node_no_output'".format(len(nids)))

    # 3rd step - select graph node
    mod_node = []
    for n in model.graph.node:
        id = n.output[0]
        if not id in nids:
            mod_node.append(n)
        else:
            for i in n.input:
                if i in data:
                    data[i]['ref'] -= 1

    mod_initializer, mod_input = pack_initializer(data, model.graph.input)
    m_graph = onnx.helper.make_graph(mod_node, model.graph.name, mod_input, model.graph.output, mod_initializer)
    m_model = onnx.helper.make_model(m_graph, producer_name=model.producer_name, producer_version=model.producer_version, opset_imports=model.opset_import)

    return m_model


def onnx_summarize_slice_input(mer, abp):
    # sort by axes
    idx = numpy.argsort(mer['axes'])
    mer['axes'], mer['starts'], mer['ends'], mer['steps'] = mer['axes'][idx], mer['starts'][idx], mer['ends'][idx], mer['steps'][idx]
    idx = numpy.argsort(abp['axes'])
    abp['axes'], abp['starts'], abp['ends'], abp['steps'] = abp['axes'][idx], abp['starts'][idx], abp['ends'][idx], abp['steps'][idx]

    # merge axes
    merged_ax = list(abp['axes'])
    for mer_ax in mer['axes']:
        if mer_ax in abp['axes']:
            continue
        merged_ax.append(mer_ax)
    merged_ax.sort()
    if len(merged_ax) > len(abp['axes']):
        if not ((merged_ax[0] >= 0 and merged_ax[-1] >= 0) or (merged_ax[0] < 0 and merged_ax[-1] < 0)):
            return None, None, None, None, False

    starts = []
    ends = []
    axes = []
    steps = []
    mer_idx = 0
    abp_idx = 0
    for idx, ax in enumerate(merged_ax):
        mer_ax = mer['axes'][mer_idx] if len(mer['axes']) > mer_idx else None
        abp_ax = abp['axes'][abp_idx] if len(abp['axes']) > abp_idx else None

        if ax == mer_ax and ax == abp_ax:
            if mer['steps'][mer_idx] != abp['steps'][abp_idx]:
                return None, None, None, None, False
            # make up start and ends signs
            if not ((mer['starts'][mer_idx] >= 0 and abp['starts'][abp_idx] >= 0) or (mer['starts'][mer_idx] < 0 and abp['starts'][abp_idx] < 0)):
                return None, None, None, None, False
            if not ((mer['ends'][mer_idx] >= 0 and abp['ends'][abp_idx] >= 0) or (mer['ends'][mer_idx] < 0 and abp['ends'][abp_idx] < 0)):
                return None, None, None, None, False

            mer_par = [mer['starts'][mer_idx], mer['ends'][mer_idx], mer['steps'][mer_idx]]
            abp_par = [abp['starts'][abp_idx], abp['ends'][abp_idx], abp['steps'][abp_idx]]
            x = range(sys.maxsize)
            y = x[abp_par[0]:abp_par[1]:abp_par[2]]
            z = y[mer_par[0]:mer_par[1]:mer_par[2]]

            # start
            if mer['starts'][mer_idx] >= 0 and abp['starts'][abp_idx] >= 0:
                if mer['steps'][mer_idx] > 0:
                    start = z.start
                else:
                    if numpy.sign(mer['starts'][mer_idx]) == numpy.sign(abp['ends'][abp_idx]):
                        start = z.stop
                    else:
                        start = mer['starts'][mer_idx] + abp['starts'][abp_idx]
            else:
                if (mer['starts'][mer_idx] >= 0 and abp['ends'][abp_idx] >= 0) or (mer['starts'][mer_idx] < 0 and abp['ends'][abp_idx] < 0):
                    start_1st = abp['starts'][abp_idx]
                    start_2nd = mer['starts'][mer_idx] * abs(abp['steps'][abp_idx]) + abp['ends'][abp_idx]
                    start = min(start_1st, start_2nd)
                else:
                    start = -1 * z.start

            # end
            if mer['ends'][mer_idx] >= 0 and abp['ends'][abp_idx] >= 0:
                if mer['steps'][mer_idx] > 0:
                    end = z.stop
                else:
                    if mer['starts'][mer_idx] >= 0 and abp['starts'][abp_idx] >= 0 and abp['ends'][abp_idx] >= 0 and mer['ends'][mer_idx] >= 0:
                        end = z.start
                    elif numpy.sign(mer['starts'][mer_idx]) == numpy.sign(abp['ends'][abp_idx]):
                        end_1st = abp['ends'][abp_idx]
                        end_2nd = mer['ends'][mer_idx] * abs(abp['steps'][abp_idx]) + abp['starts'][abp_idx]
                        end = min(end_1st, end_2nd)
                    else:
                        end = sys.maxsize - z.stop
            else:
                if mer['steps'][mer_idx] > 0:
                    end = z.stop - sys.maxsize
                else:
                    end_1st = abp['ends'][abp_idx]
                    end_2nd = mer['ends'][mer_idx] * abs(abp['steps'][abp_idx]) + abp['ends'][abp_idx]
                    end = min(end_1st, end_2nd)

            steps.append(mer['steps'][mer_idx] * abp['steps'][abp_idx])
            if mer['steps'][mer_idx] < 0:
                start, end = end, start
            mer_idx += 1
            abp_idx += 1
        elif ax == mer_ax:
            start = mer['starts'][mer_idx]
            end = mer['ends'][mer_idx]
            steps.append(mer['steps'][mer_idx])
            mer_idx += 1
        elif ax == abp_ax:
            start = abp['starts'][abp_idx]
            end = abp['ends'][abp_idx]
            steps.append(abp['steps'][abp_idx])
            abp_idx += 1
        else:
            return None, None, None, None, False

        starts.append(start)
        ends.append(end)
        axes.append(ax)

    return numpy.array(starts, dtype=numpy.int64), numpy.array(ends, dtype=numpy.int64), numpy.array(axes, dtype=numpy.int64), numpy.array(steps, dtype=numpy.int64), True


def onnx_summarize_seriese_slice(model):

    # slice-1 is unsupported
    ver = model.opset_import[0].version
    if ver < 10:
        return model

    data = listup_initializer(model)
    oname = [o.name for o in model.graph.output]

    # list up slice
    OL = 'o'
    IL = 'i'
    slc_cand = {}
    for n in model.graph.node:
        for i in n.input:
            if i in slc_cand:
                slc_cand[i]['ref'] += 1
        if n.op_type == 'Slice':
            flg = True
            for i in range(1, len(n.input)):
                if not n.input[i] in data:
                    if n.input[i] == '':
                        continue
                    flg = False
                    break
            if not flg:
                continue
            starts = data[n.input[1]]['body']
            ends = data[n.input[2]]['body']
            if len(n.input) > 3:
                if n.input[3] == '':
                    axes = numpy.arange(len(starts), dtype=numpy.int64)
                else:
                    axes = data[n.input[3]]['body']
            else:
                axes = numpy.arange(len(starts), dtype=numpy.int64)
            if len(n.input) > 4:
                if n.input[4] == '':
                    steps = numpy.ones(len(starts), dtype=numpy.int64)
                else:
                    steps = data[n.input[4]]['body']
            else:
                steps = numpy.ones((len(starts),), dtype=numpy.int64)

            slc_cand[n.output[0]] = {OL: n.output[0], IL: n.input, 'ref': 0, 'starts': starts, 'ends': ends, 'axes': axes, 'steps': steps}
    for o in model.graph.output:
        i = o.name
        if i in slc_cand:
            slc_cand[i]['ref'] += 1

    # check can be merged
    replace_node = {}
    remove_node = []
    count = 0
    for elem in slc_cand.values():
        sid = elem[IL][0]
        cid = elem[OL]
        if (not sid in slc_cand):
            continue
        starts, ends, axes, steps, valid = onnx_summarize_slice_input(elem, slc_cand[sid])
        if not valid:
            continue
        slc_cand[sid]['ref'] -= 1
        if slc_cand[sid]['ref'] <= 0:
            remove_node.append(sid)
            if sid in replace_node:
                replace_node.pop(sid)
        cur_ref = slc_cand[cid]['ref']
        replace_node[cid] = {
            IL: [slc_cand[sid][IL][0]],
            OL: elem[OL],
            'ref': cur_ref,
            'starts': starts,
            'ends': ends,
            'axes': axes,
            'steps': steps}
        slc_cand[cid] = replace_node[cid]
        count += 1

    if count == 0:
        return model

    mod_node = []
    for n in model.graph.node:
        id = n.output[0]

        # delete initializer from remove_node
        if id in remove_node:
            for i in n.input:
                if not i in data:
                    continue
                data[i]['ref'] -= 1
            continue

        if id in replace_node:  # change initilizer. replace input and output to former slice
            if data[n.input[1]]['ref'] == 1:
                data[n.input[1]]['body'] = replace_node[id]['starts']
            else:
                did = 'opt_starts_' + id
                data[did] = {'ref': 1, 'id': did, 'body': replace_node[id]['starts']}
                data[n.input[1]]['ref'] -= 1
                n.input[1] = did
            if data[n.input[2]]['ref'] == 1:
                data[n.input[2]]['body'] = replace_node[id]['ends']
            else:
                did = 'opt_ends_' + id
                data[did] = {'ref': 1, 'id': did, 'body': replace_node[id]['ends']}
                data[n.input[2]]['ref'] -= 1
                n.input[2] = did
            if len(n.input) > 3:
                if n.input[3] == '':
                    did = 'opt_axes_' + id
                    data[did] = {'ref': 1, 'id': did, 'body': replace_node[id]['axes']}
                    n.input[3] = did
                elif data[n.input[3]]['ref'] == 1:
                    data[n.input[3]]['body'] = replace_node[id]['axes']
                else:
                    did = 'opt_axes_' + id
                    data[did] = {'ref': 1, 'id': did, 'body': replace_node[id]['axes']}
                    data[n.input[3]]['ref'] -= 1
                    n.input[3] = did
            if len(n.input) > 4:
                if n.input[4] == '':
                    did = 'opt_steps_' + id
                    data[did] = {'ref': 1, 'id': did, 'body': replace_node[id]['steps']}
                    n.input[4] = did
                if data[n.input[4]]['ref'] == 1:
                    data[n.input[4]]['body'] = replace_node[id]['steps']
                else:
                    did = 'opt_steps_' + id
                    data[did] = {'ref': 1, 'id': did, 'body': replace_node[id]['steps']}
                    data[n.input[4]]['ref'] -= 1
                    n.input[4] = did

            n.input[0] = replace_node[id][IL][0]

        mod_node.append(n)

    print("    [i] report : summarize {} slice nodes with \'onnx_summarize_seriese_slice\'.".format(count))

    mod_initializer, mod_input = pack_initializer(data, model.graph.input)
    m_graph = onnx.helper.make_graph(mod_node, model.graph.name, mod_input, model.graph.output, mod_initializer)
    m_model = onnx.helper.make_model(m_graph, producer_name=model.producer_name, producer_version=model.producer_version, opset_imports=model.opset_import)

    return m_model


def onnx_replace_seriese_max_min(model):
    data = listup_initializer(model)

    OL = 'o'
    IL = 'i'
    # list up transpose node and connection
    cand = {}  # candidate
    cnct = {}  # conection
    for n in model.graph.node:
        for o in n.output:
            for i in n.input:
                cnct.setdefault(i, [])
                cnct[i].append(o)
        nmo = n.output[0]  # node main output
        if (n.op_type == 'Min' or n.op_type == 'Max'):
            if len(n.input) > 2:
                continue
            if n.input[1] in data:
                ni = n.input[0]
                ci = n.input[1]  # constant input
            elif n.input[0] in data:
                ci = n.input[0]  # constant input
                ni = n.input[1]
            else:
                continue
            if data[ci]['body'].size != 1:  # clip parameter requires scalar
                continue
            cand[nmo] = {'type': n.op_type, OL: nmo, IL: [ni, ci]}

    # verify candidate connection
    count = 0
    target = {}
    for n in cand.values():
        wi = n
        if wi[OL] in target:
            continue
        if not (wi[IL][0] in cand):
            continue
        next = wi
        start = cand[wi[IL][0]]
        if (next['type'] == start['type']):  # detect min->min or max->max pattern
            continue
        if (len(cnct[next[IL][0]]) > 1):
            continue

        if start['type'] == 'start':
            min = start[IL][1]
            max = next[IL][1]
        else:
            min = next[IL][1]
            max = start[IL][1]

        if (len(data[min]['body'].shape) != 0):
            data[min]['body'] = numpy.reshape(data[min]['body'], ())
        if (len(data[max]['body'].shape) != 0):
            data[max]['body'] = numpy.reshape(data[max]['body'], ())

        if data[min]['body'] < data[max]['body']:
            target[start[OL]] = {'type': 'Clip', OL: next[OL], IL: [start[IL][0], min, max]}
        else:
            target[start[OL]] = {'type': 'Clip', OL: next[OL], IL: [start[IL][0], max, min]}
        target[next[OL]] = {'type': None}
        count += 1

    if count == 0:
        return model

    print("    [i] report : replace {} Max+Min to Clip with \'onnx_replace_seriese_max_min\'".format(count))

    mod_node = []
    for n in model.graph.node:
        id = n.output[0]

        if not id in target:
            mod_node.append(n)
            continue

        if target[id]['type'] == None:
            continue

        wi = target[id]
        nx = onnx.helper.make_node(wi['type'], wi[IL], [wi[OL]])
        mod_node.append(nx)

    m_graph = onnx.helper.make_graph(mod_node, model.graph.name, model.graph.input, model.graph.output, model.graph.initializer)
    m_model = onnx.helper.make_model(m_graph, producer_name=model.producer_name, producer_version=model.producer_version, opset_imports=model.opset_import)

    return m_model


def onnx_summarize_gather_slice(model):
    ver = model.opset_import[0].version
    if ver < 10:
        return model
    # list up initializer
    data = listup_initializer(model)
    # shape inference
    info = ShapeAndTypeInfo(model)

    # list up slice
    OL = 'o'
    IL = 'i'
    cand = {}
    cnct = {}
    for n in model.graph.node:
        for o in n.output:
            for i in n.input:
                cnct.setdefault(i, [])
                cnct[i].append(o)
        nmo = n.output[0]
        if (n.op_type == 'Gather'):
            if not n.input[1] in data:
                continue
            if not len(data[n.input[1]]['body'].shape) == 0:
                continue
            attr = listup_attribute(n)
            cand[nmo] = {'type': n.op_type, OL: nmo, IL: n.input, 'attr': attr}
        elif (n.op_type == 'Slice'):
            if n.input[0] in data or (not n.input[1] in data) or n.input[0] in data:
                continue
            starts = numpy.array(data[n.input[1]]['body'])
            if len(n.input) > 3:
                if n.input[3] == '':
                    axes = numpy.arange(len(starts), dtype=numpy.int64)
                else:
                    axes = numpy.array(data[n.input[3]]['body'])
            else:
                axes = numpy.arange(len(starts), dtype=numpy.int64)
            if len(n.input) > 4:
                if n.input[4] == '':
                    steps = numpy.ones(len(starts), dtype=numpy.int64)
                else:
                    steps = numpy.array(data[n.input[4]]['body'])
            else:
                steps = numpy.ones((len(starts),), dtype=numpy.int64)
            cand[nmo] = {'type': n.op_type, OL: nmo, IL: n.input, 'starts': starts, 'axes': axes, 'steps': steps}

    # verify candidate connection
    count = 0
    target = {}
    for n in cand.values():
        if not (n['type'] == 'Slice'):
            continue
        wi = n
        wl = []
        if not (wi[IL][0] in cand):
            continue
        if (len(cnct[wi[IL][0]]) > 1):
            continue
        starts = wi['starts']
        axes = wi['axes']
        steps = wi['steps']
        wl.append(wi)
        wi = cand[wi[IL][0]]
        if (not wi['type'] == 'Gather'):
            continue
        axis = wi['attr'].get('axis', 0)
        for idx, a in enumerate(axes):
            if a < axis:
                continue
            else:
                axes[idx] += 1
        gather_start = numpy.array([data[wi[IL][1]]['body']])
        gather_input = wi[IL][0]
        wl.append(wi)
        count += 1
        for wi in wl:
            if wi['type'] == 'Slice':
                wi[IL][0] = gather_input
                wi['starts'] = numpy.concatenate([gather_start, starts])
                wi['end'] = gather_start + 1
                wi['axes'] = numpy.concatenate([numpy.array([axis]), axes])
                wi['steps'] = numpy.concatenate([numpy.array([1]), steps])
            target[wi[OL]] = wi

    if count == 0:
        return model

    print("    [i] report : summarize {} Gather+Slice with \'onnx_summarize_gather_slice\'".format(count))

    mod_node = []
    opt_num = 0
    for n in model.graph.node:
        id = n.output[0]

        if not id in target:
            mod_node.append(n)
            continue

        if n.op_type == 'Gather':
            data[n.input[1]]['ref'] -= 1
            continue
        else:
            wi = target[id]
            # create concat node to summarize 'ends'.
            concat_input = 'opt.gather_ends.{}.{}'.format(wi[IL][2], opt_num)
            concat_output = 'opt.gather_out.{}.{}'.format(wi[IL][2], opt_num)
            data.setdefault(concat_input, {'ref': 1, 'id': concat_input, 'body': wi['end']})
            node = onnx.helper.make_node('Concat', [concat_input, wi[IL][2]], [concat_output], **{'axis': 0})
            mod_node.append(node)

            # modify slice
            n.input[0] = wi[IL][0]
            data[n.input[1]]['ref'] -= 1
            init_id = 'opt.' + n.input[1] + '.' + nmo
            n.input[1] = init_id
            if init_id in data:
                data[init_id]['ref'] += 1
            else:
                data[init_id] = {'ref': 1, 'id': init_id, 'body': wi['starts']}

            n.input[2] = concat_output
            data[n.input[3]]['ref'] -= 1
            init_id = 'opt.{}.{}.{}'.format(n.input[3], nmo, opt_num)
            n.input[3] = init_id
            data[init_id] = {'ref': 1, 'id': init_id, 'body': wi['axes']}
            data[n.input[4]]['ref'] -= 1
            init_id = 'opt.{}.{}.{}'.format(n.input[4], nmo, opt_num)
            n.input[4] = init_id
            data[init_id] = {'ref': 1, 'id': init_id, 'body': wi['steps']}
            slice_output = 'opt.' + id
            n.output[0] = slice_output
            mod_node.append(n)

            # add squeeze to shrink dimension
            if ver < 13:
                node = onnx.helper.make_node('Squeeze', [slice_output], [id], **{'axes': numpy.array([0])})
            else:
                squeeze_input = 'opt.{}.axes'.format(id)
                data[squeeze_input] = {'ref': 1, 'id': squeeze_input, 'body': numpy.array([0])}
                node = onnx.helper.make_node('Squeeze', [slice_output, squeeze_input], [id])
            mod_node.append(node)
            opt_num += 1

    mod_initializer, mod_input = pack_initializer(data, model.graph.input)
    m_graph = onnx.helper.make_graph(mod_node, model.graph.name, mod_input, model.graph.output, mod_initializer)
    m_model = onnx.helper.make_model(m_graph, producer_name=model.producer_name, producer_version=model.producer_version, opset_imports=model.opset_import)

    return m_model


def onnx_remove_needless_expand(model):
    data = listup_initializer(model)
    OL = 'o'
    IL = 'i'
    cand = {}
    cnct = {}
    for n in model.graph.node:
        for o in n.output:
            for i in n.input:
                cnct.setdefault(i, [])
                cnct[i].append(o)
        nmo = n.output[0]
        if (n.op_type == 'Shape'):
            if n.input[0] in data:
                continue
            cand[nmo] = {'type': n.op_type, OL: nmo, IL: n.input}
        elif (n.op_type == 'ConstantOfShape'):
            if not n.input[0] in cand:
                continue
            ut = cand[n.input[0]]['type']
            if not ut == 'Shape':
                continue
            cand[nmo] = {'type': n.op_type, OL: nmo, IL: n.input, 'attr': listup_attribute(n)}
        elif (n.op_type == 'Expand'):
            if not (n.input[0] in cand and n.input[1] in cand):
                continue
            ut = cand[n.input[0]]['type']
            if not ut == 'ConstantOfShape':
                continue
            ut = cand[n.input[1]]['type']
            if not ut == 'Shape':
                continue
            cand[nmo] = {'type': n.op_type, OL: nmo, IL: n.input}
    # verify candidate connection
    count = 0
    target = {}
    for n in cand.values():
        if not (n['type'] == 'Expand'):
            continue
        wi = n
        wl = []
        while (wi[IL][0] in cand):
            if (wi['type'] == 'Expand'):
                if (len(cnct[wi[IL][0]]) > 1):
                    break
                expand_shape_ipt = wi[IL][1]
                expand_opt = wi[OL]
            elif wi['type'] == 'ConstantOfShape':
                if not wi[IL][0] == expand_shape_ipt:
                    break
                wi[OL] = expand_opt
            wl.append(wi)
            wi = cand[wi[IL][0]]
        if not wi['type'] == 'Shape':
            continue
        wl.append(wi)
        count += 1
        for wi in wl:
            target[wi[OL]] = wi

    if count == 0:
        return model

    print("    [i] report : remove {} Expand from 'Shape+ConstantOfShape+Expand'".format(count))

    mod_node = []
    for n in model.graph.node:
        id = n.output[0]

        if not (id in target):
            mod_node.append(n)
            continue

        wi = target[id]
        if (wi['type'] == 'Expand'):
            continue
        elif (wi['type'] == 'ConstantOfShape'):
            nx = onnx.helper.make_node('ConstantOfShape', wi[IL], [wi[OL]], **wi['attr'])
            mod_node.append(nx)
        else:
            mod_node.append(n)

    mod_initializer, mod_input = pack_initializer(data, model.graph.input)
    m_graph = onnx.helper.make_graph(mod_node, model.graph.name, mod_input, model.graph.output, mod_initializer)
    m_model = onnx.helper.make_model(m_graph, producer_name=model.producer_name, producer_version=model.producer_version, opset_imports=model.opset_import)

    # return model
    return m_model


def onnx_remove_expand_into_eltwise(model):
    shapes = ShapeAndTypeInfo(model)
    OL = 'o'
    IL = 'i'
    # list up replace node candidate
    cand = {}  # candidate
    cnct = {}  # conection
    for n in model.graph.node:
        for o in n.output:
            for i in n.input:
                cnct.setdefault(i, [])
                cnct[i].append(o)
        nmo = n.output[0]  # node main output
        if is_2in_eltwise(n.op_type):
            if (n.input[0] == n.input[1]):
                continue
            if (n.input[0] in cand):
                ut = cand[n.input[0]]['type']
            elif (n.input[1] in cand):
                ut = cand[n.input[1]]['type']
            else:
                continue
            if not (ut == 'Expand'):
                continue
            cand[nmo] = {'type': n.op_type, OL: nmo, IL: n.input}
        elif (n.op_type == 'Expand'):
            cand[nmo] = {'type': 'Expand', OL: nmo, IL: n.input}
        elif (n.op_type == 'Shape'):
            cand[nmo] = {'type': 'Shape', OL: nmo, IL: n.input}

    # verify candidate connection
    count = 0
    target = {}
    for n in cand.values():
        if not is_2in_eltwise(n['type']):
            continue

        # check eltwise <- expand connection
        eltwise = n
        expand_out = None
        expand_in = None
        ei_left = eltwise[IL][0]
        ei_right = eltwise[IL][1]
        if (ei_left in cand) and (cand[ei_left]['type'] == 'Expand'):
            expand_out = ei_left
            eltwise_in = ei_right
        elif (ei_right in cand) and (cand[ei_right]['type'] == 'Expand'):
            expand_out = ei_right
            eltwise_in = ei_left
        else:
            continue

        # check nodes using expand output
        expand = cand[expand_out]
        use_expand = len(cnct[expand_out])
        for x in cnct[expand_out]:
            if (x in cand) and (cand[x]['type'] != 'Expand'):
                use_expand -= 1
        if use_expand != 0:
            # other node is using this expand
            continue
        expand_in = expand[IL][0]

        if shapes.is_inferred(ei_left) and shapes.is_inferred(ei_right):
            if shapes.get_shape(ei_left) != shapes.get_shape(ei_right):
                continue
        else:
            if not expand[IL][1] in cand:
                continue
            if cand[ expand[IL][1] ]['type'] != 'Shape':
                continue
            shape = cand[ expand[IL][1] ]
            if shape[IL][0] != eltwise_in:
                continue

        # append target
        if ei_left == expand_out:
            eltwise[IL] = [expand_in, ei_right]
        else:
            eltwise[IL] = [ei_left, expand_in]
        if not expand[OL] in target:
            count += 1
            target[ expand[OL] ] = expand
        target[ eltwise[OL] ] = eltwise

    if (count == 0):
        return model

    print("    [i] report : remove {} Expand nodes with \'onnx_remove_expand_into_eltwise\'".format(count))

    mod_node = []
    for n in model.graph.node:
        id = n.output[0]

        if not (id in target):
            mod_node.append(n)
            continue

        wi = target[id]
        if n.op_type == 'Expand':
            continue
        n.input[0] = wi[IL][0]
        n.input[1] = wi[IL][1]
        mod_node.append(n)

    m_graph = onnx.helper.make_graph(mod_node, model.graph.name, model.graph.input, model.graph.output, model.graph.initializer)
    m_model = onnx.helper.make_model(m_graph, producer_name=model.producer_name, producer_version=model.producer_version, opset_imports=model.opset_import)

    return m_model


def onnx_remove_needless_mul_expand(model):
    data = listup_initializer(model)

    OL = 'o'
    IL = 'i'
    cand = {}
    cnct = {}
    for n in model.graph.node:
        for o in n.output:
            for i in n.input:
                cnct.setdefault(i, [])
                cnct[i].append(o)
        nmo = n.output[0]
        if (n.op_type == 'Expand'):
            if not (n.input[0] in data):
                continue
            if data[n.input[0]]['body'].size != 1:
                continue
            if data[n.input[0]]['body'] != 1:
                continue
            cand[nmo] = {'type': n.op_type, OL: nmo, IL: n.input}
        elif (n.op_type == 'Reshape'):  # check to prevent unexpected changes
            if not (n.input[1] in data):
                continue
            cand[nmo] = {'type': n.op_type, OL: nmo, IL: n.input}
        elif (n.op_type == 'Mul'):
            if not (len(n.input) == 2):
                continue
            if not (n.input[0] in cand and n.input[1] in cand):
                continue
            ut1 = cand[n.input[0]]['type']
            ut2 = cand[n.input[1]]['type']
            if (ut1 == ut2):
                continue
            cand[nmo] = {'type': n.op_type, OL: nmo, IL: n.input}

    # verify candidate connection
    count = 0
    target = {}
    for n in cand.values():
        if not (n['type'] == 'Mul'):
            continue
        wi = n

        if (cand[wi[IL][0]]['type'] == 'Expand'):
            expand = cand[wi[IL][0]]
            wi[IL] = [wi[IL][1], expand[IL][1]]
        elif (cand[wi[IL][1]]['type'] == 'Expand'):
            expand = cand[wi[IL][1]]
            wi[IL] = [wi[IL][0], expand[IL][1]]
        else:
            continue

        target[expand[OL]] = expand
        target[wi[OL]] = wi
        count += 1

    if count == 0:
        return model

    print("    [i] report : summarize {} Expand+Mul with \'onnx_remove_expand_broadcast\'".format(count))

    mod_node = []
    for n in model.graph.node:
        id = n.output[0]

        if not id in target:
            mod_node.append(n)
            continue

        if n.op_type == 'Expand':
            if n.input[0] in data:
                data[n.input[0]]['ref'] -= 1
            continue

        wi = target[id]

        nx = onnx.helper.make_node('Expand', wi[IL], [wi[OL]])
        mod_node.append(nx)

    mod_initializer, mod_input = pack_initializer(data, model.graph.input)
    m_graph = onnx.helper.make_graph(mod_node, model.graph.name, mod_input, model.graph.output, mod_initializer)
    m_model = onnx.helper.make_model(m_graph, producer_name=model.producer_name, producer_version=model.producer_version, opset_imports=model.opset_import)

    return m_model


def onnx_remove_needless_cast_and_ceil(model):
    # list up initializer
    data = listup_initializer(model)

    OL = 'o'
    IL = 'i'
    # list up replace node candidate
    cand = {}  # candidate
    cnct = {}  # conection
    for n in model.graph.node:
        for o in n.output:
            for i in n.input:
                cnct.setdefault(i, [])
                cnct[i].append(o)
        nmo = n.output[0]  # node main output
        if n.op_type == 'Cast':
            if n.input[0] in data:
                continue
            attr = listup_attribute(n)
            cand[nmo] = {'name': n.name, 'type': n.op_type, OL: nmo, IL: n.input, 'attr': attr}
        elif n.op_type == 'Ceil' or n.op_type == 'Floor' or n.op_type == 'Round':
            if not (n.input[0] in cand):
                continue
            ut = cand[n.input[0]]['type']
            if not ut == 'Cast':
                continue
            cand[nmo] = {'name': n.name, 'type': n.op_type, OL: nmo, IL: n.input}

    # verify candidate connection
    count = 0
    target = {}
    for n in cand.values():
        if not n['type'] == 'Cast':
            continue
        wi = n
        wl = []
        while (wi[IL][0] in cand):
            if wi['type'] == 'Ceil' or wi['type'] == 'Floor' or wi['type'] == 'Round':
                wl.append(wi)
                wi = cand[wi[IL][0]]
            elif wi['type'] == 'Cast':
                if (len(cnct[wi[IL][0]]) > 1):
                    break
                wl.append(wi)
                rear_cast_to = return_elem_datatype_str(wi['attr']['to'])
                wi = cand[wi[IL][0]]
        if not wl or not wi['type'] == 'Cast':
            continue
        front_cast_to = return_elem_datatype_str(wi['attr']['to'])

        # check if ceil can remove
        if not (('FLOAT' in front_cast_to or 'DOUBLE' in front_cast_to) and ('INT' in rear_cast_to)):
            continue
        # check if front-cast can remove
        if (len(cnct[wi[OL]]) == 1):
            wi['remove'] = True
            wl.append(wi)
        wl[0][IL] = wl[-1][IL]
        for wi in wl:
            count += 1
            target[wi[OL]] = wi

    if (count == 0):
        # no removable transpose
        return model

    count = 0
    mod_node = []
    for n in model.graph.node:
        id = n.output[0]
        if (not id in target):
            # keep node
            mod_node.append(n)
            continue

        wi = target[id]
        if wi['type'] == 'Ceil' or wi['type'] == 'Floor' or wi['type'] == 'Round':
            count += 1
            continue
        if wi['type'] == 'Cast' and 'remove' in wi:
            count += 1
            continue

        nx = onnx.helper.make_node('Cast', wi[IL], [wi[OL]], **wi['attr'], name=wi['name'])
        mod_node.append(nx)

    print("    [i] report : remove {} Cast->Ceil nodes with \'onnx_remove_needless_cast_ceil\'".format(count))
    mod_initializer, mod_input = pack_initializer(data, model.graph.input)
    m_graph = onnx.helper.make_graph(mod_node, model.graph.name, mod_input, model.graph.output, mod_initializer)
    m_model = onnx.helper.make_model(m_graph, producer_name=model.producer_name, producer_version=model.producer_version, opset_imports=model.opset_import)

    return m_model


def onnx_remove_needless_slice(model):
    # list up initializer
    data = listup_initializer(model)
    # shape inference
    info = ShapeAndTypeInfo(model)

    remove_out = []
    replace = {}
    for n in model.graph.node:
        id = n.output[0]  # node main output
        if n.op_type == 'Slice':
            if not info.is_inferred(n.input[0]):
                continue
            data_shape = info.get_shape(n.input[0])
            start_id = n.input[1]
            end_id = n.input[2]
            axes_id = n.input[3] if len(n.input) >= 4 else None
            step_id = n.input[4] if len(n.input) >= 5 else None
            if not start_id in data:
                continue
            if not end_id in data:
                continue
            starts = data[start_id]['body']
            ends = data[end_id]['body']
            if len(n.input) < 4:
                axes = numpy.arange(len(starts), dtype=numpy.int64)
            else:
                if axes_id in data:
                    axes = data[axes_id]['body']
                else:
                    continue
            if len(n.input) < 5:
                steps = numpy.ones((len(starts),), dtype=numpy.int64)
            else:
                if step_id in data:
                    steps = data[step_id]['body']
                else:
                    continue

            removable = True
            for i, k in enumerate(axes):
                if steps[i] != 1 or starts[i] != 0 or ends[i] < data_shape[k]:
                    removable = False
                    break
            if removable:
                replace[n.output[0]] = n.input[0]
                remove_out.append(n.output[0])

    count = 0
    mod_node = []
    for n in model.graph.node:
        id = n.output[0]
        if id in remove_out:
            count += 1
            continue
        for i, ipt in enumerate(n.input):
            if ipt in replace:
                n.input[i] = replace[ipt]
        mod_node.append(n)

    if count == 0:
        return model

    # adjust blob name related graph output
    adjust_graph_output(mod_node, replace, model)

    print("    [i] report : remove {} slice nodes with \'onnx_remove_needless_slice\'.".format(count))

    mod_initializer, mod_input = pack_initializer(data, model.graph.input)
    m_graph = onnx.helper.make_graph(mod_node, model.graph.name, mod_input, model.graph.output, mod_initializer)
    m_model = onnx.helper.make_model(m_graph, producer_name=model.producer_name, producer_version=model.producer_version, opset_imports=model.opset_import)

    return m_model


def onnx_optimize(onnx_path, yolov3_mode, manual, keep_err_onnx=False):

    # show information
    out_dir = os.path.dirname(onnx_path)
    out_base, out_ext = os.path.splitext(os.path.basename(onnx_path))
    out_path = out_base + ".opt" + out_ext
    if out_dir != '':
        out_path = out_dir + os.path.sep + out_path
    print("+ creating " + out_path)
    print("    from " + onnx_path + " ...")

    # load model
    model = onnx.load(onnx_path)

    # topologically sort
    model = onnx_topologically_sort(model)

    # check opset version
    support_ver = (10, 11)
    opset_ver = model.opset_import[0].version
    if opset_ver in support_ver:
        print("    * info : supported model opset_version=({}) ".format(opset_ver))
    else:
        print("    * warning : unsupported model opset_version=({}) ".format(opset_ver))
        print("                Supported opset_version=(" + ", ".join(map(str, support_ver)) + ')')
        print("      continuing optimize process..")

    # extract constant to initializer
    model = onnx_extract_constant_to_initializer(model)

    # manual optimization mode if specified.
    if manual != None:
        model = onnx_manual_optimization(model, manual[0])

    # remove expand into eltwise [ ex: Add(Expand(a), b) -> Add(a, b) ]
    model = onnx_remove_expand_into_eltwise(model)

    # reduce needless node
    model = onnx_reduce_needless_node(model)

    # remove duplicated node
    model = onnx_remove_duplicated_node(model)

    # replace node that generate the constant value to initializer
    model = onnx_convert_node_to_const(model)

    # remove needless cast
    model = onnx_remove_needless_cast(model)

    # replace to meanvarnorm (for transformer based model)
    model = onnx_replace_to_meanvarnorm(model)

    if yolov3_mode:
        # yolov3 : optimization failed / do loop replacement + add input (NonMaxSuppression threshold&iou)
        model = yolov3_special_treatment(model)

    # reduce needless squeeze/unsqueeze
    model = onnx_reduce_squeeze_unsqueeze(model)

    # convert add to mul
    model = onnx_convert_sum_to_scale(model)

    # remove eltwise that do nothing ( add(x+0), sub(x-0), mul(x*1), etc.. )
    model = onnx_reduce_needless_arithmetic(model)

    # replace keras sequence to reducel2
    model = onnx_replace_to_reducel2(model)

    # replace shape+reduceProd into Size
    model = onnx_replace_to_size(model)

    # convert Pow(x,2)->Mul(x,x) and Div(x,const)->Mul(x,1/const)
    model = onnx_convert_pow_div_to_mul(model)

    model = onnx_remove_needless_cast_and_ceil(model)

    model = onnx_summarize_seriese_slice(model)

    model = onnx_replace_seriese_max_min(model)

    model = onnx_remove_needless_expand(model)

    model = onnx_remove_needless_slice(model)

    model = onnx_move_transpose(model)
    model = onnx_summarize_transpose(model)

    # replace transpose to reshape
    model = onnx_replace_transpose_to_reshape(model)

    # replace expand to reshape
    model = onnx_replace_expand_to_reshape(model)

    # remove consecutive reshapes leaving rearmost reshape
    model = onnx_remove_needless_reshape(model)

    # replace StyleGAN Specific sequence to Reshape->ConvTranspose
    model = onnx_treat_stylegan(model)

    # fuse pad into conv
    model = onnx_fuse_pad_into_conv(model)

    # fuse mul+add to batchnorm
    model = onnx_convert_to_batchnorm(model)

    # fuse bias(add or sub) into conv(bias)
    model = onnx_fuse_bias_into_conv(model)

    # fuse mul into conv(bias)/convtranspose(bias)
    model = onnx_fuse_mul_into_conv(model)

    # fuse batchnorm into conv(bias)
    model = onnx_fuse_bn_into_conv(model)

    # replace Random*Like into Rndom
    model = onnx_replace_randomlike(model)

    # remove node which have no output
    model = onnx_remove_node_no_output(model)

    # replace reshape+lpnorm+reshape to lpnorm
    model = onnx_replace_to_lpnorm(model)

    # summarize gather+slice to multiple axes slice
    model = onnx_summarize_gather_slice(model)

    # remove needless mul+expand which use broadcast
    model = onnx_remove_needless_mul_expand(model)

    # save result
    with open(out_path, "wb") as f:
        f.write(model.SerializeToString())

    # postcheck
    try:
        onnx.checker.check_model(out_path)
    except Exception as e:
        if not keep_err_onnx:
            os.remove(out_path)
        raise


def main():
    parser = argparse.ArgumentParser(description='ONNX optimizer')
    parser.add_argument('files', nargs='+', help='optimize target file.')
    parser.add_argument('-m', '--manual', action='store', nargs=1, default=None, metavar='manual_opt.json', type=pathlib.Path, help='run manual optimization mode with json file.\n')
    parser.add_argument('-f', '--force', action='store_true', default=False, help='keep opt.onnx when verification fails.\n')
    parser.add_argument('--yolov3', action='store_true', help='run YOLOv3(from keras) model special mode.\n')
    args = parser.parse_args()

    for file in args.files:
        onnx_optimize(file, args.yolov3, args.manual, args.force)


if __name__ == "__main__":
    main()
