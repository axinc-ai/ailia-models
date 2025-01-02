CLASSES = [
            '人名', '法人名', '政治的組織名', 'その他の組織名',
            '地名', '施設名', '製品名', 'イベント名'
]

def get_aligned_span(s, input_text, start):
    s_start = input_text.find(s, start)
    if s_start != -1:
        return [s_start, s_start + len(s)]
    else:
        return -1


def entities_to_dicts(entities, input_text, tokenizer):
    # to account for <unk> and duplicated token ids
    #input_ids = tokenizer.encode(input_text)
    input_tokens = tokenizer.tokenize(input_text)
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    input_tokens = [i.replace('▁', ' ') for i in input_tokens]

    o = []

    start = 0
    prev = []
    
    unk_start = 0
    for e in entities:
#        if 5 not in e:# when the entity prediction does not match the format "(text) (type)"
#            continue
        e = tokenizer.decode(e)
        if ' ' not in e:
            continue

        # the type will not contain any spaces but the text can
        e = e[::-1]
        pred = [e[e.index(' ')+1:][::-1], e[:e.index(' ')][::-1]]

        if '<unk>' in pred[0]:# <unk> in pred[0]
            dec = pred[0].split('<unk>')
            for i in range(len(dec)-1):
                while 2 in input_ids[unk_start:]:
                    unk_idx = input_ids.index(2, unk_start)
                    if (''.join(input_tokens[unk_start:unk_idx]).endswith(dec[i]) and
                        ''.join(input_tokens[unk_idx+1:]).startswith(dec[i+1])):
                        dec[i] += input_tokens[unk_idx]
                        break
                    unk_start = unk_idx+1
            pred[0] = ''.join(dec)
        else:
            pred[0] = pred[0]

        if pred[0] == prev:# in case the same entity appears consecutively
            start += 1

        span = get_aligned_span(pred[0], input_text, start)

        # when pred[0] cannot be found in input_text
        if span == -1:
            continue
        
        if pred[1] not in CLASSES:
            for c in CLASSES:# to save errors like '政治的組織名は', '地名が'
                if c in pred[1]:
                    pred[1] = c
                    break
            else:
                continue

        o.append({
            'span': span,
            'type': pred[1],
            'text': pred[0]
        })
        start = span[0]# since predicted entities are trained to be sorted by its span (start, end)
        prev = pred[0]
    return o

def pred_dicts_to_iob(pred_dicts, input_text):
    iob_list = ['O'] * len(input_text)
    for pred in pred_dicts:
        iob_list[pred['span'][0]] = 'B-' + pred['type']
        iob_list[pred['span'][0]+1: pred['span'][1]] = ['I-' + pred['type']] * (pred['span'][1] - pred['span'][0] - 1)
    return iob_list


def decode(output, input_text, tokenizer, style='dict'):
    # all entries are supposed to be sandwiched between <extra_id_{i}> and <extra_id_{i+1}>
    entities = []
    if 32099 in output:
        for i in range(32098, 31999, -1):# reversed because token number of extra_id_{i+1} is smaller than extra_id_{i}
            if i in output:
                start = output.index(i+1)+1
                if output[start] == 5:# space token could inserted after the extra_id token when encoding the target sentence during training
                    entity = output[start+1: output.index(i)]
                else:
                    entity = output[start: output.index(i)]
                entities.append(entity)
            else:
                break
    
    pred_dicts = entities_to_dicts(entities, input_text, tokenizer)
    if style=='dict':
         return pred_dicts
    else:
        return pred_dicts_to_iob(pred_dicts, input_text)