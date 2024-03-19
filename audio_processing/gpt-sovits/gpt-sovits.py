import sys
sys.path.append('./GPT_SoVITS')

import torch
import torchaudio
from torch import nn
import onnxruntime
import LangSegment

import os
from text import cleaned_text_to_sequence
import soundfile
from my_utils import load_audio
import os
import json

def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    hann_window = torch.hann_window(win_size).to(
            dtype=y.dtype, device=y.device
        )
    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec


class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")




class T2SModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hz = 50
        self.max_sec = 54
        self.top_k = 5
        self.early_stop_num = torch.LongTensor([self.hz * self.max_sec])

    def forward(self, ref_seq, text_seq, ref_bert, text_bert, ssl_content):
        early_stop_num = self.early_stop_num

        EOS = 1024

        sess_encoder = onnxruntime.InferenceSession(f"nahida_t2s_encoder.onnx", providers=["CPUExecutionProvider"])
        sess_fsdec = onnxruntime.InferenceSession(f"nahida_t2s_fsdec.onnx", providers=["CPUExecutionProvider"])
        sess_sdec = onnxruntime.InferenceSession(f"nahida_t2s_sdec.onnx", providers=["CPUExecutionProvider"])

        #[1,N] [1,N] [N, 1024] [N, 1024] [1, 768, N]
        x, prompts = sess_encoder.run(None, {"ref_seq":ref_seq.detach().numpy(), "text_seq":text_seq.detach().numpy(), "ref_bert":ref_bert.detach().numpy(), "text_bert":text_bert.detach().numpy(), "ssl_content":ssl_content.detach().numpy()})
        x = torch.from_numpy(x)
        prompts = torch.from_numpy(prompts)

        prefix_len = prompts.shape[1]

        #[1,N,512] [1,N]
        y, k, v, y_emb, x_example = sess_fsdec.run(None, {"x":x.detach().numpy(), "prompts":prompts.detach().numpy()})
        y = torch.from_numpy(y)
        k = torch.from_numpy(k)
        v = torch.from_numpy(v)
        y_emb = torch.from_numpy(y_emb)
        x_example = torch.from_numpy(x_example)

        stop = False
        for idx in range(1, 1500):
            #[1, N] [N_layer, N, 1, 512] [N_layer, N, 1, 512] [1, N, 512] [1] [1, N, 512] [1, N]
            y, k, v, y_emb, logits, samples = sess_sdec.run(None, {"iy":y.detach().numpy(), "ik":k.detach().numpy(), "iv":v.detach().numpy(), "iy_emb":y_emb.detach().numpy(), "ix_example":x_example.detach().numpy()})
            y = torch.from_numpy(y)
            k = torch.from_numpy(k)
            v = torch.from_numpy(v)
            y_emb = torch.from_numpy(y_emb)
            logits = torch.from_numpy(logits)
            samples = torch.from_numpy(samples)
            if early_stop_num != -1 and (y.shape[1] - prefix_len) > early_stop_num:
                stop = True
            if torch.argmax(logits, dim=-1)[0] == EOS or samples[0, 0] == EOS:
                stop = True
            if stop:
                break
        y[0, -1] = 0

        return y[:, -idx:].unsqueeze(0)





class GptSoVits(nn.Module):
    def __init__(self, t2s):
        super().__init__()
        self.t2s = t2s
    
    def forward(self, ref_seq, text_seq, ref_bert, text_bert, ref_audio, ssl_content):
        pred_semantic = self.t2s(ref_seq, text_seq, ref_bert, text_bert, ssl_content)
        sess = onnxruntime.InferenceSession("nahida_vits.onnx", providers=["CPUExecutionProvider"])
        audio1 = sess.run(None, {
            "text_seq" : text_seq.detach().cpu().numpy(),
            "pred_semantic" : pred_semantic.detach().cpu().numpy(), 
            "ref_audio" : ref_audio.detach().cpu().numpy()
        })
        return torch.from_numpy(audio1[0])



class SSLModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, ref_audio_16k):
        sess = onnxruntime.InferenceSession("nahida_cnhubert.onnx", providers=["CPUExecutionProvider"])
        last_hidden_state = sess.run(None, {
            "ref_audio_16k" : ref_audio_16k.detach().cpu().numpy()
        })
        return torch.from_numpy(last_hidden_state[0])


from text import cleaned_text_to_sequence
from text.cleaner import clean_text

def get_bert_feature(text, word2ph):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = bert_model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
    assert len(word2ph) == len(text)
    phone_level_feature = []
    for i in range(len(word2ph)):
        repeat_feature = res[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    return phone_level_feature.T

def clean_text_inf(text, language):
    phones, word2ph, norm_text = clean_text(text, language)
    print(phones)
    phones = cleaned_text_to_sequence(phones)
    return phones, word2ph, norm_text

def get_bert_inf(phones, word2ph, norm_text, language):
    language=language.replace("all_","")
    if language == "zh":
        bert = get_bert_feature(norm_text, word2ph)
    else:
        bert = torch.zeros(
            (1024, len(phones)),
            dtype=torch.float32,
        )

    return bert

def get_phones_and_bert(text,language):
    if language in {"en","all_zh","all_ja"}:
        language = language.replace("all_","")
        if language == "en":
            LangSegment.setfilters(["en"])
            formattext = " ".join(tmp["text"] for tmp in LangSegment.getTexts(text))
        else:
            # 因无法区别中日文汉字,以用户输入为准
            formattext = text
        while "  " in formattext:
            formattext = formattext.replace("  ", " ")
        phones, word2ph, norm_text = clean_text_inf(formattext, language)
        if language == "zh":
            bert = get_bert_feature(norm_text, word2ph)
        else:
            bert = torch.zeros(
                (1024, len(phones)),
                dtype=torch.float32,
            )
    elif language in {"zh", "ja","auto"}:
        textlist=[]
        langlist=[]
        LangSegment.setfilters(["zh","ja","en"])
        if language == "auto":
            for tmp in LangSegment.getTexts(text):
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        else:
            for tmp in LangSegment.getTexts(text):
                if tmp["lang"] == "en":
                    langlist.append(tmp["lang"])
                else:
                    # 因无法区别中日文汉字,以用户输入为准
                    langlist.append(language)
                textlist.append(tmp["text"])
        print(textlist)
        print(langlist)
        phones_list = []
        bert_list = []
        norm_text_list = []
        for i in range(len(textlist)):
            lang = langlist[i]
            phones, word2ph, norm_text = clean_text_inf(textlist[i], lang)
            bert = get_bert_inf(phones, word2ph, norm_text, lang)
            phones_list.append(phones)
            norm_text_list.append(norm_text)
            bert_list.append(bert)
        bert = torch.cat(bert_list, dim=1)
        phones = sum(phones_list, [])
        norm_text = ''.join(norm_text_list)

    return phones,bert.to(dtype = torch.float32),norm_text

def inference():
    gpt = T2SModel()
    gpt_sovits = GptSoVits(gpt)
    ssl = SSLModel()

    ref_audio = torch.randn((1, 48000 * 5)).float()

    ref_audio = torch.tensor([load_audio("JSUT.wav", 48000)]).float()
    ref_seq = torch.LongTensor([cleaned_text_to_sequence(['m', 'i', 'z', 'u', 'o', 'm', 'a', 'r', 'e', 'e', 'sh', 'i', 'a', 'k', 'a', 'r', 'a', 'k', 'a', 'w', 'a', 'n', 'a', 'k', 'U', 't', 'e', 'w', 'a', 'n', 'a', 'r', 'a', 'n', 'a', 'i', '.'])])

    #ref_audio = torch.tensor([load_audio("kyakuno.wav", 48000)]).float()
    #ref_seq = torch.LongTensor([cleaned_text_to_sequence(['a', 'a', 'r', 'u', 'b', 'u', 'i', 'sh', 'i', 'i', 'o', 'sh', 'i', 'y', 'o', 'o', 'sh', 'I', 't', 'a', 'b', 'o', 'i', 's', 'U', 'ch', 'e', 'N', 'j', 'a', 'a', 'o', 'ts', 'U', 'k', 'u', 'r', 'u', '.'])])

    #phones1,bert1,norm_text1=get_phones_and_bert("RVCを使用したボイスチェンジャーを作る。", "all_ja")
    #print(phones1)

    #水をマレーシアから買わなくてはならない。

    #phones1,bert1,norm_text1=get_phones_and_bert("今日は晴れでしょうか。", "all_ja")
    #print(phones1)

    #text_seq = torch.LongTensor([cleaned_text_to_sequence(['m', 'i', 'z', 'u', 'w', 'a', ',', 'i', 'r', 'i', 'm', 'a', 's', 'e', 'N', 'k', 'a', '?'])])
    text_seq = torch.LongTensor([cleaned_text_to_sequence(['ky', 'o', 'o', 'w', 'a', 'h', 'a', 'r', 'e', 'd', 'e', 'sh', 'o', 'o', 'k', 'a', '?'])])
    ref_bert = torch.randn((ref_seq.shape[1], 1024)).float()
    text_bert = torch.randn((text_seq.shape[1], 1024)).float()

    ref_audio_16k = torchaudio.functional.resample(ref_audio,48000,16000).float()
    vits_hps_data_sampling_rat = 32000
    ref_audio_sr = torchaudio.functional.resample(ref_audio,48000,vits_hps_data_sampling_rat).float()

    ssl_content = ssl(ref_audio_16k).float()
    
    a = gpt_sovits(ref_seq, text_seq, ref_bert, text_bert, ref_audio_sr, ssl_content)
    soundfile.write("out.wav", a.cpu().detach().numpy(), vits_hps_data_sampling_rat)

if __name__ == "__main__":
    inference()
    