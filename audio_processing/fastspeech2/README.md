Input 

python3 fastspeech2.py 

追記:11/27
PRする際に指定された形にはできていません。最終的には形式を整えて提出します。　林

exportは以下のようにして行いました

```python
    model.eval()
    model.requires_grad_(False)

    wrapped_model = FastSpeech2Wrapper(model)

    # 4. ダミー入力の作成 (汎用設定)
    # どんなモデルでも動作するように、標準的なサイズを設定
    batch_size = 1
    text_len = 600 # Dynamic Axesを使うので、ここの長さは適当でOK
    
    # 重要: マルチ話者対応
    # Configから最大話者数を取得し、範囲内のIDを指定する
    # LJSpeechなどは n_speaker が設定ファイルにない場合もあるのでケアする
    n_speakers = model_config.get("n_speakers", 0)
    speaker_id = 0 # デフォルト
    
    print(f"Model Info: Multi-Speaker={n_speakers > 1} (n={n_speakers})")

    speakers = torch.tensor([speaker_id], dtype=torch.long, device=device)
    
    # 語彙サイズチェック (エラー回避)
    # 多くのモデルで記号含め数百程度なので、100以下なら安全
    texts = torch.randint(1, 100, (batch_size, text_len), dtype=torch.long, device=device)
    
    src_lens = torch.tensor([text_len], dtype=torch.long, device=device)
    max_src_len = torch.tensor([text_len], dtype=torch.long, device=device)
    
    # コントロール値
    p_control = torch.tensor(1.0, dtype=torch.float32, device=device)
    e_control = torch.tensor(1.0, dtype=torch.float32, device=device)
    d_control = torch.tensor(1.0, dtype=torch.float32, device=device)

    # 5. エクスポート設定 (すべてに対応するユニバーサルな設定)
    input_names = ["speakers", "texts", "src_lens", "max_src_len", "p_control", "e_control", "d_control"]
    
    output_names = [
        "output", 
        "postnet_output", 
        "p_predictions", 
        "e_predictions", 
        "log_d_predictions", 
        "d_rounded", 
        "src_masks", 
        "mel_masks", 
        "src_lens_out", 
        "mel_lens_out"
    ]
    
    dynamic_axes = {
        "texts": {1: "text_length"},
        "output": {1: "mel_length"},
        "postnet_output": {1: "mel_length"},
        "p_predictions": {1: "mel_length"},
        "e_predictions": {1: "mel_length"},
        "log_d_predictions": {1: "text_length"},
    }

    print(f"Exporting to {args.output}...")
    
    torch.onnx.export(
        wrapped_model,
        (speakers, texts, src_lens, max_src_len, p_control, e_control, d_control),
        args.output,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=11, # 安定版
        do_constant_folding=False,
        keep_initializers_as_inputs=True,
        verbose=False
    )
```