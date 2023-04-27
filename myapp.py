import io
import os

# os.system("wget -P cvec/ https://huggingface.co/spaces/innnky/nanami/resolve/main/checkpoint_best_legacy_500.pt")
import gradio as gr
import gradio.processing_utils as gr_pu
import librosa
import numpy as np
import soundfile
from inference.infer_tool import Svc
import logging
import re
import json

import subprocess
import edge_tts
import asyncio
from scipy.io import wavfile
import librosa
import torch
import time
import traceback
from itertools import chain
from utils import mix_model

logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('markdown_it').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('multipart').setLevel(logging.WARNING)

model = None
spk = None
debug = False

cuda = {}
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        device_name = torch.cuda.get_device_properties(i).name
        cuda[f"CUDA:{i} {device_name}"] = f"cuda:{i}"


def vc_fn(sid, input_audio, vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale, pad_seconds, cl_num, lg_num,
          lgr_num, F0_mean_pooling, enhancer_adaptive_key, cr_threshold):
    global model
    try:
        if input_audio is None:
            raise gr.Error("你需要上传音频")
        if model is None:
            raise gr.Error("你需要指定模型")
        sampling_rate, audio = input_audio
        # print(audio.shape,sampling_rate)
        audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.transpose(1, 0))
        temp_path = "temp.wav"
        soundfile.write(temp_path, audio, sampling_rate, format="wav")
        _audio = model.slice_inference(temp_path, sid, vc_transform, slice_db, cluster_ratio, auto_f0, noise_scale,
                                       pad_seconds, cl_num, lg_num, lgr_num, F0_mean_pooling, enhancer_adaptive_key,
                                       cr_threshold)
        model.clear_empty()
        os.remove(temp_path)
        # 构建保存文件的路径，并保存到results文件夹内
        try:
            timestamp = str(int(time.time()))
            filename = sid + "_" + timestamp + ".wav"
            output_file = os.path.join("./results", filename)
            soundfile.write(output_file, _audio, model.target_sample, format="wav")
            return f"推理成功，音频文件保存为results/{filename}", (model.target_sample, _audio)
        except Exception as e:
            if debug: traceback.print_exc()
            return f"文件保存失败，请手动保存", (model.target_sample, _audio)
    except Exception as e:
        if debug: traceback.print_exc()
        raise gr.Error(e)


def tts_func(_text, _rate, _voice):
    # 使用edge-tts把文字转成音频
    # voice = "zh-CN-XiaoyiNeural"#女性，较高音
    # voice = "zh-CN-YunxiNeural"#男性
    # print(_text) # 测试
    voice = "zh-CN-YunxiNeural"  # 男性
    if (_voice == "女"): voice = "zh-CN-XiaoyiNeural"
    elif (_voice== "女(美英)"): voice = "en-US-MichelleNeural"
    # output_file = _text[0:10] + ".wav" 有些标点符号会导致名字出错
    timestamp = str(int(time.time()))
    output_file = timestamp + ".wav"
    
    # communicate = edge_tts.Communicate(_text, voice)
    # await communicate.save(output_file)
    if _rate >= 0:
        ratestr = "+{:.0%}".format(_rate)
    elif _rate < 0:
        ratestr = "{:.0%}".format(_rate)  # 减号自带

    p = subprocess.Popen("edge-tts " +
                         " --text " + _text +
                         " --write-media " + output_file +
                         " --voice " + voice +
                         " --rate=" + ratestr
                         , shell=True,
                         stdout=subprocess.PIPE,
                         stdin=subprocess.PIPE)
    p.wait()
    return output_file


def text_clear(text, voice):
    if voice == "女(美英)":
        # return '"'+re.sub(r"[\n\,\(\)]", "", text).replace(" ",",")+'"'
        return '"'+re.sub(r"[\n\\(\)]", "", text)+'"'
    else:
        return '"'+re.sub(r"[\n\,\(\) ]", "", text)+'"'


def vc_fn2(sid, input_audio, vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale, pad_seconds, cl_num, lg_num,
           lgr_num, text2tts, tts_rate, tts_voice, F0_mean_pooling, enhancer_adaptive_key, cr_threshold):
    # 使用edge-tts把文字转成音频
    text2tts = text_clear(text2tts, tts_voice)
    output_file = tts_func(text2tts, tts_rate, tts_voice)

    # 调整采样率
    sr2 = 44100
    wav, sr = librosa.load(output_file)
    wav2 = librosa.resample(wav, orig_sr=sr, target_sr=sr2)
    # save_path2 = text2tts[0:10] + "_44k" + ".wav"
    save_path2 = output_file + "_44k" + ".wav"
    wavfile.write(save_path2, sr2,
                  (wav2 * np.iinfo(np.int16).max).astype(np.int16)
                  )

    # 读取音频
    sample_rate, data = gr_pu.audio_from_file(save_path2)
    vc_input = (sample_rate, data)

    a, b = vc_fn(sid, vc_input, vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale, pad_seconds, cl_num,
                 lg_num, lgr_num, F0_mean_pooling, enhancer_adaptive_key, cr_threshold)
    os.remove(output_file)
    os.remove(save_path2)
    return a, b


def debug_change():
    global debug
    debug = debug_button.value

def modelUnload():
    global model
    if model is None:
        return sid.update(choices = [],value=""),"没有模型需要卸载!"
    else:
        model.unload_model()
        model = None
        torch.cuda.empty_cache()
        return sid.update(choices = [],value=""),"模型卸载完毕!"


def sid_change(sid,device="Auto"):
    global model
    # 清空模型
    modelUnload()
    # speakSetting.json文件中读取配置
    with open('custom_configs/speakSetting.json', 'r') as f:
        speakSetting = json.load(f)

    try:
        print(sid)
        device = cuda[device] if "CUDA" in device else device
        model_path = speakSetting["spk"][sid]["model"]
        config_path = speakSetting["spk"][sid]["config"]
        cluster_path = speakSetting["spk"][sid]["cluster_model"]
        model = Svc(model_path, config_path, device=device if device != "Auto" else None,
                    cluster_model_path=cluster_path,
                    nsf_hifigan_enhance=False)
        device_name = torch.cuda.get_device_properties(model.dev).name if "cuda" in str(model.dev) else str(model.dev)
        msg = f"成功加载模型到设备{device_name}上\n"
        return msg
    except Exception as e:
        if debug: traceback.print_exc()
        raise gr.Error(e)


with gr.Blocks(
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.green,
            font=["Source Sans Pro", "Arial", "sans-serif"],
            font_mono=['JetBrains mono', "Consolas", 'Courier New']
        ),
) as app:
    with gr.Tabs():
        with gr.TabItem("推理"):
            gr.Markdown(value="""
                So-vits-svc 4.0 推理 webui
                """)
            with gr.Row(variant="panel"):
                with gr.Column():
                    gr.Markdown(value="""
                        <font size=3>若音色中无选项或无法加载，请确定speakSetting.json是否设置正确：</font>
                        """)
                    # speakSetting.json文件中读取配置
                    with open("custom_configs/speakSetting.json", "r") as f:
                        speakSetting = json.load(f)

                    # 获取spk下的所有元素名
                    spk_names = list(speakSetting["spk"].keys())

                    sid = gr.Dropdown(label="音色", choices = spk_names)
                    sid_output = gr.Textbox(label="Output Message")

            with gr.Row(variant="panel"):
                with gr.Column():
                    gr.Markdown(value="""
                        <font size=2> 推理设置</font>
                        """)
                    auto_f0 = gr.Checkbox(
                        label="自动f0预测，配合聚类模型f0预测效果更好,会导致变调功能失效（仅限转换语音，歌声勾选此项会究极跑调）",
                        value=False)
                    F0_mean_pooling = gr.Checkbox(
                        label="是否对F0使用均值滤波器(池化)，对部分哑音有改善。注意，启动该选项会导致推理速度下降，默认关闭",
                        value=False)
                    vc_transform = gr.Number(label="变调（整数，可以正负，半音数量，升高八度就是12）", value=0)
                    cluster_ratio = gr.Number(
                        label="聚类模型混合比例，0-1之间，0即不启用聚类。使用聚类模型能提升音色相似度，但会导致咬字下降（如果使用建议0.5左右）",
                        value=0)
                    slice_db = gr.Number(label="切片阈值", value=-40)
                    noise_scale = gr.Number(label="noise_scale 建议不要动，会影响音质，玄学参数", value=0.4)
                with gr.Column():
                    pad_seconds = gr.Number(
                        label="推理音频pad秒数，由于未知原因开头结尾会有异响，pad一小段静音段后就不会出现", value=0.5)
                    cl_num = gr.Number(label="音频自动切片，0为不切片，单位为秒(s)", value=0)
                    lg_num = gr.Number(
                        label="两端音频切片的交叉淡入长度，如果自动切片后出现人声不连贯可调整该数值，如果连贯建议采用默认值0，注意，该设置会影响推理速度，单位为秒/s",
                        value=0)
                    lgr_num = gr.Number(
                        label="自动音频切片后，需要舍弃每段切片的头尾。该参数设置交叉长度保留的比例，范围0-1,左开右闭",
                        value=0.75)
                    enhancer_adaptive_key = gr.Number(label="使增强器适应更高的音域(单位为半音数)|默认为0", value=0)
                    cr_threshold = gr.Number(
                        label="F0过滤阈值，只有启动f0_mean_pooling时有效. 数值范围从0-1. 降低该值可减少跑调概率，但会增加哑音",
                        value=0.05)
            with gr.Tabs():
                with gr.TabItem("音频转音频"):
                    vc_input3 = gr.Audio(label="选择音频")
                    vc_submit = gr.Button("音频转换", variant="primary")
                with gr.TabItem("文字转音频"):
                    text2tts = gr.Textbox(label="在此输入要转译的文字。注意，使用该功能建议打开F0预测，不然会很怪。")
                    tts_rate = gr.Number(label="tts语速", value=0)
                    tts_voice = gr.Radio(label="语类", choices=["男", "女", "女(美英)"], value="男")
                    vc_submit2 = gr.Button("文字转换", variant="primary")
            with gr.Row():
                with gr.Column():
                    vc_output1 = gr.Textbox(label="Output Message")
                with gr.Column():
                    vc_output2 = gr.Audio(label="Output Audio", interactive=False)

        

    with gr.Tabs():
        with gr.Row(variant="panel"):
            with gr.Column():
                gr.Markdown(value="""
                    <font size=2> WebUI设置</font>
                    """)
                debug_button = gr.Checkbox(label="Debug模式，如果向社区反馈BUG需要打开，打开后控制台可以显示具体错误提示",
                                           value=debug)
        vc_submit.click(vc_fn,
                        [sid, vc_input3, vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale, pad_seconds,
                         cl_num, lg_num, lgr_num, F0_mean_pooling, enhancer_adaptive_key, cr_threshold],
                        [vc_output1, vc_output2])
        vc_submit2.click(vc_fn2,
                         [sid, vc_input3, vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale, pad_seconds,
                          cl_num, lg_num, lgr_num, text2tts, tts_rate, tts_voice, F0_mean_pooling,
                          enhancer_adaptive_key, cr_threshold], [vc_output1, vc_output2])
        debug_button.change(debug_change, [], [])
        sid.change(sid_change, [sid], [sid_output])
    # 启动gradio，设置server_name为0.0.0.0以便局域网内其他设备可以通过ip:7860访问。
    app.launch(server_name='0.0.0.0')
