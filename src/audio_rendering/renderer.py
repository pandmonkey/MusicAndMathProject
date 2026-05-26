"""MIDI → WAV 音频渲染

使用 FluidSynth + SoundFont 将 MIDI 文件渲染为 WAV 音频。
也提供基于 pretty_midi 的简易合成备选方案。
"""

from pathlib import Path
import numpy as np
import soundfile as sf

from src.config import RenderConfig


def render_with_fluidsynth(
    midi_path: Path,
    output_path: Path,
    config: RenderConfig,
) -> Path:
    """使用 FluidSynth 渲染 MIDI 为 WAV。

    Args:
        midi_path: 输入 MIDI 文件路径
        output_path: 输出 WAV 文件路径
        config: 渲染配置

    Returns:
        写入的 WAV 文件路径
    """
    import fluidsynth

    fs = fluidsynth.Synth(samplerate=float(config.sample_rate))
    sfid = fs.sfload(config.soundfont_path)
    fs.program_select(0, sfid, 0, 0)

    # 使用 pretty_midi 解析 MIDI 并逐音符合成
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    audio = pm.fluidsynth(
        fs=config.sample_rate,
        sf2_path=config.soundfont_path,
    )

    # 归一化避免裁切
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio)) * 0.9

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio, config.sample_rate)

    fs.delete()
    return output_path


def render_with_pretty_midi(
    midi_path: Path,
    output_path: Path,
    config: RenderConfig,
) -> Path:
    """使用 pretty_midi 内置合成器渲染（备选方案，无需 SoundFont）。"""
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(str(midi_path))
    audio = pm.synthesize(fs=config.sample_rate)

    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio)) * 0.9

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio, config.sample_rate)

    return output_path
