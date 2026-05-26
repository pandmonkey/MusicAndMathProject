"""MIDI 文件写入

将 Note 序列写入标准 MIDI 文件。
"""

from pathlib import Path
from midiutil import MIDIFile

from src.note import Note
from src.config import MelodyConfig


def write_midi(
    notes: list[Note],
    config: MelodyConfig,
    output_path: Path,
) -> Path:
    """将音符序列写入 MIDI 文件。

    Args:
        notes: 音符列表
        config: 旋律配置
        output_path: 输出路径 (.mid)

    Returns:
        写入的文件路径
    """
    midi = MIDIFile(1)  # 单轨道
    track = 0
    channel = 0

    midi.addTempo(track, 0, config.bpm)
    midi.addTimeSignature(
        track, 0,
        config.time_signature[0],
        int.bit_length(config.time_signature[1]) - 1,  # denominator power
        24, 8,
    )
    midi.addProgramChange(track, channel, 0, config.midi_program)

    for note in notes:
        midi.addNote(
            track=track,
            channel=channel,
            pitch=note.pitch,
            time=note.start_time,
            duration=note.duration,
            volume=note.velocity,
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        midi.writeFile(f)

    return output_path
