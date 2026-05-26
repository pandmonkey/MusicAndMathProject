"""音符数据结构"""

from dataclasses import dataclass


@dataclass
class Note:
    """单个音符"""
    pitch: int          # MIDI 音高 (0-127)
    duration: float     # 时值（四分音符=1.0）
    start_time: float   # 起始拍位
    velocity: int = 80  # 力度

    @property
    def end_time(self) -> float:
        return self.start_time + self.duration

    @property
    def pitch_name(self) -> str:
        names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        return f"{names[self.pitch % 12]}{self.pitch // 12 - 1}"

    def __repr__(self) -> str:
        return f"Note({self.pitch_name}, dur={self.duration}, t={self.start_time:.2f})"
