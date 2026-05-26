"""Pipeline 配置定义"""

from dataclasses import dataclass, field
from pathlib import Path
import json

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# SoundFont 路径
SOUNDFONT_PATH = PROJECT_ROOT / "soundfonts" / "TimGM6mb.sf2"


# ---------- 音阶定义 ----------
# MIDI pitch class (0-11) 对应 C, C#, D, ..., B
SCALES = {
    "C_major": [0, 2, 4, 5, 7, 9, 11],          # C D E F G A B
    "C_minor": [0, 2, 3, 5, 7, 8, 10],           # C D Eb F G Ab Bb
    "pentatonic_major": [0, 2, 4, 7, 9],          # C D E G A
    "pentatonic_minor": [0, 3, 5, 7, 10],         # C Eb F G Bb
    "chromatic": list(range(12)),                   # 全部半音
}

# 时值定义（以四分音符 = 1.0 拍为基准）
DURATIONS = {
    "whole": 4.0,
    "half": 2.0,
    "quarter": 1.0,
    "eighth": 0.5,
    "sixteenth": 0.25,
}

# 默认时值概率分布
DEFAULT_DURATION_WEIGHTS = {
    "whole": 0.05,
    "half": 0.15,
    "quarter": 0.35,
    "eighth": 0.30,
    "sixteenth": 0.15,
}


@dataclass
class NoiseConfig:
    """噪声生成参数"""
    beta: float = 1.0          # 频谱指数: 1=pink, 2=brown, 0=white
    num_samples: int = 256     # 噪声采样点数（每个点映射一个音符）
    seed: int | None = None    # 随机种子，None 表示不固定


@dataclass
class MelodyConfig:
    """旋律映射参数"""
    scale: str = "C_major"             # 音阶名称
    octave_low: int = 4                # 最低八度 (MIDI: octave*12)
    octave_high: int = 5               # 最高八度
    bpm: int = 100                     # 速度
    time_signature: tuple[int, int] = (4, 4)  # 拍号
    min_measures: int = 16             # 最少小节数
    duration_weights: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_DURATION_WEIGHTS)
    )
    midi_program: int = 0              # General MIDI 乐器编号 (0=钢琴)
    velocity: int = 80                 # 音符力度 (0-127)


@dataclass
class RenderConfig:
    """音频渲染参数"""
    sample_rate: int = 44100
    soundfont_path: str = str(SOUNDFONT_PATH)


@dataclass
class PipelineConfig:
    """完整 pipeline 配置"""
    name: str = "default"
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    melody: MelodyConfig = field(default_factory=MelodyConfig)
    render: RenderConfig = field(default_factory=RenderConfig)

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> "PipelineConfig":
        with open(path) as f:
            data = json.load(f)
        noise = NoiseConfig(**data.get("noise", {}))
        melody_data = data.get("melody", {})
        if "time_signature" in melody_data:
            melody_data["time_signature"] = tuple(melody_data["time_signature"])
        melody = MelodyConfig(**melody_data)
        render = RenderConfig(**data.get("render", {}))
        return cls(
            name=data.get("name", "default"),
            noise=noise,
            melody=melody,
            render=render,
        )
