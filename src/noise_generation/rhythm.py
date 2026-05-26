"""节奏时值分配

为音符序列分配时值（全音符/二分/四分/八分/十六分音符等），
确保总时值覆盖至少指定小节数。
"""

import numpy as np

from src.config import MelodyConfig, DURATIONS


def assign_durations(
    num_notes: int,
    config: MelodyConfig,
    noise: np.ndarray | None = None,
) -> list[float]:
    """为指定数量的音符分配时值。

    Args:
        num_notes: 音符数量
        config: 旋律配置（含时值权重）
        noise: 可选的噪声序列，用于噪声驱动的时值分配。
               若为 None 则使用纯概率分配。

    Returns:
        时值列表（四分音符=1.0）
    """
    dur_names = list(config.duration_weights.keys())
    dur_values = [DURATIONS[d] for d in dur_names]
    weights = np.array([config.duration_weights[d] for d in dur_names])
    weights = weights / weights.sum()  # 归一化

    if noise is not None:
        # 噪声驱动：将噪声值映射到时值索引
        indices = np.clip(
            (noise[:num_notes] * len(dur_values)).astype(int),
            0, len(dur_values) - 1,
        )
        durations = [dur_values[i] for i in indices]
    else:
        # 概率分配
        indices = np.random.choice(len(dur_values), size=num_notes, p=weights)
        durations = [dur_values[i] for i in indices]

    return durations


def build_note_sequence(
    pitches: np.ndarray,
    config: MelodyConfig,
    duration_noise: np.ndarray | None = None,
) -> list:
    """从音高序列和时值配置构建完整音符序列。

    确保总长度至少覆盖 min_measures 个小节。

    Returns:
        Note 对象列表
    """
    from src.note import Note

    beats_per_measure = config.time_signature[0]
    target_beats = config.min_measures * beats_per_measure

    durations = assign_durations(len(pitches), config, duration_noise)

    notes = []
    current_time = 0.0

    for i, (pitch, dur) in enumerate(zip(pitches, durations)):
        notes.append(Note(
            pitch=int(pitch),
            duration=dur,
            start_time=current_time,
            velocity=config.velocity,
        ))
        current_time += dur

        # 达到目标小节数后停止
        if current_time >= target_beats:
            break

    # 如果音符不够填满目标小节数，循环使用
    if current_time < target_beats:
        idx = 0
        while current_time < target_beats:
            pitch = int(pitches[idx % len(pitches)])
            dur = durations[idx % len(durations)]
            notes.append(Note(
                pitch=pitch,
                duration=dur,
                start_time=current_time,
                velocity=config.velocity,
            ))
            current_time += dur
            idx += 1

    return notes
