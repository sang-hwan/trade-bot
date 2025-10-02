# splits.py
"""
워크포워드(rolling/expanding) 분할 도우미.

공개 API
- walk_forward(index_or_df, *, train_years=3, test_years=1, step_years=None,
               mode="rolling", allow_incomplete_last=False) -> list[Split]
- iter_splits(index_or_df, **kwargs) -> Iterator[Split]
- slice_by_split(df, split: Split) -> tuple[pd.DataFrame, pd.DataFrame]

계약
- 입력 인덱스는 DatetimeIndex(tz='UTC') 단조 증가·중복 없음(quality_gate에서 보장).
- 학습/검증 경계는 시간 기준으로 잡고, 실제 슬라이스는 인덱스의 바 시각으로 매핑(양 끝 포함).
- mode="rolling": 학습 윈도 길이 고정, 시작을 step_years만큼 전진.
  mode="expanding": 학습 시작 고정, 종료 시점만 step_years만큼 확장.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Tuple, Union, List
import pandas as pd


__all__ = ["Split", "walk_forward", "iter_splits", "slice_by_split"]


@dataclass(frozen=True)
class Split:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


IndexLike = Union[pd.DatetimeIndex, pd.Series, pd.DataFrame]


def _as_index(obj: IndexLike) -> pd.DatetimeIndex:
    """DatetimeIndex(UTC) 추출 및 검증: tz-aware·UTC·단조증가·중복 없음."""
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        idx = obj.index
    elif isinstance(obj, pd.DatetimeIndex):
        idx = obj
    else:
        raise TypeError("index_or_df must be a DataFrame/Series with DatetimeIndex or a DatetimeIndex.")

    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError("Index must be a pandas DatetimeIndex.")

    if idx.tz is None:
        raise ValueError("DatetimeIndex must be tz-aware (UTC).")

    # 다양한 tz 구현(pytz/zoneinfo)에 안전하게 UTC 판정
    tzname = getattr(idx.tz, "key", getattr(idx.tz, "zone", str(idx.tz)))
    if tzname != "UTC":
        raise ValueError("DatetimeIndex timezone must be 'UTC'.")

    if not idx.is_monotonic_increasing:
        raise ValueError("DatetimeIndex must be strictly increasing.")
    if idx.has_duplicates:
        raise ValueError("DatetimeIndex must not contain duplicates.")

    return idx


def _years_offset(ts: pd.Timestamp, years: int) -> pd.Timestamp:
    """연 단위 오프셋(윤년 고려)."""
    if years <= 0:
        raise ValueError("years must be positive.")
    return ts + pd.DateOffset(years=years)


def _bound_to_index(idx: pd.DatetimeIndex, left: pd.Timestamp, right: pd.Timestamp) -> Tuple[int, int]:
    """시간 경계를 인덱스 구간 [i_left, i_right]로 매핑(양 끝 포함)."""
    i_left = int(idx.searchsorted(left, side="left"))
    i_right = int(idx.searchsorted(right, side="right")) - 1
    return i_left, i_right


def _validate_mode(mode: str) -> None:
    if mode not in {"rolling", "expanding"}:
        raise ValueError("mode must be 'rolling' or 'expanding'.")


def iter_splits(
    index_or_df: IndexLike,
    *,
    train_years: int = 3,
    test_years: int = 1,
    step_years: int | None = None,
    mode: str = "rolling",
    allow_incomplete_last: bool = False,
) -> Iterator[Split]:
    """
    워크포워드 분할 제너레이터.
    - train_years/test_years: 기간(년) 기반 분할.
    - step_years: None이면 test_years와 동일.
    - allow_incomplete_last: 마지막 검증 구간이 부분만 차도 포함 여부.
    """
    _validate_mode(mode)
    if train_years <= 0 or test_years <= 0:
        raise ValueError("train_years and test_years must be positive integers.")
    if step_years is None:
        step_years = test_years
    if step_years <= 0:
        raise ValueError("step_years must be positive.")

    idx = _as_index(index_or_df)
    if len(idx) < 2:
        return

    first, last = idx[0], idx[-1]
    train_start_time = first
    train_end_time = _years_offset(train_start_time, train_years)

    while True:
        i_tr_start, i_tr_end = _bound_to_index(idx, train_start_time, train_end_time)
        if i_tr_end < i_tr_start:
            break

        if i_tr_end + 1 >= len(idx):
            break  # 학습으로 끝까지 소진

        test_start_ts = idx[i_tr_end + 1]
        test_end_time = _years_offset(test_start_ts, test_years)
        i_te_start, i_te_end = _bound_to_index(idx, test_start_ts, test_end_time)
        if i_te_end < i_te_start:
            break

        is_incomplete = (idx[i_te_end] < test_end_time) and (i_te_end == len(idx) - 1)
        if is_incomplete and not allow_incomplete_last:
            break

        yield Split(
            train_start=idx[i_tr_start],
            train_end=idx[i_tr_end],
            test_start=idx[i_te_start],
            test_end=idx[i_te_end],
        )

        if mode == "rolling":
            train_start_time = _years_offset(train_start_time, step_years)
            train_end_time = _years_offset(train_start_time, train_years)
            if train_start_time > last:
                break
        else:  # expanding
            train_end_time = _years_offset(train_end_time, step_years)
            # 다음 루프에서 경계 검증으로 자연 종료


def walk_forward(
    index_or_df: IndexLike,
    *,
    train_years: int = 3,
    test_years: int = 1,
    step_years: int | None = None,
    mode: str = "rolling",
    allow_incomplete_last: bool = False,
) -> List[Split]:
    """워크포워드 분할을 리스트로 수집."""
    return list(
        iter_splits(
            index_or_df,
            train_years=train_years,
            test_years=test_years,
            step_years=step_years,
            mode=mode,
            allow_incomplete_last=allow_incomplete_last,
        )
    )


def slice_by_split(df: pd.DataFrame, split: Split) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split 경계로 DataFrame을 학습/검증 슬라이스(양 끝 포함). 비어 있으면 실패."""
    _ = _as_index(df)  # 계약 검증
    train = df.loc[split.train_start : split.train_end]
    test = df.loc[split.test_start : split.test_end]
    if train.empty or test.empty:
        raise ValueError("Empty train/test slice produced; check split boundaries and index coverage.")
    return train, test
