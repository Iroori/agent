"""DateTime tool with KST-based precise time handling.

Provides accurate date/time operations to prevent LLM hallucinations.
All operations use Asia/Seoul timezone (UTC+9).
"""

from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Literal

from loguru import logger
from pydantic import BaseModel, Field

from app.tools.registry import get_tool_registry

# Korea Standard Time (UTC+9)
KST = timezone(timedelta(hours=9))

# Day names in Korean
_DAY_NAMES_KO = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]
_DAY_NAMES_EN = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Common datetime input formats for parsing
_INPUT_FORMATS = [
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
    "%Y/%m/%d %H:%M:%S.%f",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d %H:%M",
    "%Y/%m/%d",
    "%Y.%m.%d %H:%M:%S.%f",
    "%Y.%m.%d %H:%M:%S",
    "%Y.%m.%d %H:%M",
    "%Y.%m.%d",
    "%d-%m-%Y %H:%M:%S",
    "%d/%m/%Y %H:%M:%S",
    "%Y%m%d%H%M%S",
    "%Y%m%d",
]


class DateTimeAction(str, Enum):
    """Available datetime actions."""

    GET_NOW = "get_now"
    CALCULATE_DELTA = "calculate_delta"
    DIFF_BETWEEN = "diff_between"


class DateTimeToolInput(BaseModel):
    """Input schema for DateTimeTool."""

    action: DateTimeAction = Field(
        description=(
            "Action to perform: "
            "'get_now' - get current KST time, "
            "'calculate_delta' - calculate time before/after a reference, "
            "'diff_between' - calculate difference between two times"
        )
    )
    reference_time: str | None = Field(
        default=None,
        description=(
            "Reference datetime string for calculate_delta or first datetime for diff_between. "
            "Supports various formats: 'YYYY-MM-DD HH:mm:ss.SSS', 'YYYY-MM-DD', etc. "
            "If not provided for calculate_delta, uses current time."
        ),
    )
    target_time: str | None = Field(
        default=None,
        description="Second datetime string for diff_between action.",
    )
    days: int = Field(
        default=0,
        description="Number of days for calculate_delta (positive=future, negative=past).",
    )
    hours: int = Field(
        default=0,
        description="Number of hours for calculate_delta (positive=future, negative=past).",
    )
    minutes: int = Field(
        default=0,
        description="Number of minutes for calculate_delta (positive=future, negative=past).",
    )
    seconds: int = Field(
        default=0,
        description="Number of seconds for calculate_delta (positive=future, negative=past).",
    )
    milliseconds: int = Field(
        default=0,
        description="Number of milliseconds for calculate_delta (positive=future, negative=past).",
    )


def _format_datetime(dt: datetime) -> str:
    """Format datetime to standard output format.

    Args:
        dt: Datetime object

    Returns:
        Formatted string: YYYY-MM-DD HH:mm:ss.SSS KST
    """
    return dt.strftime("%Y-%m-%d %H:%M:%S.") + f"{dt.microsecond // 1000:03d} KST"


def _parse_datetime(dt_string: str) -> datetime:
    """Parse datetime string with various formats.

    Args:
        dt_string: Datetime string in various formats

    Returns:
        Datetime object in KST

    Raises:
        ValueError: If format is not recognized
    """
    dt_string = dt_string.strip()

    # Remove timezone suffix if present
    for suffix in [" KST", " UTC", " GMT", "+09:00", "+0900"]:
        if dt_string.endswith(suffix):
            dt_string = dt_string[: -len(suffix)].strip()
            break

    # Try each format
    for fmt in _INPUT_FORMATS:
        try:
            dt = datetime.strptime(dt_string, fmt)
            # Assume input is in KST
            return dt.replace(tzinfo=KST)
        except ValueError:
            continue

    # Try ISO format
    try:
        dt = datetime.fromisoformat(dt_string.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=KST)
        else:
            dt = dt.astimezone(KST)
        return dt
    except ValueError:
        pass

    raise ValueError(
        f"Unable to parse datetime: '{dt_string}'. "
        "Supported formats: YYYY-MM-DD HH:mm:ss.SSS, YYYY-MM-DD, YYYY/MM/DD, etc."
    )


def _get_now() -> dict:
    """Get current KST time with detailed information.

    Returns:
        Dict with datetime, day of week, and Unix timestamp
    """
    now = datetime.now(KST)
    weekday_idx = now.weekday()

    return {
        "datetime": _format_datetime(now),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S.") + f"{now.microsecond // 1000:03d}",
        "year": now.year,
        "month": now.month,
        "day": now.day,
        "hour": now.hour,
        "minute": now.minute,
        "second": now.second,
        "millisecond": now.microsecond // 1000,
        "weekday": _DAY_NAMES_EN[weekday_idx],
        "weekday_ko": _DAY_NAMES_KO[weekday_idx],
        "weekday_number": weekday_idx + 1,  # 1=Monday, 7=Sunday
        "unix_timestamp": int(now.timestamp()),
        "unix_timestamp_ms": int(now.timestamp() * 1000),
        "timezone": "Asia/Seoul (KST, UTC+9)",
    }


def _calculate_delta(
    reference_time: str | None,
    days: int,
    hours: int,
    minutes: int,
    seconds: int,
    milliseconds: int,
) -> dict:
    """Calculate datetime delta from reference time.

    Args:
        reference_time: Reference datetime string (None = current time)
        days: Days to add/subtract
        hours: Hours to add/subtract
        minutes: Minutes to add/subtract
        seconds: Seconds to add/subtract
        milliseconds: Milliseconds to add/subtract

    Returns:
        Dict with reference, result, and delta info
    """
    # Parse reference time or use current time
    if reference_time:
        ref_dt = _parse_datetime(reference_time)
    else:
        ref_dt = datetime.now(KST)

    # Calculate delta
    delta = timedelta(
        days=days,
        hours=hours,
        minutes=minutes,
        seconds=seconds,
        milliseconds=milliseconds,
    )
    result_dt = ref_dt + delta

    # Determine direction
    if delta.total_seconds() > 0:
        direction = "future"
    elif delta.total_seconds() < 0:
        direction = "past"
    else:
        direction = "same"

    weekday_idx = result_dt.weekday()

    return {
        "reference": _format_datetime(ref_dt),
        "result": _format_datetime(result_dt),
        "result_date": result_dt.strftime("%Y-%m-%d"),
        "result_time": result_dt.strftime("%H:%M:%S.") + f"{result_dt.microsecond // 1000:03d}",
        "weekday": _DAY_NAMES_EN[weekday_idx],
        "weekday_ko": _DAY_NAMES_KO[weekday_idx],
        "delta": {
            "days": days,
            "hours": hours,
            "minutes": minutes,
            "seconds": seconds,
            "milliseconds": milliseconds,
            "total_seconds": delta.total_seconds(),
        },
        "direction": direction,
        "unix_timestamp": int(result_dt.timestamp()),
    }


def _diff_between(reference_time: str, target_time: str) -> dict:
    """Calculate precise difference between two datetimes.

    Args:
        reference_time: First datetime string
        target_time: Second datetime string

    Returns:
        Dict with detailed time difference
    """
    ref_dt = _parse_datetime(reference_time)
    target_dt = _parse_datetime(target_time)

    # Calculate difference
    diff = target_dt - ref_dt
    total_seconds = diff.total_seconds()
    is_future = total_seconds >= 0

    # Convert to absolute for breakdown
    abs_seconds = abs(total_seconds)

    # Break down into components
    days = int(abs_seconds // 86400)
    remaining = abs_seconds % 86400
    hours = int(remaining // 3600)
    remaining = remaining % 3600
    minutes = int(remaining // 60)
    seconds = int(remaining % 60)
    milliseconds = int((abs_seconds * 1000) % 1000)

    return {
        "reference": _format_datetime(ref_dt),
        "target": _format_datetime(target_dt),
        "difference": {
            "days": days if is_future else -days,
            "hours": hours if is_future else -hours,
            "minutes": minutes if is_future else -minutes,
            "seconds": seconds if is_future else -seconds,
            "milliseconds": milliseconds if is_future else -milliseconds,
        },
        "breakdown": {
            "total_days": abs(diff.days),
            "total_hours": int(abs_seconds // 3600),
            "total_minutes": int(abs_seconds // 60),
            "total_seconds": int(abs_seconds),
            "total_milliseconds": int(abs_seconds * 1000),
        },
        "human_readable": _format_human_readable(days, hours, minutes, seconds, is_future),
        "direction": "future" if is_future else "past",
        "is_same_day": ref_dt.date() == target_dt.date(),
    }


def _format_human_readable(days: int, hours: int, minutes: int, seconds: int, is_future: bool) -> str:
    """Format time difference as human-readable string.

    Args:
        days: Number of days
        hours: Number of hours
        minutes: Number of minutes
        seconds: Number of seconds
        is_future: True if target is in future

    Returns:
        Human-readable string
    """
    parts = []
    if days > 0:
        parts.append(f"{days}일")
    if hours > 0:
        parts.append(f"{hours}시간")
    if minutes > 0:
        parts.append(f"{minutes}분")
    if seconds > 0 or not parts:
        parts.append(f"{seconds}초")

    time_str = " ".join(parts)
    if is_future:
        return f"{time_str} 후"
    else:
        return f"{time_str} 전"


async def datetime_operate(
    action: DateTimeAction,
    reference_time: str | None = None,
    target_time: str | None = None,
    days: int = 0,
    hours: int = 0,
    minutes: int = 0,
    seconds: int = 0,
    milliseconds: int = 0,
) -> str:
    """Perform datetime operation.

    Args:
        action: Operation to perform
        reference_time: Reference datetime for delta/diff operations
        target_time: Target datetime for diff operation
        days: Days delta
        hours: Hours delta
        minutes: Minutes delta
        seconds: Seconds delta
        milliseconds: Milliseconds delta

    Returns:
        JSON-formatted result string
    """
    import json

    try:
        if action == DateTimeAction.GET_NOW:
            result = _get_now()
        elif action == DateTimeAction.CALCULATE_DELTA:
            result = _calculate_delta(
                reference_time, days, hours, minutes, seconds, milliseconds
            )
        elif action == DateTimeAction.DIFF_BETWEEN:
            if not reference_time or not target_time:
                return "Error: Both reference_time and target_time are required for diff_between"
            result = _diff_between(reference_time, target_time)
        else:
            return f"Error: Unknown action '{action}'"

        return json.dumps(result, ensure_ascii=False, indent=2)

    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        logger.error(f"Unexpected error in datetime operation: {e}")
        return f"Error: {e}"


def register_datetime_tool() -> None:
    """Register the datetime tool with the global registry."""
    registry = get_tool_registry()

    # Check if already registered
    if registry.get("datetime") is not None:
        logger.debug("DateTime tool already registered")
        return

    @registry.register(
        name="datetime",
        description=(
            "Get current time or perform date/time calculations with KST (Korea Standard Time). "
            "Use this tool for any date/time related queries to ensure accuracy. "
            "Actions: 'get_now' (current KST time, weekday, timestamp), "
            "'calculate_delta' (add/subtract days/hours/minutes/seconds from a time), "
            "'diff_between' (calculate precise difference between two times). "
            "All times use Asia/Seoul timezone (UTC+9). "
            "Examples: get current time, calculate 3 days from now, find days between two dates."
        ),
        category="datetime",
        args_schema=DateTimeToolInput,
    )
    async def datetime_tool(
        action: DateTimeAction,
        reference_time: str | None = None,
        target_time: str | None = None,
        days: int = 0,
        hours: int = 0,
        minutes: int = 0,
        seconds: int = 0,
        milliseconds: int = 0,
    ) -> str:
        """Perform datetime operation."""
        return await datetime_operate(
            action, reference_time, target_time, days, hours, minutes, seconds, milliseconds
        )

    logger.debug("DateTime tool registered")


# Auto-register on module import for mandatory binding
register_datetime_tool()
