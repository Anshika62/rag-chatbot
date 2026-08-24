import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from langchain_core.tools import tool


logger = logging.getLogger(__name__)


@tool
def get_current_datetime(
    timezone: str = "Asia/Kolkata",
) -> str:
    """Get the current date and time for a specified IANA timezone."""

    try:
        now = datetime.now(
            ZoneInfo(timezone)
        )

        return (
            f"Current date: {now.strftime('%Y-%m-%d')}\n"
            f"Current time: {now.strftime('%I:%M %p')}\n"
            f"Timezone: {timezone}"
        )

    except Exception as exc:
        logger.exception(
            "DATETIME TOOL ERROR: timezone=%s",
            timezone,
        )

        return (
            f"Unable to get current date and time "
            f"for timezone: {timezone}"
        )