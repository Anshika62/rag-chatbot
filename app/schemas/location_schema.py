from enum import Enum

from pydantic import BaseModel, Field


class LocationSource(str, Enum):
    CURRENT_LOCATION = "current_location"
    SEARCH = "search"
    MAP = "map"


class LocationSubmission(BaseModel):
    """
    Payload the frontend sends after the user provides a location
    via any of the three supported UI modes (current location,
    search, or map selection).

    latitude/longitude are the only required fields. address/name
    are optional context the frontend may already have (e.g. from
    a search suggestion or reverse geocoding). source records
    which UI mode was used, for logging/analytics only — it does
    not change backend behavior.
    """

    latitude: float = Field(
        ...,
        ge=-90,
        le=90,
        description="Latitude in decimal degrees, -90 to 90.",
    )

    longitude: float = Field(
        ...,
        ge=-180,
        le=180,
        description="Longitude in decimal degrees, -180 to 180.",
    )

    address: str | None = Field(
        default=None,
        description="Optional human-readable address.",
    )

    name: str | None = Field(
        default=None,
        description="Optional short place name (e.g. 'Vijay Nagar, Indore').",
    )

    source: LocationSource | None = Field(
        default=None,
        description="Which UI mode produced this location.",
    )