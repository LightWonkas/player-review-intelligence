"""Store collectors behind one `ReviewSource` interface."""

from review_intel.collectors.app_store import AppStoreSource
from review_intel.collectors.base import RawReview, ReviewSource
from review_intel.collectors.google_play import GooglePlaySource
from review_intel.domain import Store


def build_sources() -> dict[Store, ReviewSource]:
    """Production sources, one per store. Tests swap this out for fakes."""
    return {Store.GOOGLE_PLAY: GooglePlaySource(), Store.APP_STORE: AppStoreSource()}


__all__ = ["AppStoreSource", "GooglePlaySource", "RawReview", "ReviewSource", "build_sources"]
