"""Build the Anthropic client from settings."""

import anthropic

from review_intel.settings import Settings

# The SDK retries 408/409/429/5xx and connection errors with exponential backoff;
# a few more attempts than its default of 2 rides out short rate-limit bursts.
MAX_RETRIES = 4


def make_client(settings: Settings) -> anthropic.Anthropic:
    """Use the key from `.env` if set; otherwise the SDK resolves credentials itself
    (ANTHROPIC_API_KEY in the environment, or an `ant auth login` profile)."""
    key = settings.anthropic_api_key
    if key is not None and key.get_secret_value():
        return anthropic.Anthropic(api_key=key.get_secret_value(), max_retries=MAX_RETRIES)
    return anthropic.Anthropic(max_retries=MAX_RETRIES)
