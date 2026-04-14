from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from typing import Any

import requests
from Crypto.Cipher import AES

WIDGET_ID = "a8bdc7fc82120c79a3718bae1c884714"
DEFAULT_ORIGIN = "https://www.nesine.com"
DEFAULT_ACCESS_LEVEL = "trial"
DEFAULT_LANGUAGE = "en"
DEFAULT_TIMEZONE = "Etc:UTC"


def evp_bytes_to_key(password: bytes, salt: bytes, key_len: int, iv_len: int) -> tuple[bytes, bytes]:
    """OpenSSL EVP_BytesToKey using MD5 and one iteration."""
    digest = b""
    previous = b""

    while len(digest) < key_len + iv_len:
        previous = hashlib.md5(previous + password + salt).digest()
        digest += previous

    return digest[:key_len], digest[key_len : key_len + iv_len]


def decrypt_openssl_aes(ciphertext_b64: str, passphrase: str) -> str:
    raw = base64.b64decode(ciphertext_b64)
    if raw[:8] != b"Salted__":
        raise ValueError("Invalid OpenSSL format")

    salt = raw[8:16]
    ciphertext = raw[16:]
    key, iv = evp_bytes_to_key(passphrase.encode(), salt, key_len=32, iv_len=16)

    cipher = AES.new(key, AES.MODE_CBC, iv)
    decrypted = cipher.decrypt(ciphertext)
    pad_len = decrypted[-1]
    plaintext = decrypted[:-pad_len]
    return plaintext.decode("utf-8")


def sport_event_id_to_match_id(sport_event_id: str) -> int:
    parts = sport_event_id.split(":")
    if len(parts) != 3 or parts[0] != "sr" or parts[1] != "sport_event":
        raise ValueError(f"Unexpected sport_event id format: {sport_event_id}")
    return int(parts[2])


@dataclass(frozen=True)
class SportradarApiConfig:
    api_key: str
    access_level: str = DEFAULT_ACCESS_LEVEL
    language: str = DEFAULT_LANGUAGE
    format: str = "json"
    base_url: str = "https://api.sportradar.com"

    def api_url(self, path: str) -> str:
        return f"{self.base_url}/soccer/{self.access_level}/v4/{self.language}/{path}.{self.format}"


@dataclass(frozen=True)
class LmtConfig:
    widget_id: str = WIDGET_ID
    origin: str = DEFAULT_ORIGIN
    language: str = DEFAULT_LANGUAGE
    timezone: str = DEFAULT_TIMEZONE
    widgets_base_url: str = "https://widgets.sir.sportradar.com"
    lmt_base_url: str = "https://lmt.fn.sportradar.com"

    @property
    def licensing_url(self) -> str:
        return f"{self.widgets_base_url}/{self.widget_id}/licensing"

    @property
    def passphrase(self) -> str:
        return self.widget_id

    def gismo_url(self, endpoint: str, entity_id: int | str, token: str) -> str:
        return (
            f"{self.lmt_base_url}/common/{self.language}/{self.timezone}/gismo/"
            f"{endpoint}/{entity_id}?T={token}"
        )


class SportradarApiClient:
    def __init__(
        self,
        config: SportradarApiConfig,
        *,
        session: requests.Session | None = None,
        timeout: int = 30,
    ) -> None:
        self.config = config
        self.session = session or requests.Session()
        self.timeout = timeout

    def _get_json(self, path: str) -> dict[str, Any]:
        response = self.session.get(
            self.config.api_url(path),
            headers={"x-api-key": self.config.api_key},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def fetch_seasons(self) -> dict[str, Any]:
        return self._get_json("seasons")

    def fetch_season_schedule(self, season_id: str) -> dict[str, Any]:
        return self._get_json(f"seasons/{season_id}/schedules")


class LmtClient:
    def __init__(
        self,
        config: LmtConfig | None = None,
        *,
        session: requests.Session | None = None,
        timeout: int = 30,
    ) -> None:
        self.config = config or LmtConfig()
        self.session = session or requests.Session()
        self.timeout = timeout
        self._token: str | None = None

    def _request_headers(self) -> dict[str, str]:
        return {
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Accept-Language": "en-GB;q=0.9",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Origin": self.config.origin,
            "Pragma": "no-cache",
            "Referer": f"{self.config.origin}/",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "cross-site",
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:149.0) "
                "Gecko/20100101 Firefox/149.0"
            ),
        }

    def fetch_license_payload(self) -> dict[str, Any]:
        response = self.session.get(
            self.config.licensing_url,
            headers=self._request_headers(),
            timeout=self.timeout,
        )
        response.raise_for_status()
        encrypted_text = response.json().get("text")
        if not encrypted_text:
            raise ValueError("Licensing response did not contain a 'text' field")

        decrypted = decrypt_openssl_aes(encrypted_text, self.config.passphrase)
        return json.loads(decrypted)

    def fetch_fishnet_token(self) -> str:
        payload = self.fetch_license_payload()
        token = payload.get("fishnetToken", {}).get("token")
        if not token:
            raise ValueError("Licensing payload did not contain fishnetToken['token']")
        self._token = token
        return token

    def _get_json(self, endpoint: str, entity_id: int | str) -> dict[str, Any]:
        if not self._token:
            self.fetch_fishnet_token()

        assert self._token is not None
        url = self.config.gismo_url(endpoint, entity_id, self._token)

        try:
            response = self.session.get(url, headers=self._request_headers(), timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as exc:
            response = exc.response
            if response is None or response.status_code != 403:
                raise

        fresh_token = self.fetch_fishnet_token()
        retry_url = self.config.gismo_url(endpoint, entity_id, fresh_token)
        retry_response = self.session.get(retry_url, headers=self._request_headers(), timeout=self.timeout)
        retry_response.raise_for_status()
        return retry_response.json()

    def fetch_endpoint(self, endpoint: str, entity_id: int | str) -> dict[str, Any]:
        return self._get_json(endpoint, entity_id)

    def fetch_match_timeline(self, match_id: int) -> dict[str, Any]:
        return self.fetch_endpoint("match_timeline", match_id)

    def fetch_match_info(self, match_id: int) -> dict[str, Any]:
        return self.fetch_endpoint("match_info", match_id)

    def fetch_match_detailsextended(self, match_id: int) -> dict[str, Any]:
        return self.fetch_endpoint("match_detailsextended", match_id)

    def fetch_match_phrases(self, match_id: int) -> dict[str, Any]:
        return self.fetch_endpoint("match_phrases", match_id)

    def fetch_match_squads(self, match_id: int) -> dict[str, Any]:
        return self.fetch_endpoint("match_squads", match_id)

    def fetch_match_bundle(
        self,
        match_id: int,
        *,
        include_phrases: bool = True,
        include_squads: bool = True,
    ) -> dict[str, dict[str, Any]]:
        bundle = {
            "match_timeline": self.fetch_match_timeline(match_id),
            "match_info": self.fetch_match_info(match_id),
            "match_detailsextended": self.fetch_match_detailsextended(match_id),
        }
        if include_phrases:
            bundle["match_phrases"] = self.fetch_match_phrases(match_id)
        if include_squads:
            bundle["match_squads"] = self.fetch_match_squads(match_id)
        return bundle


def main() -> None:
    client = LmtClient()
    print(json.dumps(client.fetch_license_payload(), indent=2))


if __name__ == "__main__":
    main()
