from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any, Iterable

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pymongo import ASCENDING, MongoClient, UpdateOne
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

DATABASE_NAME = "football_analytics"
COLLECTION_NAME = "mackolik"
BASE_URL = "https://www.mackolik.com"
ARCHIVE_BASE_URL = "https://arsiv.mackolik.com"
LIVEDATA_URL = "https://vd.mackolik.com/livedata"
MATCH_HEADER_URL = f"{BASE_URL}/perform/p0/ajax/components/match/matchHeader"
KEY_EVENTS_URL = f"{BASE_URL}/ajax/football/key-events"
DEFAULT_TIMEOUT = 30
DEFAULT_MAX_WORKERS = min(8, max(os.cpu_count() or 1, 1))
ARCHIVE_OPTA_STATS_RAW_URL = f"{ARCHIVE_BASE_URL}/AjaxHandlers/MatchHandler.aspx"
OPTA_AUTH_URL = "https://omo.akamai.opta.net/auth/"
OPTA_WIDGET_USER = "OW2017"
OPTA_WIDGET_PASSWORD = "dXWg5gVZ"
OPTA_WIDGET_SPS = "widgets"
OPTA_FEED_NOT_FOUND_TEXT = "feed_type, game_id combination not found in the feed repository"


class FatalRateLimitError(Exception):
    pass


@dataclass(slots=True)
class MatchRef:
    match_id: str
    match_slug: str | None
    match_date: date
    home_name: str
    away_name: str
    current_match_id: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Mackolik football match data and store it in MongoDB.")
    parser.add_argument("--start-date", default=None, help="Inclusive start date in YYYY-MM-DD format. Defaults to today.")
    parser.add_argument("--end-date", default=None, help="Inclusive end date in YYYY-MM-DD format. Defaults to start-date.")
    parser.add_argument("--days-back", type=int, default=None, help="If provided, fetch from today - days_back through today.")
    parser.add_argument("--reverse", action="store_true", help="Fetch dates in reverse order from end-date back to start-date.")
    parser.add_argument(
        "--continue",
        dest="continue_fetch",
        action="store_true",
        help="Skip matches whose match_id is already present in MongoDB.",
    )
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS, help="Number of worker threads for per-match fetches.")
    parser.add_argument("--limit-matches", type=int, default=None)
    parser.add_argument("--upsert-batch-size", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_iso_date(value: str | None) -> date | None:
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%d").date()


def format_livedata_date(value: date) -> str:
    return value.strftime("%d/%m/%Y")


def resolve_date_range(args: argparse.Namespace) -> tuple[date, date]:
    today = datetime.now(UTC).date()
    if args.days_back is not None:
        start_date = today - timedelta(days=max(args.days_back, 0))
        return start_date, today
    start_date = parse_iso_date(args.start_date) or today
    end_date = parse_iso_date(args.end_date) or start_date
    if end_date < start_date:
        raise ValueError("end-date must be on or after start-date")
    return start_date, end_date


def daterange(start_date: date, end_date: date) -> Iterable[date]:
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def iter_target_dates(start_date: date, end_date: date, *, reverse: bool = False) -> Iterable[date]:
    if not reverse:
        yield from daterange(start_date, end_date)
        return
    current = end_date
    while current >= start_date:
        yield current
        current -= timedelta(days=1)


def progress_iterable(
    iterable: Iterable[Any],
    *,
    total: int | None = None,
    desc: str,
    unit: str,
) -> Iterable[Any]:
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, unit=unit)


def normalize_max_workers(value: int | str | None) -> int:
    if value is None:
        return 1
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return 1
    return max(1, numeric)


def livedata_row_match_id(match_row: Any) -> str | None:
    if not isinstance(match_row, list) or not match_row:
        return None
    match_id = match_row[0]
    if match_id in (None, ""):
        return None
    return str(match_id)


def create_session() -> requests.Session:
    retry = Retry(
        total=3,
        backoff_factor=3.0,
        status_forcelist=(500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0 Safari/537.36",
            "Accept-Language": "tr-TR,tr;q=0.9,en;q=0.8",
        }
    )
    return session


def load_collection() -> tuple[MongoClient, Any]:
    load_dotenv(".env")
    mongo_connection_string = os.getenv("MONGO_CONNECTION_STRING")
    if not mongo_connection_string:
        raise RuntimeError("MONGO_CONNECTION_STRING not found in .env")
    client = MongoClient(mongo_connection_string, serverSelectionTimeoutMS=20_000)
    client.admin.command("ping")
    collection = client[DATABASE_NAME][COLLECTION_NAME]
    collection.create_index([("match_id", ASCENDING)], unique=True, name="match_id_unique")
    collection.create_index([("match_date", ASCENDING)], name="match_date_idx")
    collection.create_index([("competition.id", ASCENDING), ("match_date", ASCENDING)], name="competition_date_idx")
    return client, collection


def request_json(
    session: requests.Session,
    url: str,
    *,
    params: list[tuple[str, str]] | None = None,
    referer: str,
    timeout: int,
    extra_headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    headers = {"X-Requested-With": "XMLHttpRequest", "Referer": referer}
    if extra_headers:
        headers.update(extra_headers)
    response = session.get(
        url,
        params=params,
        timeout=timeout,
        headers=headers,
    )
    if response.status_code == 429:
        raise FatalRateLimitError()
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected payload type from {url}")
    return payload


def request_text(
    session: requests.Session,
    url: str,
    *,
    params: list[tuple[str, str]] | None = None,
    referer: str,
    timeout: int,
    extra_headers: dict[str, str] | None = None,
) -> str:
    headers = {"Referer": referer}
    if extra_headers:
        headers.update(extra_headers)
    response = session.get(url, params=params, timeout=timeout, headers=headers)
    if response.status_code == 429:
        raise FatalRateLimitError()
    response.raise_for_status()
    return response.text


def fetch_livescores(session: requests.Session, match_date: date, timeout: int) -> dict[str, Any]:
    return request_json(
        session,
        LIVEDATA_URL,
        params=[("date", format_livedata_date(match_date))],
        referer=BASE_URL,
        timeout=timeout,
    )


def is_finished_livedata_row(row: list[Any]) -> bool:
    if len(row) < 7:
        return False
    row_state = coerce_int(row[5])
    if row_state == 4:
        return True
    status_text = str(row[6]).strip() if row[6] is not None else ""
    return status_text in {"MS", "Ert.", "İpt.", "Tatil", "Hük."}


def parse_livedata_match_row(row: list[Any], match_date: date) -> dict[str, Any] | None:
    if len(row) < 37:
        return None
    home_score = coerce_int(row[29])
    away_score = coerce_int(row[30])
    status_text = str(row[6]).strip() if row[6] is not None else None
    row_state = coerce_int(row[5])
    score_payload = {
        "home": home_score if home_score is not None else "",
        "away": away_score if away_score is not None else "",
        "ht": row[7] if row[7] else None,
        "agg": None,
        "pen": None,
    }
    competition_row = row[36] if isinstance(row[36], list) else []
    competition_name = competition_row[3] if len(competition_row) > 3 else None
    competition_country = competition_row[1] if len(competition_row) > 1 else None
    current_state = "pre"
    if is_finished_livedata_row(row):
        current_state = "post"
    elif row_state in {1, 2, 3} or status_text:
        current_state = "live"
    return {
        "id": str(row[0]),
        "rbId": None,
        "matchName": f"{row[2]} vs {row[4]}",
        "homeTeam": {"id": str(row[1]), "name": str(row[2]), "slug": None},
        "awayTeam": {"id": str(row[3]), "name": str(row[4]), "slug": None},
        "matchSlug": None,
        "seriesId": None,
        "stageId": None,
        "formatId": None,
        "sortOrder": 0,
        "mstUtc": None,
        "competitionId": str(competition_row[4]) if len(competition_row) > 4 and competition_row[4] is not None else None,
        "periodId": None,
        "periodStart": None,
        "lastUpdated": None,
        "status": str(row[34]) if row[34] is not None else None,
        "state": current_state,
        "substate": "none",
        "score": score_payload,
        "statusBoxContent": status_text or None,
        "winner": None,
        "advancingTeam": None,
        "redCards": {"home": coerce_int(row[11]) or 0, "away": coerce_int(row[12]) or 0},
        "iddaaCode": coerce_int(row[14]),
        "liveBetting": bool(coerce_int(row[23])),
        "competitionName": competition_name,
        "longName": f"{competition_country} {competition_name}".strip() if competition_name else None,
        "startTime": row[16],
        "matchDateText": str(row[35]) if row[35] else None,
        "livedataRow": row,
    }


def odds_markets_from_livedata(match_payload: dict[str, Any]) -> list[dict[str, Any]]:
    row = match_payload.get("livedataRow")
    if not isinstance(row, list):
        return []
    markets: list[dict[str, Any]] = []
    one = coerce_float(row[18]) if len(row) > 18 else None
    draw = coerce_float(row[19]) if len(row) > 19 else None
    two = coerce_float(row[20]) if len(row) > 20 else None
    if any(value is not None for value in (one, draw, two)):
        markets.append(
            {
                "market_code": "match_result",
                "market_name": "Maç Sonucu",
                "outcomes": [
                    {"name": "1", "odd_text": str(row[18]) if len(row) > 18 else None, "odd": one, "dialog_href": None},
                    {"name": "X", "odd_text": str(row[19]) if len(row) > 19 else None, "odd": draw, "dialog_href": None},
                    {"name": "2", "odd_text": str(row[20]) if len(row) > 20 else None, "odd": two, "dialog_href": None},
                ],
            }
        )
    under = coerce_float(row[21]) if len(row) > 21 else None
    over = coerce_float(row[22]) if len(row) > 22 else None
    if under is not None or over is not None:
        markets.append(
            {
                "market_code": "total_2_5",
                "market_name": "(2,5) Alt/Üst",
                "outcomes": [
                    {"name": "Alt", "odd_text": str(row[21]) if len(row) > 21 else None, "odd": under, "dialog_href": None},
                    {"name": "Üst", "odd_text": str(row[22]) if len(row) > 22 else None, "odd": over, "dialog_href": None},
                ],
            }
        )
    return markets


def match_ref_from_livescore(match_payload: dict[str, Any], match_date: date) -> MatchRef | None:
    match_id = match_payload.get("id")
    home_team = match_payload.get("homeTeam") or {}
    away_team = match_payload.get("awayTeam") or {}
    if not match_id or not home_team.get("name") or not away_team.get("name"):
        return None
    match_id_text = str(match_id)
    current_match_id = None if match_id_text.isdigit() else match_id_text
    return MatchRef(
        match_id=match_id_text,
        match_slug=str(match_payload.get("matchSlug")) if match_payload.get("matchSlug") else None,
        match_date=match_date,
        home_name=str(home_team["name"]),
        away_name=str(away_team["name"]),
        current_match_id=current_match_id,
    )


def fetch_match_header(session: requests.Session, match_ref: MatchRef, timeout: int) -> dict[str, Any]:
    if not match_ref.current_match_id:
        return {}
    match_url = build_match_url(match_ref)
    return request_json(
        session,
        MATCH_HEADER_URL,
        params=[
            ("matchId", match_ref.current_match_id),
            ("sdapiLanguageCode", "tr-mk"),
            ("ajaxViewName", "match-details"),
            ("ajaxPartialViewName", "match-details-status"),
            ("displayMode", "all"),
        ],
        referer=match_url,
        timeout=timeout,
    )


def fetch_key_events(session: requests.Session, match_ref: MatchRef, timeout: int) -> dict[str, Any]:
    if not match_ref.current_match_id:
        return {}
    return request_json(
        session,
        KEY_EVENTS_URL,
        params=[
            ("ajaxViewName", "events"),
            ("matchId", match_ref.current_match_id),
        ],
        referer=build_match_url(match_ref),
        timeout=timeout,
    )


def fetch_match_page_html(session: requests.Session, match_ref: MatchRef, timeout: int) -> str:
    if not match_ref.current_match_id:
        return ""
    return request_text(
        session,
        build_match_url(match_ref),
        referer=f"{BASE_URL}/canli-sonuclar",
        timeout=timeout,
    )


def fetch_match_plus_page_html(session: requests.Session, match_ref: MatchRef, timeout: int) -> str:
    if not match_ref.current_match_id:
        return ""
    return request_text(
        session,
        build_match_plus_url(match_ref),
        referer=build_match_url(match_ref),
        timeout=timeout,
    )


def fetch_iddaa_page_html(session: requests.Session, match_ref: MatchRef, timeout: int) -> str:
    if not match_ref.current_match_id:
        return ""
    return request_text(
        session,
        build_iddaa_url(match_ref),
        referer=build_match_url(match_ref),
        timeout=timeout,
    )


def fetch_archive_match_page_html(session: requests.Session, match_ref: MatchRef, timeout: int) -> str:
    return request_text(
        session,
        build_archive_match_url(match_ref),
        referer=build_match_url(match_ref),
        timeout=timeout,
    )


def fetch_archive_match_plus_page_html(session: requests.Session, match_ref: MatchRef, timeout: int) -> str:
    return request_text(
        session,
        build_archive_match_plus_url(match_ref),
        referer=build_archive_match_url(match_ref),
        timeout=timeout,
    )


def fetch_archive_opta_stats_raw(session: requests.Session, match_ref: MatchRef, timeout: int) -> dict[str, Any]:
    return request_json(
        session,
        ARCHIVE_OPTA_STATS_RAW_URL,
        params=[("command", "optaStatsRaw"), ("id", match_ref.match_id)],
        referer=build_archive_match_plus_url(match_ref),
        timeout=timeout,
    )


def parse_jsonp_payload(raw_value: str) -> dict[str, Any]:
    stripped = raw_value.strip()
    start = stripped.find("(")
    end = stripped.rfind(")")
    if start < 0 or end <= start:
        raise RuntimeError("Unexpected JSONP payload")
    payload = safe_json_loads(stripped[start + 1 : end])
    if not isinstance(payload, dict):
        raise RuntimeError("Unexpected JSON payload type")
    return payload


def is_missing_opta_feed_payload(raw_value: str) -> bool:
    return OPTA_FEED_NOT_FOUND_TEXT in raw_value.lower()


def fetch_direct_opta_feed(
    session: requests.Session,
    match_ref: MatchRef,
    *,
    opta_match_id: str,
    feed_type: str,
    timeout: int,
) -> dict[str, Any]:
    referer = build_archive_match_plus_url(match_ref)
    raw_payload = request_text(
        session,
        OPTA_AUTH_URL,
        params=[
            ("feed_type", feed_type),
            ("game_id", str(opta_match_id)),
            ("user", OPTA_WIDGET_USER),
            ("psw", OPTA_WIDGET_PASSWORD),
            ("sps", OPTA_WIDGET_SPS),
            ("jsoncallback", f"{feed_type}_{opta_match_id}"),
        ],
        referer=referer,
        timeout=timeout,
        extra_headers={"Origin": ARCHIVE_BASE_URL},
    )
    if is_missing_opta_feed_payload(raw_payload):
        raise RuntimeError(f"Opta {feed_type} feed unavailable for game_id={opta_match_id}")
    return parse_jsonp_payload(raw_payload)


def build_match_url(match_ref: MatchRef) -> str:
    if match_ref.current_match_id:
        if match_ref.match_slug:
            return f"{BASE_URL}/mac/{match_ref.match_slug}/{match_ref.current_match_id}"
        return f"{BASE_URL}/mac/{match_ref.current_match_id}"
    return build_archive_match_url(match_ref)


def build_match_plus_url(match_ref: MatchRef) -> str:
    if match_ref.current_match_id:
        if match_ref.match_slug:
            return f"{BASE_URL}/mac/{match_ref.match_slug}/istatistik/{match_ref.current_match_id}"
        return f"{BASE_URL}/mac/istatistik/{match_ref.current_match_id}"
    return build_archive_match_plus_url(match_ref)


def build_iddaa_url(match_ref: MatchRef) -> str:
    if match_ref.current_match_id:
        if match_ref.match_slug:
            return f"{BASE_URL}/mac/{match_ref.match_slug}/iddaa/{match_ref.current_match_id}"
        return f"{BASE_URL}/mac/iddaa/{match_ref.current_match_id}"
    return build_archive_match_url(match_ref)


def build_archive_match_url(match_ref: MatchRef) -> str:
    return f"{ARCHIVE_BASE_URL}/Mac/{match_ref.match_id}/1"


def build_archive_match_plus_url(match_ref: MatchRef) -> str:
    return f"{ARCHIVE_BASE_URL}/Mac-Plus/{match_ref.match_id}/1"


def safe_json_loads(raw_value: str) -> Any:
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError:
        return None


def coerce_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(str(value).strip().replace(",", ".")))
    except (TypeError, ValueError):
        return None


def coerce_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(str(value).strip().replace(",", "."))
    except (TypeError, ValueError):
        return None


def normalize_market_outcomes(raw_market: dict[str, Any]) -> list[dict[str, Any]]:
    outcomes = raw_market.get("outcomes")
    if not isinstance(outcomes, list):
        outcome_collection = raw_market.get("outcomeCollection")
        if isinstance(outcome_collection, dict):
            outcomes = list(outcome_collection.values())
    if not isinstance(outcomes, list):
        return []

    normalized: list[dict[str, Any]] = []
    for outcome in outcomes:
        if not isinstance(outcome, dict):
            continue
        name = outcome.get("name") or outcome.get("label") or outcome.get("outcomeName")
        odd = outcome.get("odd") or outcome.get("odds") or outcome.get("price")
        if name is None and odd is None:
            continue
        odd_float = None
        try:
            odd_float = float(str(odd).replace(",", ".")) if odd is not None else None
        except ValueError:
            odd_float = None
        normalized.append(
            {
                "name": str(name) if name is not None else None,
                "odd_text": str(odd) if odd is not None else None,
                "odd": odd_float,
                "dialog_href": None,
            }
        )
    return normalized


def normalize_market(raw_market: dict[str, Any], *, source_module: str) -> dict[str, Any] | None:
    if not isinstance(raw_market, dict):
        return None
    market_name = raw_market.get("name") or raw_market.get("marketName") or raw_market.get("title")
    outcomes = normalize_market_outcomes(raw_market)
    if market_name is None and not outcomes:
        return None
    return {
        "market_code": int(raw_market.get("id")) if raw_market.get("id") is not None and str(raw_market.get("id")).isdigit() else raw_market.get("id"),
        "market_name": str(market_name) if market_name is not None else None,
        "outcomes": outcomes,
    }


def extract_market_collections(settings_payload: Any) -> list[tuple[str, dict[str, Any]]]:
    collections: list[tuple[str, dict[str, Any]]] = []

    def visit(node: Any, trail: str) -> None:
        if isinstance(node, dict):
            market_collection = node.get("marketCollection")
            if isinstance(market_collection, dict):
                for market_id, market_payload in market_collection.items():
                    if isinstance(market_payload, dict):
                        collections.append((trail or "marketCollection", {"id": market_id, **market_payload}))
            for key, value in node.items():
                next_trail = f"{trail}.{key}" if trail else str(key)
                visit(value, next_trail)
        elif isinstance(node, list):
            for index, item in enumerate(node):
                next_trail = f"{trail}[{index}]"
                visit(item, next_trail)

    visit(settings_payload, "")
    return collections


def extract_iddaa_markets(page_html: str) -> list[dict[str, Any]]:
    soup = BeautifulSoup(page_html, "html.parser")
    markets: list[dict[str, Any]] = []
    seen_keys: set[tuple[str | None, str | None, tuple[tuple[str | None, str | None], ...]]] = set()

    for node in soup.select("[data-module='iddaa/markets']"):
        raw_settings = node.get("data-settings")
        if not raw_settings:
            continue
        settings = safe_json_loads(raw_settings)
        if settings is None:
            continue
        for source_module, raw_market in extract_market_collections(settings):
            normalized = normalize_market(raw_market, source_module=source_module)
            if normalized is None:
                continue
            outcome_signature = tuple((item.get("name"), item.get("odd")) for item in normalized["outcomes"])
            dedupe_key = (normalized.get("market_id"), normalized.get("market_name"), outcome_signature)
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)
            markets.append(normalized)
    return markets


def strip_text(node: Any) -> str | None:
    if node is None:
        return None
    text = node.get_text(" ", strip=True)
    return text or None


def parse_page_meta(html: str) -> dict[str, str | None]:
    if not html:
        return {"title": None, "description": None, "canonical_url": None}
    soup = BeautifulSoup(html, "html.parser")
    title_node = soup.title
    description_node = soup.select_one("meta[name='description']")
    canonical_node = soup.select_one("link[rel='canonical']")
    return {
        "title": title_node.get_text(strip=True) if title_node else None,
        "description": description_node.get("content") if description_node and description_node.get("content") else None,
        "canonical_url": canonical_node.get("href") if canonical_node and canonical_node.get("href") else None,
    }


def parse_competition_from_page(html: str) -> dict[str, Any]:
    if not html:
        return {}
    soup = BeautifulSoup(html, "html.parser")
    competition_link = soup.select_one(".p0c-soccer-match-details-header__competition-link")
    if not competition_link:
        return {}
    href = competition_link.get("href")
    season_id = href.rstrip("/").split("/")[-1] if href else None
    return {
        "name": strip_text(competition_link),
        "url": href,
        "season_id": season_id,
    }


def parse_venue_from_page(html: str) -> dict[str, Any]:
    if not html:
        return {}
    normalized = " ".join(BeautifulSoup(html, "html.parser").get_text(" ", strip=True).split())
    match = re.search(r"\|\s+(.*?)\s+\((\d+)\)\s+\|", normalized)
    if not match:
        match = re.search(r"\|\s+(.*?)\s+\|\s+\d{2}\.\d{2}\.\d{4}", normalized)
    if not match:
        return {}
    venue_payload: dict[str, Any] = {"name": match.group(1).strip()}
    if match.lastindex and match.lastindex >= 2 and match.group(2):
        venue_payload["capacity_text"] = match.group(2)
        try:
            venue_payload["capacity"] = int(match.group(2))
        except ValueError:
            pass
    return venue_payload


def parse_tabs_from_page(html: str) -> list[dict[str, str]]:
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    tabs: list[dict[str, str]] = []
    seen: set[str] = set()
    for anchor in soup.find_all("a", href=True):
        href = anchor.get("href")
        label = strip_text(anchor)
        if not href or not label or href in seen:
            continue
        if "/mac/" not in href:
            continue
        tabs.append({"label": label, "url": href})
        seen.add(href)
    return tabs


def absolute_url(href: str | None, *, base_url: str = ARCHIVE_BASE_URL) -> str | None:
    if not href:
        return None
    if href.startswith("http://") or href.startswith("https://"):
        return href
    if href.startswith("//"):
        return f"https:{href}"
    if href.startswith("/"):
        return f"{base_url}{href}"
    return f"{base_url}/{href.lstrip('/')}"


def parse_archive_match_info(html: str) -> dict[str, Any]:
    if not html:
        return {}
    soup = BeautifulSoup(html, "html.parser")
    info: dict[str, Any] = {}

    match_time = soup.select_one(".mac-time, #dvStatusText")
    if match_time:
        info["status_text"] = strip_text(match_time)

    venue_link = soup.select_one("a[href*='/Stadyum/']")
    if venue_link:
        venue_href = absolute_url(venue_link.get("href"))
        venue_id_match = re.search(r"/Stadyum/(\d+)/", venue_href or "")
        info["venue"] = {
            "name": strip_text(venue_link),
            "url": venue_href,
            "venue_id": int(venue_id_match.group(1)) if venue_id_match else None,
        }

    referee_links = soup.select("a[href*='/Hakem/']")
    officials: list[dict[str, Any]] = []
    if referee_links:
        primary = referee_links[0]
        href = absolute_url(primary.get("href"))
        referee_id_match = re.search(r"/Hakem/(\d+)/", href or "")
        officials.append(
            {
                "role": "referee",
                "name": strip_text(primary),
                "url": href,
                "referee_id": int(referee_id_match.group(1)) if referee_id_match else None,
            }
        )
        for assistant in referee_links[1:]:
            href = absolute_url(assistant.get("href"))
            referee_id_match = re.search(r"/Hakem/(\d+)/", href or "")
            officials.append(
                {
                    "role": "assistant_referee",
                    "name": strip_text(assistant),
                    "url": href,
                    "referee_id": int(referee_id_match.group(1)) if referee_id_match else None,
                }
            )
    if officials:
        info["officials"] = officials

    broadcasts: list[dict[str, Any]] = []
    for text_node in soup.find_all(string=re.compile(r"beIN|S Sports|TRT|A Spor|EXXEN|TV8", re.IGNORECASE)):
        text = " ".join(str(text_node).split())
        if not text:
            continue
        if any(item.get("name") == text for item in broadcasts):
            continue
        broadcasts.append({"name": text, "url": None})
    if broadcasts:
        info["broadcasts"] = broadcasts

    home_coach = soup.select_one("a[href*='/Antrenor/'], a[href*='/Trainer/']")
    if home_coach:
        info["home_coach"] = strip_text(home_coach)
    coach_links = soup.select("a[href*='/Antrenor/'], a[href*='/Trainer/']")
    if len(coach_links) >= 2:
        info["away_coach"] = strip_text(coach_links[1])

    return info


def parse_archive_match_stats(html: str) -> dict[str, dict[str, Any]]:
    if not html:
        return {"genel": {}, "hucum": {}, "savunma": {}}
    soup = BeautifulSoup(html, "html.parser")
    stats: dict[str, dict[str, Any]] = {"genel": {}, "hucum": {}, "savunma": {}}
    category_map = {
        "Topla Oynama": "genel",
        "Başarılı Paslar": "genel",
        "Pas Başarı(%)": "genel",
        "Toplam Şut": "hucum",
        "İsabetli Şut": "hucum",
        "Korner": "hucum",
        "Orta": "hucum",
        "Ofsayt": "hucum",
        "Faul": "savunma",
    }
    for row in soup.select(".match-statistics-rows, .match-statistics-rows-2"):
        value_nodes = row.select(".team-1-statistics-text, .team-2-statistics-text")
        title_node = row.select_one(".statistics-title-text")
        if len(value_nodes) != 2 or title_node is None:
            continue
        stat_name = strip_text(title_node)
        if not stat_name:
            continue
        bucket = category_map.get(stat_name, "genel")
        stats[bucket][stat_name] = {
            "home": strip_text(value_nodes[0]),
            "away": strip_text(value_nodes[1]),
        }
    return stats


def parse_archive_standings(html: str) -> dict[str, Any]:
    if not html:
        return {"competition_name": None, "competition_url": None, "rows": []}
    soup = BeautifulSoup(html, "html.parser")
    container = soup.select_one("#dvStanding .card")
    if container is None:
        return {"competition_name": None, "competition_url": None, "rows": []}
    title_link = container.select_one(".standing-title a")
    rows: list[dict[str, Any]] = []
    for row in container.select("table.list-table tr.row"):
        cells = row.find_all("td")
        if len(cells) < 8:
            continue
        team_link = row.select_one("a[href*='/Takim/']")
        if team_link is None:
            continue
        href = absolute_url(team_link.get("href"))
        team_id_match = re.search(r"/Takim/(\d+)/", href or "")
        rows.append(
            {
                "position": int(strip_text(cells[0]) or 0),
                "team_name": strip_text(team_link),
                "team_url": href,
                "team_id": int(team_id_match.group(1)) if team_id_match else None,
                "played": int(strip_text(cells[3]) or 0),
                "wins": int(strip_text(cells[4]) or 0),
                "draws": int(strip_text(cells[5]) or 0),
                "losses": int(strip_text(cells[6]) or 0),
                "points": int(strip_text(cells[7]) or 0),
            }
        )
    return {
        "competition_name": strip_text(title_link),
        "competition_url": absolute_url(title_link.get("href")) if title_link else None,
        "rows": rows,
    }


def parse_archive_other_matches(html: str) -> list[dict[str, Any]]:
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    result: list[dict[str, Any]] = []
    for row in soup.select("#dvOtherMatches tr.row"):
        cells = row.find_all("td")
        if len(cells) < 4:
            continue
        home_link = cells[1].select_one("a")
        score_link = cells[2].select_one("a")
        away_link = cells[3].select_one("a")
        if home_link is None or score_link is None or away_link is None:
            continue
        match_href = absolute_url(score_link.get("href"))
        match_id_match = re.search(r"/Mac/(\d+)/", match_href or "")
        result.append(
            {
                "status": strip_text(cells[0]),
                "home_team": strip_text(home_link),
                "away_team": strip_text(away_link),
                "score": strip_text(score_link),
                "match_url": match_href,
                "match_id": match_id_match.group(1) if match_id_match else None,
            }
        )
    return result


def parse_opta_widget_config(node: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key in (
        "widget",
        "template",
        "competition",
        "season",
        "match",
        "side",
        "show_maps",
        "show_graphics",
        "show_team_sheets",
        "show_subs",
        "preselected_event",
        "title",
        "stats_categories",
    ):
        if node.get(key) is not None:
            payload[key] = node.get(key)
    return payload


def parse_stats_categories(raw_value: str | None) -> dict[str, list[str]]:
    if not raw_value:
        return {}
    result: dict[str, list[str]] = {}
    for group in raw_value.split("$"):
        label, _, values = group.partition("|")
        label = label.strip()
        if not label:
            continue
        result[label] = [item.strip() for item in values.split(",") if item.strip()]
    return result


def parse_archive_match_plus(html: str) -> dict[str, Any]:
    if not html:
        return {}
    soup = BeautifulSoup(html, "html.parser")
    tabs = [strip_text(node) for node in soup.select("#mac-tabbed-widgets > ul > li")]
    mac_team_systems = [strip_text(node) for node in soup.select(".mac-team-system")]
    fragments: dict[str, str] = {}
    widget_configs: list[dict[str, Any]] = []
    for script in soup.select("script[type='text/template'][id^='tmpl-widget-']"):
        template_id = script.get("id")
        html_fragment = script.decode_contents().strip()
        if template_id and html_fragment:
            fragments[template_id] = html_fragment
            fragment_soup = BeautifulSoup(html_fragment, "html.parser")
            for node in fragment_soup.select("opta-widget"):
                widget_configs.append(parse_opta_widget_config(node))
    widget_names = [config.get("widget") for config in widget_configs if config.get("widget")]
    stats_widget = next((config for config in widget_configs if config.get("widget") == "matchstats" and config.get("stats_categories")), None)
    sample = next((config for config in widget_configs if config.get("competition") and config.get("season") and config.get("match")), {})

    return {
        "opta_identifiers": {
            "match_id": sample.get("match"),
            "competition_id": sample.get("competition"),
            "season_year": sample.get("season"),
            "widget_names": widget_names,
            "stats_categories": parse_stats_categories(stats_widget.get("stats_categories")) if stats_widget else {},
            "stats_categories_source": "archive_match_plus_opta_widgets",
        },
        "widgets": widget_configs,
        "tabs": tabs,
        "team_systems": mac_team_systems,
        "mac_page_fragments": fragments,
        "squads_available": any(config.get("show_team_sheets") == "true" for config in widget_configs),
    }


def ensure_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def opta_uid_number(value: Any) -> int | None:
    if value is None:
        return None
    match = re.search(r"(\d+)", str(value))
    if not match:
        return None
    return int(match.group(1))


def coerce_number(value: Any) -> int | float | str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    if re.fullmatch(r"-?\d+\.\d+", text):
        return float(text)
    return text


def opta_stat_map(raw_stats: Any) -> dict[str, int | float | str | None]:
    stats: dict[str, int | float | str | None] = {}
    for item in ensure_list(raw_stats):
        if not isinstance(item, dict):
            continue
        attrs = item.get("@attributes") if isinstance(item.get("@attributes"), dict) else {}
        stat_type = attrs.get("Type")
        if not stat_type:
            continue
        stats[str(stat_type)] = coerce_number(item.get("@value"))
    return stats


def opta_full_name(person_name: Any) -> str | None:
    if not isinstance(person_name, dict):
        return None
    known = person_name.get("Known")
    if known:
        return str(known)
    first = str(person_name.get("First") or "").strip()
    last = str(person_name.get("Last") or "").strip()
    full = " ".join(part for part in (first, last) if part)
    return full or None


def opta_role_name(role: str | None) -> str:
    normalized = str(role or "").strip().lower()
    role_map = {
        "main": "referee",
        "assistant referee 1": "assistant_referee",
        "assistant referee 2": "assistant_referee",
        "fourth official": "fourth_official",
        "video assistant referee": "var",
        "assistant var official": "assistant_var",
    }
    return role_map.get(normalized, normalized.replace(" ", "_") or "official")


def extract_f9_document(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    soccer_feed = payload.get("SoccerFeed")
    if not isinstance(soccer_feed, dict):
        return {}
    soccer_document = soccer_feed.get("SoccerDocument")
    return soccer_document if isinstance(soccer_document, dict) else {}


def build_f9_team_lookup(soccer_document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    teams: dict[str, dict[str, Any]] = {}
    for raw_team in ensure_list(soccer_document.get("Team")):
        if not isinstance(raw_team, dict):
            continue
        attrs = raw_team.get("@attributes") if isinstance(raw_team.get("@attributes"), dict) else {}
        team_ref = attrs.get("uID")
        if not team_ref:
            continue
        players: dict[str, dict[str, Any]] = {}
        for raw_player in ensure_list(raw_team.get("Player")):
            if not isinstance(raw_player, dict):
                continue
            player_attrs = raw_player.get("@attributes") if isinstance(raw_player.get("@attributes"), dict) else {}
            player_ref = player_attrs.get("uID")
            if not player_ref:
                continue
            players[str(player_ref)] = {
                "player_id": opta_uid_number(player_ref),
                "name": opta_full_name(raw_player.get("PersonName")),
                "position": player_attrs.get("Position"),
            }
        official = raw_team.get("TeamOfficial") if isinstance(raw_team.get("TeamOfficial"), dict) else {}
        teams[str(team_ref)] = {
            "team_id": opta_uid_number(team_ref),
            "name": raw_team.get("Name"),
            "official_name": raw_team.get("Official_name"),
            "short_name": raw_team.get("Short_name"),
            "coach_name": opta_full_name(official.get("PersonName")),
            "coach_id": opta_uid_number((official.get("@attributes") or {}).get("uID")) if isinstance(official.get("@attributes"), dict) else None,
            "players": players,
        }
    return teams


def normalize_f9_player(raw_player: dict[str, Any], team_lookup: dict[str, Any]) -> dict[str, Any]:
    attrs = raw_player.get("@attributes") if isinstance(raw_player.get("@attributes"), dict) else {}
    player_ref = attrs.get("PlayerRef")
    lookup = team_lookup.get(str(player_ref), {}) if player_ref else {}
    stats = opta_stat_map(raw_player.get("Stat"))
    return {
        "player_id": lookup.get("player_id") or opta_uid_number(player_ref),
        "player_ref": player_ref,
        "player_name": lookup.get("name"),
        "position": attrs.get("Position") or lookup.get("position"),
        "shirt_number": coerce_number(attrs.get("ShirtNumber")),
        "status": attrs.get("Status"),
        "minutes_played": stats.get("mins_played"),
        "stats": stats,
    }


def normalize_f9_officials(match_data: dict[str, Any]) -> list[dict[str, Any]]:
    officials: list[dict[str, Any]] = []
    match_official = match_data.get("MatchOfficial") if isinstance(match_data.get("MatchOfficial"), dict) else {}
    if match_official:
        attrs = match_official.get("@attributes") if isinstance(match_official.get("@attributes"), dict) else {}
        official_data = match_official.get("OfficialData") if isinstance(match_official.get("OfficialData"), dict) else {}
        official_ref = official_data.get("OfficialRef") if isinstance(official_data.get("OfficialRef"), dict) else {}
        official_ref_attrs = official_ref.get("@attributes") if isinstance(official_ref.get("@attributes"), dict) else {}
        officials.append(
            {
                "role": opta_role_name(official_ref_attrs.get("Type") or attrs.get("Type")),
                "name": opta_full_name(match_official.get("OfficialName")),
                "url": None,
                "referee_id": opta_uid_number(attrs.get("uID")),
            }
        )
    assistant_officials = match_data.get("AssistantOfficials") if isinstance(match_data.get("AssistantOfficials"), dict) else {}
    for raw_official in ensure_list(assistant_officials.get("AssistantOfficial")):
        if not isinstance(raw_official, dict):
            continue
        attrs = raw_official.get("@attributes") if isinstance(raw_official.get("@attributes"), dict) else {}
        full_name = " ".join(part for part in (attrs.get("FirstName"), attrs.get("LastName")) if part).strip() or None
        officials.append(
            {
                "role": opta_role_name(attrs.get("Type")),
                "name": full_name,
                "url": None,
                "referee_id": opta_uid_number(attrs.get("uID")),
            }
        )
    return [official for official in officials if official.get("name")]


def build_player_performance(players: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    categories: dict[str, dict[str, Any]] = {
        "faul": {},
        "genel": {},
        "hucum": {},
        "kaleci": {},
        "pas": {},
        "savunma": {},
    }
    for player in players:
        player_id = player.get("player_id")
        if player_id is None:
            continue
        stats = player.get("stats") if isinstance(player.get("stats"), dict) else {}
        base = {
            "player_id": player_id,
            "player_name": player.get("player_name"),
            "position": player.get("position"),
            "minutes_played": player.get("minutes_played"),
        }
        categories["genel"][str(player_id)] = {
            **base,
            "stats": {
                key: value
                for key, value in stats.items()
                if key in {"mins_played", "touches", "goals", "goal_assist", "total_pass", "accurate_pass"}
            },
        }
        bucket_map = {
            "faul": ("foul",),
            "kaleci": ("keeper", "save", "goals_conceded", "goal_kicks", "clean_sheet"),
            "pas": ("pass", "cross", "corner"),
            "savunma": ("tackle", "interception", "clearance", "recovery", "duel", "aerial", "block", "poss_won"),
            "hucum": ("shot", "scoring_att", "goal", "assist", "chance", "take_on", "dribble", "carry", "final_third", "pen_area", "offside"),
        }
        for category, needles in bucket_map.items():
            subset = {
                key: value
                for key, value in stats.items()
                if any(needle in key for needle in needles)
            }
            if subset:
                categories[category][str(player_id)] = {**base, "stats": subset}
    return categories


def parse_f9_enrichment(payload: dict[str, Any] | None) -> dict[str, Any]:
    soccer_document = extract_f9_document(payload)
    if not soccer_document:
        return {}
    match_data = soccer_document.get("MatchData") if isinstance(soccer_document.get("MatchData"), dict) else {}
    team_lookup = build_f9_team_lookup(soccer_document)
    home_players: list[dict[str, Any]] = []
    away_players: list[dict[str, Any]] = []
    home_team_meta: dict[str, Any] = {}
    away_team_meta: dict[str, Any] = {}
    for raw_team_data in ensure_list(match_data.get("TeamData")):
        if not isinstance(raw_team_data, dict):
            continue
        attrs = raw_team_data.get("@attributes") if isinstance(raw_team_data.get("@attributes"), dict) else {}
        team_ref = str(attrs.get("TeamRef") or "")
        lineup = raw_team_data.get("PlayerLineUp") if isinstance(raw_team_data.get("PlayerLineUp"), dict) else {}
        players = [
            normalize_f9_player(raw_player, team_lookup.get(team_ref, {}).get("players", {}))
            for raw_player in ensure_list(lineup.get("MatchPlayer"))
            if isinstance(raw_player, dict)
        ]
        side = str(attrs.get("Side") or "").lower()
        if side == "home":
            home_players = players
            home_team_meta = {key: value for key, value in team_lookup.get(team_ref, {}).items() if key != "players"}
        elif side == "away":
            away_players = players
            away_team_meta = {key: value for key, value in team_lookup.get(team_ref, {}).items() if key != "players"}

    competition = soccer_document.get("Competition") if isinstance(soccer_document.get("Competition"), dict) else {}
    competition_stats = opta_stat_map(competition.get("Stat"))
    match_info = match_data.get("MatchInfo") if isinstance(match_data.get("MatchInfo"), dict) else {}
    venue = soccer_document.get("Venue") if isinstance(soccer_document.get("Venue"), dict) else {}
    all_players = [*home_players, *away_players]
    return {
        "competition": {
            "name": competition.get("Name"),
            "season_id": competition_stats.get("season_id"),
            "season_name": competition_stats.get("season_name"),
            "matchday": competition_stats.get("matchday"),
        },
        "venue": {
            "name": venue.get("Name"),
            "venue_id": opta_uid_number((venue.get("@attributes") or {}).get("uID")) if isinstance(venue.get("@attributes"), dict) else None,
            "attendance": coerce_number(match_info.get("Attendance")),
        },
        "officials": normalize_f9_officials(match_data),
        "home_team": {
            **home_team_meta,
            "players": home_players,
        },
        "away_team": {
            **away_team_meta,
            "players": away_players,
        },
        "player_performance": build_player_performance(all_players),
    }

def format_capacity_text(capacity: int | None) -> str | None:
    if capacity is None:
        return None
    return f"{capacity:,}".replace(",", ".")


def parse_match_header_html(html: str) -> dict[str, Any]:
    if not html:
        return {}
    soup = BeautifulSoup(html, "html.parser")
    detail: dict[str, Any] = {}
    score_node = soup.select_one(".widget-scoreboard-score")
    if score_node:
        detail["score"] = strip_text(score_node)
    venue_node = soup.select_one("[data-testid='venue'], .widget-scoreboard-venue")
    if venue_node:
        detail["venue_text"] = strip_text(venue_node)
    competition_node = soup.select_one(".widget-scoreboard-subtitle, .widget-scoreboard-competition")
    if competition_node:
        detail["competition_text"] = strip_text(competition_node)
    date_node = soup.select_one("time, .widget-scoreboard-date")
    if date_node:
        detail["date_text"] = strip_text(date_node)
    return detail


def team_side_code(value: Any) -> int | None:
    if isinstance(value, int):
        return value if value in (1, 2) else None
    text = str(value or "").strip().lower()
    if text in {"home", "left", "1"}:
        return 1
    if text in {"away", "right", "2"}:
        return 2
    return None


def parse_player_id(raw_event: dict[str, Any]) -> int | None:
    candidate = raw_event.get("playerId")
    if candidate is not None:
        try:
            return int(candidate)
        except (TypeError, ValueError):
            pass
    player_url = raw_event.get("playerUrl")
    if player_url:
        match = re.search(r"/(\d+)(?:/|$)", str(player_url))
        if match:
            return int(match.group(1))
    return None


def normalize_key_event(raw_event: dict[str, Any]) -> dict[str, Any] | None:
    minute = raw_event.get("timeMin")
    if minute is None:
        return None
    try:
        minute_int = int(minute)
    except (TypeError, ValueError):
        return None
    second_value = raw_event.get("seconds")
    try:
        second_int = int(second_value) if second_value is not None else 0
    except (TypeError, ValueError):
        second_int = 0
    return {
        "team_side": team_side_code(raw_event.get("position")),
        "minute": minute_int,
        "player_id": parse_player_id(raw_event),
        "player_name": raw_event.get("playerName"),
        "event_type": raw_event.get("type"),
        "details": {
            "second": second_int,
            "sub_type": raw_event.get("subType"),
            "assist_player_name": raw_event.get("assistPlayerName"),
            "score": raw_event.get("score"),
            "period_id": raw_event.get("periodId"),
            "qualifiers": [str(raw_event.get("periodId"))] if raw_event.get("periodId") is not None else [],
            "playerUrl": raw_event.get("playerUrl"),
            "assistPlayerUrl": raw_event.get("assistPlayerUrl"),
            "description": raw_event.get("description"),
            "subtype": raw_event.get("subType"),
            "is_goal": str(raw_event.get("type")).lower() == "goal",
        },
    }


def normalize_key_events_payload(payload: dict[str, Any]) -> dict[str, Any]:
    data = payload.get("data") or {}
    key_events = data.get("keyEvents")
    events: list[dict[str, Any]] = []
    if isinstance(key_events, list):
        for raw_event in key_events:
            if not isinstance(raw_event, dict):
                continue
            normalized = normalize_key_event(raw_event)
            if normalized is not None:
                events.append(normalized)
    return {
        "events": events,
        "finished_period_ids": data.get("finishedPeriodIds"),
        "match_state": data.get("matchState"),
        "match_start_time": data.get("matchStartTime"),
    }


def build_legacy_match_detail(match_payload: dict[str, Any], parsed_header: dict[str, Any]) -> dict[str, Any]:
    score = match_payload.get("score") or {}
    home_score = score.get("home")
    away_score = score.get("away")
    full_time_score = f"{home_score}-{away_score}" if home_score is not None and away_score is not None else parsed_header.get("score")
    half_time = score.get("ht") or {}
    half_time_score = None
    if isinstance(half_time, dict) and half_time.get("home") is not None and half_time.get("away") is not None:
        half_time_score = f"{half_time.get('home')}-{half_time.get('away')}"
    penalty_score = score.get("pen")
    if isinstance(penalty_score, dict) and penalty_score.get("home") is not None and penalty_score.get("away") is not None:
        penalty_score = f"{penalty_score.get('home')}-{penalty_score.get('away')}"
    else:
        penalty_score = None
    return {
        "extra_time_score": None,
        "full_time_score": full_time_score,
        "half_time_score": half_time_score,
        "is_playing": str(match_payload.get("state")) not in {"post", "cancelled"},
        "penalty_score": penalty_score,
        "score": full_time_score,
        "status": match_payload.get("statusBoxContent"),
        "time": parsed_header.get("date_text") or match_payload.get("matchDateText") or match_payload.get("startTime"),
    }


def empty_legacy_sections(competition: dict[str, Any]) -> dict[str, Any]:
    return {
        "opta_identifiers": {
            "match_id": None,
            "competition_id": None,
            "season_year": None,
            "widget_names": [],
            "stats_categories": {"GENEL": [], "SAVUNMA": [], "HUCUM": []},
            "stats_categories_source": "current_fetcher_placeholder",
        },
        "opta_feeds": {
            "f24": None,
            "f24_available": False,
            "f9_available": False,
            "mackolik_raw_available": False,
            "raw": {"f24": None, "f9": None, "mackolik_opta_stats": None},
        },
        "match_stats": {"genel": {}, "hucum": {}, "savunma": {}},
        "standings": {
            "competition_name": competition.get("name"),
            "competition_url": competition.get("url"),
            "rows": [],
        },
        "other_matches": [],
        "player_performance": {
            "faul": {},
            "genel": {},
            "hucum": {},
            "kaleci": {},
            "pas": {},
            "savunma": {},
        },
        "top_performers": {
            "FoulList": [],
            "PassFailList": [],
            "PassList": [],
            "PassSuccList": [],
            "ShotList": [],
            "TackleList": [],
            "TakeonList": [],
        },
        "top_performers_html": [],
        "mac_page_fragments": {},
        "mac_page_stats": {"opta_stats": {}, "rb_stats": {}},
    }


def build_document(
    match_payload: dict[str, Any],
    match_ref: MatchRef,
    header_payload: dict[str, Any],
    key_events_payload: dict[str, Any],
    match_page_html: str,
    match_plus_page_html: str,
    iddaa_page_html: str,
    archive_match_page_html: str,
    archive_match_plus_page_html: str,
    archive_opta_stats_raw: dict[str, Any] | None,
    direct_opta_f24: dict[str, Any] | None,
    direct_opta_f9: dict[str, Any] | None,
    livedata_odds_markets: list[dict[str, Any]] | None,
    fetch_errors: list[dict[str, str]],
) -> dict[str, Any]:
    header_data = header_payload.get("data") or {}
    key_event_data = normalize_key_events_payload(key_events_payload)
    parsed_header = parse_match_header_html(str(header_data.get("html") or ""))
    match_page_meta = parse_page_meta(match_page_html)
    match_plus_meta = parse_page_meta(match_plus_page_html)
    competition_from_page = parse_competition_from_page(match_page_html)
    venue_from_page = parse_venue_from_page(match_page_html)
    tabs_from_page = parse_tabs_from_page(match_page_html)
    archive_match_info = parse_archive_match_info(archive_match_page_html)
    archive_match_stats = parse_archive_match_stats(archive_match_page_html)
    archive_standings = parse_archive_standings(archive_match_page_html)
    archive_other_matches = parse_archive_other_matches(archive_match_page_html)
    archive_plus = parse_archive_match_plus(archive_match_plus_page_html)
    f9_enrichment = parse_f9_enrichment(direct_opta_f9)

    competition = {
        "name": competition_from_page.get("name") or f9_enrichment.get("competition", {}).get("name") or match_payload.get("competitionName") or match_payload.get("longName"),
        "url": competition_from_page.get("url") or archive_standings.get("competition_url"),
        "season_id": competition_from_page.get("season_id") or archive_plus.get("opta_identifiers", {}).get("season_year") or f9_enrichment.get("competition", {}).get("season_id"),
        "season_name": f9_enrichment.get("competition", {}).get("season_name"),
        "matchday": f9_enrichment.get("competition", {}).get("matchday"),
    }

    score = match_payload.get("score") or {}
    home_team = match_payload.get("homeTeam") or {}
    away_team = match_payload.get("awayTeam") or {}
    capacity = venue_from_page.get("capacity")
    legacy_sections = empty_legacy_sections(competition)
    match_detail = build_legacy_match_detail(match_payload, parsed_header)
    match_data_raw = {
        "match_id": match_ref.match_id,
        "seq": 0,
        "home": home_team.get("name"),
        "away": away_team.get("name"),
        "d": match_detail,
        "h": f9_enrichment.get("home_team", {}).get("players", []),
        "a": f9_enrichment.get("away_team", {}).get("players", []),
        "e": key_event_data.get("events", []),
        "sv": [],
        "live_payload": match_payload,
        "key_events_payload": key_events_payload,
    }

    document = {
        "match_id": match_ref.match_id,
        "source": {
            "provider": "mackolik",
            "page_type": "Mac-Plus",
        },
        "source_url": build_match_plus_url(match_ref),
        "canonical_url": match_plus_meta.get("canonical_url") or build_match_plus_url(match_ref),
        "title": match_plus_meta.get("title"),
        "description": match_plus_meta.get("description"),
        "match_date_text": parsed_header.get("date_text") or match_payload.get("matchDateText") or match_payload.get("startTime"),
        "fetched_at": datetime.now(UTC).isoformat(),
        "competition": competition,
        "home_team": {
            "team_id": str(home_team.get("id")) if home_team.get("id") is not None else None,
            "name": home_team.get("name") or f9_enrichment.get("home_team", {}).get("name"),
            "url": f"{BASE_URL}/takim/{home_team.get('slug')}/ma%C3%A7lar/{home_team.get('id')}" if home_team.get("slug") and home_team.get("id") else None,
            "recent_form": home_team.get("recentForm"),
            "logo_url": None,
            "coach_name": archive_match_info.get("home_coach") or f9_enrichment.get("home_team", {}).get("coach_name"),
        },
        "away_team": {
            "team_id": str(away_team.get("id")) if away_team.get("id") is not None else None,
            "name": away_team.get("name") or f9_enrichment.get("away_team", {}).get("name"),
            "url": f"{BASE_URL}/takim/{away_team.get('slug')}/ma%C3%A7lar/{away_team.get('id')}" if away_team.get("slug") and away_team.get("id") else None,
            "recent_form": away_team.get("recentForm"),
            "logo_url": None,
            "coach_name": archive_match_info.get("away_coach") or f9_enrichment.get("away_team", {}).get("coach_name"),
        },
        "score": {
            "home": score.get("home"),
            "away": score.get("away"),
        },
        "status_text": match_payload.get("statusBoxContent"),
        "venue": {
            "name": venue_from_page.get("name") or (archive_match_info.get("venue") or {}).get("name") or f9_enrichment.get("venue", {}).get("name"),
            "url": (archive_match_info.get("venue") or {}).get("url"),
            "venue_id": (archive_match_info.get("venue") or {}).get("venue_id") or f9_enrichment.get("venue", {}).get("venue_id"),
            "capacity_text": format_capacity_text(capacity) or venue_from_page.get("capacity_text") or ((archive_match_info.get("venue") or {}).get("capacity_text")),
            "attendance": f9_enrichment.get("venue", {}).get("attendance"),
        },
        "pages": {
            "mac_plus_url": build_match_plus_url(match_ref),
            "mac_url": build_match_url(match_ref),
            "mac_plus_available": bool(match_plus_page_html),
            "mac_available": bool(match_page_html),
            "archive_mac_plus_url": build_archive_match_plus_url(match_ref),
            "archive_mac_url": build_archive_match_url(match_ref),
            "archive_mac_plus_available": bool(archive_match_plus_page_html),
            "archive_mac_available": bool(archive_match_page_html),
        },
        "mac_page": {
            "source_url": build_match_url(match_ref),
            "canonical_url": match_page_meta.get("canonical_url") or build_match_url(match_ref),
            "title": match_page_meta.get("title"),
            "description": match_page_meta.get("description"),
            "venue": {
                "name": venue_from_page.get("name") or (archive_match_info.get("venue") or {}).get("name") or f9_enrichment.get("venue", {}).get("name"),
                "url": (archive_match_info.get("venue") or {}).get("url"),
                "venue_id": (archive_match_info.get("venue") or {}).get("venue_id") or f9_enrichment.get("venue", {}).get("venue_id"),
            },
            "officials": archive_match_info.get("officials", []) or f9_enrichment.get("officials", []),
            "broadcasts": archive_match_info.get("broadcasts", []),
            "live_cast": {
                "enabled": False,
                "cast_id": None,
                "width": None,
                "token": None,
                "token_available": False,
            },
            "tabs": tabs_from_page or [{"label": label, "url": None} for label in archive_plus.get("tabs", []) if label],
            "squads_available": {
                "home": bool(archive_plus.get("squads_available")),
                "away": bool(archive_plus.get("squads_available")),
            },
        },
        "match_data": {
            "source_url": build_match_url(match_ref),
            "sequence": 0,
            "home_team_name": home_team.get("name") or f9_enrichment.get("home_team", {}).get("name"),
            "away_team_name": away_team.get("name") or f9_enrichment.get("away_team", {}).get("name"),
            "detail": match_detail,
            "home_players": f9_enrichment.get("home_team", {}).get("players", []),
            "away_players": f9_enrichment.get("away_team", {}).get("players", []),
            "events": key_event_data.get("events", []),
            "squad_values": [],
            "raw": match_data_raw,
        },
        "odds_markets": extract_iddaa_markets(iddaa_page_html or match_page_html) or (livedata_odds_markets or []),
        "fetch_errors": fetch_errors,
        **legacy_sections,
    }
    if archive_plus.get("opta_identifiers"):
        document["opta_identifiers"] = archive_plus["opta_identifiers"]
    if archive_match_stats:
        document["match_stats"] = archive_match_stats
    if archive_standings.get("rows"):
        document["standings"] = archive_standings
    if archive_other_matches:
        document["other_matches"] = archive_other_matches
    if archive_opta_stats_raw:
        document["opta_feeds"]["mackolik_raw_available"] = True
        document["opta_feeds"]["raw"]["mackolik_opta_stats"] = archive_opta_stats_raw
        document["top_performers"] = archive_opta_stats_raw
    if direct_opta_f24:
        document["opta_feeds"]["f24_available"] = True
        document["opta_feeds"]["f24"] = {
            "source": "direct_opta_auth",
            "match_id": archive_plus.get("opta_identifiers", {}).get("match_id"),
        }
        document["opta_feeds"]["raw"]["f24"] = direct_opta_f24
    if direct_opta_f9:
        document["opta_feeds"]["f9_available"] = True
        document["opta_feeds"]["raw"]["f9"] = direct_opta_f9
        document["player_performance"] = f9_enrichment.get("player_performance", document["player_performance"])
    if archive_plus.get("mac_page_fragments"):
        document["mac_page_fragments"] = archive_plus["mac_page_fragments"]
    if archive_plus.get("widgets"):
        document["mac_page_stats"] = {
            "opta_stats": {
                "widgets": archive_plus["widgets"],
                "tabs": archive_plus.get("tabs", []),
                "team_systems": archive_plus.get("team_systems", []),
            },
            "rb_stats": archive_match_stats,
        }
    return document


def upsert_documents(collection: Any, documents: list[dict[str, Any]], batch_size: int) -> tuple[int, int]:
    inserted_or_updated = 0
    matched = 0
    for start in range(0, len(documents), batch_size):
        batch = documents[start : start + batch_size]
        operations = [
            UpdateOne({"match_id": document["match_id"]}, {"$set": document}, upsert=True)
            for document in batch
        ]
        if not operations:
            continue
        result = collection.bulk_write(operations, ordered=False)
        inserted_or_updated += result.upserted_count + result.modified_count
        matched += result.matched_count
    return inserted_or_updated, matched


def filter_existing_match_rows(collection: Any, match_rows: list[Any]) -> tuple[list[Any], int]:
    candidate_ids = [match_id for match_row in match_rows if (match_id := livedata_row_match_id(match_row))]
    if not candidate_ids:
        return match_rows, 0
    existing_ids = set(collection.distinct("match_id", {"match_id": {"$in": candidate_ids}}))
    if not existing_ids:
        return match_rows, 0
    filtered_rows = [match_row for match_row in match_rows if livedata_row_match_id(match_row) not in existing_ids]
    return filtered_rows, len(match_rows) - len(filtered_rows)


def fetch_match_document(
    target_date: date,
    match_row: list[Any],
    *,
    timeout: int,
) -> dict[str, Any] | None:
    if not isinstance(match_row, list):
        return None
    normalized_match = parse_livedata_match_row(match_row, target_date)
    if normalized_match is None:
        return None
    match_ref = match_ref_from_livescore(normalized_match, target_date)
    if match_ref is None:
        return None

    livedata_odds_markets = odds_markets_from_livedata(normalized_match)
    fetch_errors: list[dict[str, str]] = []
    session = create_session()
    try:
        header_payload: dict[str, Any] = {}
        key_events_payload: dict[str, Any] = {}
        match_page_html = ""
        match_plus_page_html = ""
        iddaa_page_html = ""
        if match_ref.current_match_id:
            try:
                header_payload = fetch_match_header(session, match_ref, timeout)
            except FatalRateLimitError:
                raise
            except requests.RequestException as exc:
                fetch_errors.append({"step": "match_header", "error": str(exc)})
            try:
                key_events_payload = fetch_key_events(session, match_ref, timeout)
            except FatalRateLimitError:
                raise
            except requests.RequestException as exc:
                fetch_errors.append({"step": "key_events", "error": str(exc)})
            try:
                match_page_html = fetch_match_page_html(session, match_ref, timeout)
            except FatalRateLimitError:
                raise
            except requests.RequestException as exc:
                fetch_errors.append({"step": "match_page", "error": str(exc)})
            try:
                match_plus_page_html = fetch_match_plus_page_html(session, match_ref, timeout)
            except FatalRateLimitError:
                raise
            except requests.RequestException as exc:
                fetch_errors.append({"step": "match_plus_page", "error": str(exc)})
            try:
                iddaa_page_html = fetch_iddaa_page_html(session, match_ref, timeout)
            except FatalRateLimitError:
                raise
            except requests.RequestException as exc:
                fetch_errors.append({"step": "iddaa_page", "error": str(exc)})
        try:
            archive_match_page_html = fetch_archive_match_page_html(session, match_ref, timeout)
        except FatalRateLimitError:
            raise
        except requests.RequestException as exc:
            archive_match_page_html = ""
            fetch_errors.append({"step": "archive_match_page", "error": str(exc)})
        try:
            archive_match_plus_page_html = fetch_archive_match_plus_page_html(session, match_ref, timeout)
        except FatalRateLimitError:
            raise
        except requests.RequestException as exc:
            archive_match_plus_page_html = ""
            fetch_errors.append({"step": "archive_match_plus_page", "error": str(exc)})
        archive_plus = parse_archive_match_plus(archive_match_plus_page_html)
        archive_opta_stats_raw = None
        direct_opta_f24 = None
        direct_opta_f9 = None
        try:
            archive_opta_stats_raw = fetch_archive_opta_stats_raw(session, match_ref, timeout)
        except FatalRateLimitError:
            raise
        except (requests.RequestException, RuntimeError) as exc:
            fetch_errors.append({"step": "archive_opta_stats_raw", "error": str(exc)})
        opta_match_id = (archive_plus.get("opta_identifiers") or {}).get("match_id")
        if opta_match_id:
            try:
                direct_opta_f24 = fetch_direct_opta_feed(
                    session,
                    match_ref,
                    opta_match_id=str(opta_match_id),
                    feed_type="f24",
                    timeout=timeout,
                )
            except FatalRateLimitError:
                raise
            except (requests.RequestException, RuntimeError) as exc:
                fetch_errors.append({"step": "direct_opta_f24", "error": str(exc)})
            try:
                direct_opta_f9 = fetch_direct_opta_feed(
                    session,
                    match_ref,
                    opta_match_id=str(opta_match_id),
                    feed_type="f9",
                    timeout=timeout,
                )
            except FatalRateLimitError:
                raise
            except (requests.RequestException, RuntimeError) as exc:
                fetch_errors.append({"step": "direct_opta_f9", "error": str(exc)})
        return build_document(
            match_payload=normalized_match,
            match_ref=match_ref,
            header_payload=header_payload,
            key_events_payload=key_events_payload,
            match_page_html=match_page_html,
            match_plus_page_html=match_plus_page_html,
            iddaa_page_html=iddaa_page_html,
            archive_match_page_html=archive_match_page_html,
            archive_match_plus_page_html=archive_match_plus_page_html,
            archive_opta_stats_raw=archive_opta_stats_raw,
            direct_opta_f24=direct_opta_f24,
            direct_opta_f9=direct_opta_f9,
            livedata_odds_markets=livedata_odds_markets,
            fetch_errors=fetch_errors,
        )
    finally:
        session.close()


def fetch_for_date(session: requests.Session, target_date: date, timeout: int) -> list[dict[str, Any]]:
    return fetch_for_date_with_limit(
        session,
        target_date,
        timeout,
        collection=None,
        continue_fetch=False,
        limit_matches=None,
        max_workers=1,
    )


def fetch_for_date_with_limit(
    session: requests.Session,
    target_date: date,
    timeout: int,
    *,
    collection: Any | None,
    continue_fetch: bool,
    limit_matches: int | None,
    max_workers: int,
) -> list[dict[str, Any]]:
    payload = fetch_livescores(session, target_date, timeout)
    matches = payload.get("m") if isinstance(payload.get("m"), list) else []
    raw_match_items = list(matches)
    match_items = [match_row for match_row in raw_match_items if isinstance(match_row, list) and is_finished_livedata_row(match_row)]
    skipped_unfinished = len(raw_match_items) - len(match_items)
    if skipped_unfinished:
        print(f"{target_date.isoformat()} skipped unfinished matches: {skipped_unfinished}")
    if continue_fetch and collection is not None and match_items:
        match_items, skipped_existing = filter_existing_match_rows(collection, match_items)
        if skipped_existing:
            print(f"{target_date.isoformat()} skipped existing matches: {skipped_existing}")
    if limit_matches is not None:
        match_items = match_items[:limit_matches]
    if not match_items:
        return []

    worker_count = min(normalize_max_workers(max_workers), len(match_items))
    if worker_count <= 1:
        documents = [
            document
            for document in progress_iterable(
                (
                    fetch_match_document(
                        target_date,
                        match_row,
                        timeout=timeout,
                    )
                    for match_row in match_items
                ),
                total=len(match_items),
                desc=f"Matches {target_date.isoformat()}",
                unit="match",
            )
            if document is not None
        ]
        return documents

    ordered_results: list[dict[str, Any] | None] = [None] * len(match_items)
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                fetch_match_document,
                target_date,
                match_row,
                timeout=timeout,
            ): index
            for index, match_row in enumerate(match_items)
        }
        completed = progress_iterable(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc=f"Matches {target_date.isoformat()}",
            unit="match",
        )
        for future in completed:
            index = futures[future]
            ordered_results[index] = future.result()
    return [document for document in ordered_results if document is not None]


def main() -> int:
    args = parse_args()
    start_date, end_date = resolve_date_range(args)
    session = create_session()
    client = None
    collection = None
    sampled_documents: list[dict[str, Any]] = []
    total_documents = 0
    total_changed = 0
    total_matched = 0

    try:
        if not args.dry_run or args.continue_fetch:
            client, collection = load_collection()

        target_dates = list(iter_target_dates(start_date, end_date, reverse=args.reverse))
        for target_date in progress_iterable(
            target_dates,
            total=len(target_dates),
            desc="Dates",
            unit="day",
        ):
            remaining = None if args.limit_matches is None else max(args.limit_matches - total_documents, 0)
            documents = fetch_for_date_with_limit(
                session,
                target_date,
                args.timeout,
                collection=collection,
                continue_fetch=args.continue_fetch,
                limit_matches=remaining,
                max_workers=args.max_workers,
            )
            total_documents += len(documents)
            if args.dry_run and len(sampled_documents) < 3:
                sampled_documents.extend(documents[: 3 - len(sampled_documents)])
            elif collection is not None and documents:
                changed, matched = upsert_documents(collection, documents, args.upsert_batch_size)
                total_changed += changed
                total_matched += matched
            print(f"{target_date.isoformat()} fetched matches: {len(documents)}")
            if args.limit_matches is not None and total_documents >= args.limit_matches:
                break

        if args.dry_run:
            print(f"Dry run complete. Prepared {total_documents} documents.")
            for document in sampled_documents:
                print(json.dumps({"match_id": document["match_id"], "market_count": len(document["odds_markets"])}, ensure_ascii=True))
            return 0

        print(f"Fetched documents: {total_documents}")
        print(f"Matched existing documents: {total_matched}")
        print(f"Inserted or modified documents: {total_changed}")
        return 0
    except FatalRateLimitError:
        return 1
    finally:
        if client is not None:
            client.close()
        session.close()


if __name__ == "__main__":
    raise SystemExit(main())
