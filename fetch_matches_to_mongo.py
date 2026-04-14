from __future__ import annotations

import argparse
import os
import sys
from datetime import UTC, date, datetime
from typing import Any

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import PyMongoError
from pymongo.operations import UpdateOne
from requests import HTTPError, Session
from tqdm import tqdm

from sportsradar_reverse import (
    DEFAULT_ACCESS_LEVEL,
    DEFAULT_LANGUAGE,
    DEFAULT_ORIGIN,
    LmtClient,
    LmtConfig,
    SportradarApiClient,
    SportradarApiConfig,
    sport_event_id_to_match_id,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch seasons, ended season schedules, and LMT match data into MongoDB.",
    )
    parser.add_argument(
        "--access-level",
        default=DEFAULT_ACCESS_LEVEL,
        help="Sportradar soccer API access level, for example 'trial' or 'production'.",
    )
    parser.add_argument(
        "--language",
        default=DEFAULT_LANGUAGE,
        help="Language code used for both Sportradar API and LMT requests.",
    )
    parser.add_argument(
        "--origin",
        default=DEFAULT_ORIGIN,
        help="Origin header used while requesting LMT widget endpoints.",
    )
    parser.add_argument(
        "--season-limit",
        type=int,
        default=None,
        help="Optional limit on how many ended seasons to process after fetching all seasons.",
    )
    parser.add_argument(
        "--match-limit",
        type=int,
        default=None,
        help="Optional global limit on how many scheduled matches receive LMT fetches.",
    )
    parser.add_argument(
        "--database",
        default=None,
        help="Optional MongoDB database override. Falls back to URI database or football_analytics.",
    )
    parser.add_argument(
        "--refresh-seasons",
        action="store_true",
        help="Force a fresh seasons API fetch instead of resuming from cached seasons.",
    )
    parser.add_argument(
        "--seasons-collection",
        default="seasons",
        help="Collection used for season documents.",
    )
    parser.add_argument(
        "--schedules-collection",
        default="season_schedules",
        help="Collection used for schedule item documents.",
    )
    parser.add_argument(
        "--matches-collection",
        default="matches",
        help="Collection used for per-match LMT documents.",
    )
    return parser.parse_args()


def get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} not found in .env")
    return value


def get_database(client: MongoClient, explicit_name: str | None):
    if explicit_name:
        return client[explicit_name]

    try:
        return client.get_default_database(default="football_analytics")
    except TypeError:
        try:
            return client.get_default_database()
        except Exception:
            return client["football_analytics"]


def ensure_indexes(
    seasons_collection: Collection,
    schedules_collection: Collection,
    matches_collection: Collection,
) -> None:
    seasons_collection.create_index("season_id", unique=True)
    seasons_collection.create_index("competition_id")
    seasons_collection.create_index("end_date")
    seasons_collection.create_index("is_ended")
    seasons_collection.create_index("meta.fetched")

    schedules_collection.create_index("sport_event_id", unique=True)
    schedules_collection.create_index("season_id")
    schedules_collection.create_index("match_id")
    schedules_collection.create_index("start_time")
    schedules_collection.create_index("status")

    matches_collection.create_index("match_id", unique=True)
    matches_collection.create_index("season_id")
    matches_collection.create_index("sport_event_id")
    matches_collection.create_index("match_date_utc")
    matches_collection.create_index("competition.competition_id")



def is_ended_season(season: dict[str, Any], reference_date: date) -> bool:
    end_date = season.get("end_date")
    return bool(end_date and end_date < reference_date.isoformat())


def limit_items(items: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None:
        return items
    return items[:limit]


def extract_lmt_doc(payload: dict[str, Any]) -> dict[str, Any]:
    docs = payload.get("doc")
    if not isinstance(docs, list) or not docs:
        raise ValueError("LMT payload did not contain a non-empty 'doc' list")
    return docs[0]


def extract_lmt_section(payload: dict[str, Any]) -> dict[str, Any]:
    doc = extract_lmt_doc(payload)
    return {
        "query_url": payload.get("queryUrl"),
        "event": doc.get("event"),
        "dob": doc.get("_dob"),
        "maxage": doc.get("_maxage"),
        "data": doc.get("data"),
    }


def build_season_document(season: dict[str, Any], *, fetched_at: datetime, reference_date: date) -> dict[str, Any]:
    return {
        "season_id": season.get("id"),
        "competition_id": season.get("competition_id"),
        "name": season.get("name"),
        "year": season.get("year"),
        "start_date": season.get("start_date"),
        "end_date": season.get("end_date"),
        "is_ended": is_ended_season(season, reference_date),
        "fetched_at": fetched_at,
        "raw": season,
    }


def load_cached_seasons(seasons_collection: Collection) -> list[dict[str, Any]]:
    seasons = []
    cursor = seasons_collection.find({}, {"_id": 0, "raw": 1})
    for document in cursor:
        raw = document.get("raw")
        if isinstance(raw, dict):
            seasons.append(raw)
    return seasons


def extract_competitor(competitors: list[dict[str, Any]], qualifier: str) -> dict[str, Any]:
    for competitor in competitors:
        if competitor.get("qualifier") == qualifier:
            return competitor
    return {}


def build_schedule_document(
    schedule_item: dict[str, Any],
    *,
    fetched_at: datetime,
) -> dict[str, Any]:
    sport_event = schedule_item.get("sport_event", {})
    context = sport_event.get("sport_event_context", {})
    status = schedule_item.get("sport_event_status", {})
    competitors = sport_event.get("competitors", [])
    sport_event_id = sport_event.get("id")
    match_id = sport_event_id_to_match_id(sport_event_id) if sport_event_id else None

    home = extract_competitor(competitors, "home")
    away = extract_competitor(competitors, "away")

    return {
        "sport_event_id": sport_event_id,
        "match_id": match_id,
        "season_id": context.get("season", {}).get("id"),
        "competition_id": context.get("competition", {}).get("id"),
        "start_time": sport_event.get("start_time"),
        "status": status.get("status"),
        "match_status": status.get("match_status"),
        "home_competitor": home,
        "away_competitor": away,
        "fetched_at": fetched_at,
        "context": {
            "sport": context.get("sport"),
            "category": context.get("category"),
            "competition": context.get("competition"),
            "season": context.get("season"),
            "stage": context.get("stage"),
            "round": context.get("round"),
            "groups": context.get("groups"),
            "venue": sport_event.get("venue"),
            "coverage": sport_event.get("coverage"),
        },
        "raw": schedule_item,
    }


def build_match_document(
    schedule_item: dict[str, Any],
    lmt_bundle: dict[str, dict[str, Any]],
    *,
    fetched_at: datetime,
    access_level: str,
    language: str,
    origin: str,
) -> dict[str, Any]:
    sport_event = schedule_item.get("sport_event", {})
    context = sport_event.get("sport_event_context", {})
    status = schedule_item.get("sport_event_status", {})
    sport_event_id = sport_event.get("id")
    match_id = sport_event_id_to_match_id(sport_event_id)

    timeline = extract_lmt_section(lmt_bundle["match_timeline"])
    info = extract_lmt_section(lmt_bundle["match_info"])
    detailsextended = extract_lmt_section(lmt_bundle["match_detailsextended"])
    phrases = extract_lmt_section(lmt_bundle["match_phrases"])
    squads = extract_lmt_section(lmt_bundle["match_squads"])

    timeline_match = timeline.get("data", {}).get("match", {})
    info_data = info.get("data", {})
    competitors = sport_event.get("competitors", [])

    return {
        "match_id": match_id,
        "sport_event_id": sport_event_id,
        "season_id": context.get("season", {}).get("id"),
        "competition_id": context.get("competition", {}).get("id"),
        "match_date_utc": sport_event.get("start_time", "")[:10] or None,
        "start_time": sport_event.get("start_time"),
        "status": status.get("status"),
        "match_status": status.get("match_status"),
        "fetched_at": fetched_at,
        "source": {
            "fetch_mode": "season_schedule",
            "api_access_level": access_level,
            "language": language,
            "lmt_origin": origin,
            "lmt_endpoints": [
                "match_timeline",
                "match_info",
                "match_detailsextended",
                "match_phrases",
                "match_squads",
            ],
        },
        "competition": {
            "sport": context.get("sport"),
            "category": context.get("category"),
            "competition": context.get("competition"),
            "competition_id": context.get("competition", {}).get("id"),
            "season": context.get("season"),
            "stage": context.get("stage"),
            "round": context.get("round"),
            "groups": context.get("groups"),
            "realcategory": info_data.get("realcategory"),
            "tournament": info_data.get("tournament"),
            "uniquetournament": info_data.get("uniquetournament"),
        },
        "teams": {
            "schedule_competitors": competitors,
            "lmt_teams": timeline_match.get("teams"),
            "jerseys": info_data.get("jerseys"),
        },
        "status_info": {
            "schedule_status": status,
            "lmt_status": timeline_match.get("status"),
            "lmt_matchstatus": timeline_match.get("matchstatus"),
            "result": timeline_match.get("result"),
            "periods": timeline_match.get("periods"),
            "coverage": timeline_match.get("coverage"),
        },
        "venue": {
            "schedule_venue": sport_event.get("venue"),
            "stadium": info_data.get("stadium"),
            "cities": info_data.get("cities"),
        },
        "lmt": {
            "match_timeline": timeline,
            "match_info": info,
            "match_detailsextended": detailsextended,
            "match_phrases": phrases,
            "match_squads": squads,
        },
        "raw": {
            "season_schedule_item": schedule_item,
            "lmt": lmt_bundle,
        },
    }


def upsert_seasons(
    seasons_collection: Collection,
    seasons: list[dict[str, Any]],
    *,
    fetched_at: datetime,
    reference_date: date,
) -> None:
    operations = []
    with tqdm(seasons, desc="Seasons", unit="season", dynamic_ncols=True) as season_bar:
        for season in season_bar:
            document = build_season_document(season, fetched_at=fetched_at, reference_date=reference_date)
            operations.append(
                UpdateOne(
                    {"season_id": document["season_id"]},
                    {
                        "$set": document,
                        "$setOnInsert": {
                            "meta": {
                                "fetched": False,
                            }
                        },
                    },
                    upsert=True,
                )
            )

    if operations:
        seasons_collection.bulk_write(operations, ordered=False)


def ingest_ended_seasons(
    *,
    ended_seasons: list[dict[str, Any]],
    api_client: SportradarApiClient,
    lmt_client: LmtClient,
    seasons_collection: Collection,
    schedules_collection: Collection,
    matches_collection: Collection,
    access_level: str,
    language: str,
    origin: str,
    match_limit: int | None,
    resume: bool,
) -> tuple[int, int, int, list[str], list[int]]:
    fetched_schedule_count = 0
    fetched_match_count = 0
    stored_schedule_count = 0
    failed_seasons: list[str] = []
    failed_matches: list[int] = []
    remaining_match_budget = match_limit

    with tqdm(ended_seasons, desc="Ended seasons", unit="season", dynamic_ncols=True) as season_bar:
        for season in season_bar:
            season_id = season.get("id")
            season_bar.set_postfix(season=season_id, matches=fetched_match_count)
            if not season_id:
                failed_seasons.append("<missing-season-id>")
                continue

            season_document = seasons_collection.find_one(
                {"season_id": season_id},
                {"_id": 0, "meta": 1},
            ) or {}
            season_meta = season_document.get("meta", {})

            if resume and season_meta.get("fetched"):
                schedules = [
                    document["raw"]
                    for document in schedules_collection.find(
                        {"season_id": season_id},
                        {"_id": 0, "raw": 1},
                    ).sort("start_time", 1)
                    if isinstance(document.get("raw"), dict)
                ]

                # Migration may mark a season as fetched before cached schedules exist.
                if not schedules:
                    season_meta = {}

            if not (resume and season_meta.get("fetched")):
                try:
                    schedule_payload = api_client.fetch_season_schedule(season_id)
                except Exception:
                    failed_seasons.append(season_id)
                    continue

                fetched_schedule_count += 1
                schedules = schedule_payload.get("schedules", [])
                schedule_fetched_at = datetime.now(UTC)
                operations = []

                for schedule_item in schedules:
                    schedule_document = build_schedule_document(schedule_item, fetched_at=schedule_fetched_at)
                    operations.append(
                        UpdateOne(
                            {"sport_event_id": schedule_document["sport_event_id"]},
                            {"$set": schedule_document},
                            upsert=True,
                        )
                    )

                if operations:
                    schedules_collection.bulk_write(operations, ordered=False)
                stored_schedule_count += len(schedules)
                seasons_collection.update_one(
                    {"season_id": season_id},
                    {
                        "$set": {
                            "meta.fetched": True,
                            "meta.schedule_item_count": len(schedules),
                            "meta.schedule_fetched_at": schedule_fetched_at,
                        }
                    },
                )

            if remaining_match_budget == 0:
                break

            existing_match_ids = set(
                matches_collection.distinct("match_id", {"season_id": season_id})
            )
            pending_schedule_items = []
            for schedule_item in schedules:
                sport_event_id = schedule_item.get("sport_event", {}).get("id")
                if not sport_event_id:
                    continue
                match_id = sport_event_id_to_match_id(sport_event_id)
                if match_id not in existing_match_ids:
                    pending_schedule_items.append(schedule_item)

            if resume and season_meta.get("fetched") and not pending_schedule_items:
                continue

            schedule_items_for_matches = pending_schedule_items
            if remaining_match_budget is not None:
                schedule_items_for_matches = pending_schedule_items[:remaining_match_budget]

            with tqdm(
                schedule_items_for_matches,
                desc=f"{season_id}",
                unit="match",
                leave=False,
                dynamic_ncols=True,
            ) as match_bar:
                for schedule_item in match_bar:
                    sport_event_id = schedule_item.get("sport_event", {}).get("id")
                    if not sport_event_id:
                        continue

                    try:
                        match_id = sport_event_id_to_match_id(sport_event_id)
                        lmt_bundle = lmt_client.fetch_match_bundle(match_id, include_phrases=True)
                        match_document = build_match_document(
                            schedule_item,
                            lmt_bundle,
                            fetched_at=datetime.now(UTC),
                            access_level=access_level,
                            language=language,
                            origin=origin,
                        )
                        matches_collection.replace_one(
                            {"match_id": match_document["match_id"]},
                            match_document,
                            upsert=True,
                        )
                        fetched_match_count += 1
                        existing_match_ids.add(match_id)
                    except Exception:
                        failed_match_id = sport_event_id_to_match_id(sport_event_id)
                        failed_matches.append(failed_match_id)

                    if remaining_match_budget is not None:
                        remaining_match_budget -= 1
                        if remaining_match_budget <= 0:
                            remaining_match_budget = 0

                    match_bar.set_postfix(stored=fetched_match_count, failed=len(failed_matches))

                    if remaining_match_budget == 0:
                        break

            if remaining_match_budget == 0:
                break

    return (
        fetched_schedule_count,
        stored_schedule_count,
        fetched_match_count,
        failed_seasons,
        failed_matches,
    )


def main() -> int:
    load_dotenv(".env")
    args = parse_args()

    api_key = get_required_env("SPORTRADAR_API_KEY")
    mongo_connection_string = get_required_env("MONGO_CONNECTION_STRING")

    api_session = Session()
    lmt_session = Session()

    api_client = SportradarApiClient(
        SportradarApiConfig(
            api_key=api_key,
            access_level=args.access_level,
            language=args.language,
        ),
        session=api_session,
    )
    lmt_client = LmtClient(
        LmtConfig(
            origin=args.origin,
            language=args.language,
        ),
        session=lmt_session,
    )

    mongo_client = MongoClient(mongo_connection_string, serverSelectionTimeoutMS=20000)

    try:
        mongo_client.admin.command("ping")
        database = get_database(mongo_client, args.database)
        seasons_collection = database[args.seasons_collection]
        schedules_collection = database[args.schedules_collection]
        matches_collection = database[args.matches_collection]
        ensure_indexes(seasons_collection, schedules_collection, matches_collection)

        reference_date = datetime.now(UTC).date()

        if (
            not args.refresh_seasons
            and seasons_collection.count_documents({}) > 0
        ):
            seasons = load_cached_seasons(seasons_collection)
        else:
            seasons_payload = api_client.fetch_seasons()
            seasons = seasons_payload.get("seasons", [])
            seasons_fetched_at = datetime.now(UTC)
            upsert_seasons(
                seasons_collection,
                seasons,
                fetched_at=seasons_fetched_at,
                reference_date=reference_date,
            )

        ended_seasons = [season for season in seasons if is_ended_season(season, reference_date)]
        ended_seasons.sort(key=lambda season: season.get("end_date", ""), reverse=True)
        ended_seasons = limit_items(ended_seasons, args.season_limit)

        (
            fetched_schedule_count,
            stored_schedule_count,
            fetched_match_count,
            failed_seasons,
            failed_matches,
        ) = ingest_ended_seasons(
            ended_seasons=ended_seasons,
            api_client=api_client,
            lmt_client=lmt_client,
            seasons_collection=seasons_collection,
            schedules_collection=schedules_collection,
            matches_collection=matches_collection,
            access_level=args.access_level,
            language=args.language,
            origin=args.origin,
            match_limit=args.match_limit,
            resume=not args.refresh_seasons,
        )
    except (PyMongoError, OSError, ValueError, HTTPError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        mongo_client.close()
        api_session.close()
        lmt_session.close()

    print(
        f"Fetched {len(seasons)} seasons. "
        f"Processed {len(ended_seasons)} ended seasons as of {reference_date.isoformat()}. "
        f"Stored {stored_schedule_count} schedule items from {fetched_schedule_count} fetched schedules. "
        f"Stored {fetched_match_count} LMT matches. "
        f"Failed seasons: {len(failed_seasons)}. Failed matches: {len(failed_matches)}"
    )
    if failed_seasons:
        print(f"Failed season IDs: {failed_seasons}", file=sys.stderr)
    if failed_matches:
        print(f"Failed match IDs: {failed_matches}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
