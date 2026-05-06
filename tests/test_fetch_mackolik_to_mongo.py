from __future__ import annotations

import unittest
from datetime import date

from fetch_mackolik_to_mongo import (
    MatchRef,
    build_document,
    filter_existing_match_rows,
    is_finished_livedata_row,
    is_missing_opta_feed_payload,
    iter_target_dates,
    odds_markets_from_livedata,
    parse_livedata_match_row,
    parse_f9_enrichment,
    parse_archive_match_info,
    parse_archive_match_plus,
    parse_archive_match_stats,
    parse_archive_other_matches,
    parse_archive_standings,
    parse_jsonp_payload,
)


ARCHIVE_MAC_HTML = """
<html>
  <body>
    <div class="match-statistics-rows">
      <div class="team-1-statistics-text">%44</div>
      <div class="statistics-title-text">Topla Oynama</div>
      <div class="team-2-statistics-text">%56</div>
    </div>
    <div class="match-statistics-rows-2">
      <div class="team-1-statistics-text">8</div>
      <div class="statistics-title-text">Toplam Şut</div>
      <div class="team-2-statistics-text">11</div>
    </div>
    <div id="dvStanding">
      <div class="card">
        <div class="standing-title"><h2><a href="/Puan-Durumu/s=70381/Trendyol-Super-Lig">Trendyol Süper Lig</a></h2></div>
        <table class="list-table">
          <tr class="row alt1" data-teamid="656">
            <td><b>13</b></td>
            <td><a href="/Takim/656/Kasimpasa/2025/2026">Kasımpaşa</a></td>
            <td>&nbsp;</td>
            <td>31</td><td>7</td><td>10</td><td>14</td><td><b>31</b></td>
          </tr>
        </table>
      </div>
    </div>
    <div id="dvOtherMatches">
      <tr class="row alt1 match_4308331">
        <td class="matchStatus"><b>MS</b></td>
        <td><a href="/Takim/3/Besiktas/2025/2026">Beşiktaş</a></td>
        <td><a class="matchScore" href="/Mac/4308331/Besiktas-Eyupspor">2 - 1</a></td>
        <td><a href="/Takim/557/Eyupspor/2025/2026">Eyüpspor</a></td>
      </tr>
    </div>
    <a href="/Stadyum/747/Recep-Tayyip-Erdogan-Stadyumu">Recep Tayyip Erdoğan Stadyumu</a>
    <a href="/Hakem/14846/Cihan-Aydin"><b>Cihan Aydın</b></a>
    <a href="/Hakem/8314/Deniz-Caner-Ozaral">Deniz Caner Özaral</a>
    <a href="/Hakem/6388/Bilal-Golen">Bilal Gölen</a>
    beIN SPORTS 1
    <a href="/Antrenor/1/Shota-Arveladze">Shota Arveladze</a>
    <a href="/Antrenor/2/Fatih-Tekke">Fatih Tekke</a>
  </body>
</html>
"""

ARCHIVE_MAC_PLUS_HTML = """
<html>
  <body>
    <script type="text/template" id="tmpl-widget-heatmap">
      <opta-widget sport="football" widget="heatmap" competition="115" season="2025" match="2574034" show_team_sheets="true" show_subs="true"></opta-widget>
    </script>
    <script type="text/template" id="tmpl-widget-matchstats1">
      <opta-widget sport="football" widget="matchstats" competition="115" season="2025" match="2574034"
        stats_categories="GENEL|possession,passes_accuracy$SAVUNMA|tackles_accuracy,cards_red$HÜCUM|shots,shots_on_target"></opta-widget>
    </script>
    <script type="text/template" id="tmpl-widget-pass-matrix">
      <opta-widget sport="football" widget="pass_matrix" competition="115" season="2025" match="2574034"></opta-widget>
    </script>
    <div id="mac-tabbed-widgets">
      <ul>
        <li data-widget-id="heatmap">ISI HARİTASI</li>
        <li data-widget-id="pass-matrix">PAS AĞI</li>
      </ul>
    </div>
    <div class="mac-team-system">4-2-3-1</div>
    <div class="mac-team-system">4-3-3</div>
  </body>
</html>
"""

F9_JSONP = """
f9_2574034({
  "SoccerFeed": {
    "SoccerDocument": {
      "Competition": {
        "Name": "Turkish Super Lig",
        "Stat": [
          {"@value": "2025", "@attributes": {"Type": "season_id"}},
          {"@value": "2", "@attributes": {"Type": "matchday"}}
        ]
      },
      "MatchData": {
        "MatchInfo": {"Attendance": "3525"},
        "MatchOfficial": {
          "OfficialData": {"OfficialRef": {"@attributes": {"Type": "Main"}}},
          "OfficialName": {"First": "Cihan", "Last": "Aydin"},
          "@attributes": {"uID": "o56515"}
        },
        "AssistantOfficials": {
          "AssistantOfficial": [
            {"@attributes": {"FirstName": "Deniz Caner", "LastName": "Ozaral", "Type": "Assistant Referee 1", "uID": "o53089"}}
          ]
        },
        "TeamData": [
          {
            "@attributes": {"Side": "Home", "TeamRef": "t656"},
            "PlayerLineUp": {
              "MatchPlayer": [
                {
                  "@attributes": {"PlayerRef": "p111", "Position": "Goalkeeper", "ShirtNumber": "1", "Status": "Start"},
                  "Stat": [
                    {"@value": "90", "@attributes": {"Type": "mins_played"}},
                    {"@value": "38", "@attributes": {"Type": "accurate_pass"}}
                  ]
                }
              ]
            }
          },
          {
            "@attributes": {"Side": "Away", "TeamRef": "t4"},
            "PlayerLineUp": {
              "MatchPlayer": [
                {
                  "@attributes": {"PlayerRef": "p222", "Position": "Striker", "ShirtNumber": "9", "Status": "Start"},
                  "Stat": [
                    {"@value": "90", "@attributes": {"Type": "mins_played"}},
                    {"@value": "2", "@attributes": {"Type": "total_scoring_att"}}
                  ]
                }
              ]
            }
          }
        ]
      },
      "Team": [
        {
          "@attributes": {"uID": "t656"},
          "Name": "Kasımpaşa",
          "Short_name": "Kasimpasa",
          "TeamOfficial": {"PersonName": {"First": "Shota", "Last": "Arveladze"}, "@attributes": {"uID": "man1"}},
          "Player": [{"PersonName": {"First": "Andreas", "Last": "Gianniotis"}, "@attributes": {"Position": "Goalkeeper", "uID": "p111"}}]
        },
        {
          "@attributes": {"uID": "t4"},
          "Name": "Trabzonspor",
          "Short_name": "Trabzon",
          "TeamOfficial": {"PersonName": {"First": "Fatih", "Last": "Tekke"}, "@attributes": {"uID": "man2"}},
          "Player": [{"PersonName": {"First": "Paul", "Last": "Onuachu"}, "@attributes": {"Position": "Striker", "uID": "p222"}}]
        }
      ],
      "Venue": {"Name": "Recep Tayyip Erdogan Stadyumu", "@attributes": {"uID": "v3427"}}
    }
  }
})
"""

F24_PAYLOAD = {
    "Games": {
        "Game": {
            "@attributes": {
                "id": "2574034",
                "competition_id": "115",
                "season_id": "2025",
            },
            "Event": [{"@attributes": {"type_id": "16", "min": "8", "sec": "0", "team_id": "656"}}],
        }
    }
}

TOP_PERFORMERS_RAW = {
    "ShotList": [{"OYUNCU_ID": 222, "OYUNCU_ADI": "Paul Onuachu", "TAKIM_ID": 4, "FP_TOTAL_SCORING_ATT": 2}],
    "PassList": [],
    "PassSuccList": [],
    "TakeonList": [],
    "TackleList": [],
    "PassFailList": [],
    "FoulList": [],
}

LIVEDATA_ROW = [
    4308603,
    2,
    "Fenerbahçe",
    451,
    "Başakşehir FK",
    0,
    "",
    "",
    0,
    0,
    0,
    0,
    0,
    0,
    2837518,
    {"aeleme": 0, "e": 0, "goal": "", "h1": 0, "h2": 0, "k1": 0, "k2": 0, "ogd": 1, "tId": 0},
    "20:00",
    0,
    "1.48",
    "3.74",
    "3.85",
    "2.19",
    "1.33",
    1,
    "0.0",
    "0.0",
    "0.0",
    "0.0",
    "0.0",
    "0",
    "0",
    "0",
    "0",
    None,
    "1",
    "02/05/2026",
    [1, "Türkiye", 1, "Süper Lig", 70381, "2025/2026", "", 0, 1, "TSL", 0, 1],
    1,
]

FINAL_LIVEDATA_ROW = list(LIVEDATA_ROW)
FINAL_LIVEDATA_ROW[5] = 4
FINAL_LIVEDATA_ROW[6] = "MS"
FINAL_LIVEDATA_ROW[7] = "1-0"
FINAL_LIVEDATA_ROW[29] = "1"
FINAL_LIVEDATA_ROW[30] = "0"


class FetchMackolikToMongoTests(unittest.TestCase):
    def test_iter_target_dates_supports_reverse_order(self) -> None:
        days = list(iter_target_dates(date(2025, 8, 16), date(2025, 8, 18), reverse=True))
        self.assertEqual(days, [date(2025, 8, 18), date(2025, 8, 17), date(2025, 8, 16)])

    def test_parse_livedata_match_row_extracts_numeric_match_id(self) -> None:
        parsed = parse_livedata_match_row(LIVEDATA_ROW, date(2026, 5, 2))
        assert parsed is not None
        self.assertEqual(parsed["id"], "4308603")
        self.assertEqual(parsed["homeTeam"]["id"], "2")
        self.assertEqual(parsed["awayTeam"]["name"], "Başakşehir FK")
        self.assertEqual(parsed["competitionName"], "Süper Lig")
        self.assertEqual(parsed["matchDateText"], "02/05/2026")
        self.assertEqual(parsed["state"], "pre")

    def test_parse_livedata_match_row_marks_finished_match_as_post(self) -> None:
        parsed = parse_livedata_match_row(FINAL_LIVEDATA_ROW, date(2026, 5, 2))
        assert parsed is not None
        self.assertEqual(parsed["state"], "post")
        self.assertEqual(parsed["score"]["home"], 1)
        self.assertEqual(parsed["statusBoxContent"], "MS")

    def test_odds_markets_from_livedata_extracts_1x2_and_totals(self) -> None:
        parsed = parse_livedata_match_row(LIVEDATA_ROW, date(2026, 5, 2))
        assert parsed is not None
        markets = odds_markets_from_livedata(parsed)
        self.assertEqual(markets[0]["market_name"], "Maç Sonucu")
        self.assertEqual(markets[0]["outcomes"][0]["odd"], 1.48)
        self.assertEqual(markets[1]["market_name"], "(2,5) Alt/Üst")
        self.assertEqual(markets[1]["outcomes"][1]["odd"], 1.33)

    def test_is_finished_livedata_row_requires_terminal_match_state(self) -> None:
        self.assertFalse(is_finished_livedata_row(LIVEDATA_ROW))
        self.assertTrue(is_finished_livedata_row(FINAL_LIVEDATA_ROW))

    def test_filter_existing_match_rows_skips_already_stored_match_ids(self) -> None:
        class FakeCollection:
            def distinct(self, field: str, query: dict[str, object]) -> list[str]:
                self.field = field
                self.query = query
                return ["4308603"]

        second_row = list(LIVEDATA_ROW)
        second_row[0] = 4308604
        collection = FakeCollection()
        filtered_rows, skipped = filter_existing_match_rows(collection, [LIVEDATA_ROW, second_row])
        self.assertEqual(skipped, 1)
        self.assertEqual(len(filtered_rows), 1)
        self.assertEqual(filtered_rows[0][0], 4308604)
        self.assertEqual(collection.field, "match_id")

    def test_parse_archive_match_info_extracts_officials_and_broadcast(self) -> None:
        parsed = parse_archive_match_info(ARCHIVE_MAC_HTML)
        self.assertEqual(parsed["venue"]["venue_id"], 747)
        self.assertEqual(parsed["officials"][0]["name"], "Cihan Aydın")
        self.assertEqual(parsed["officials"][1]["role"], "assistant_referee")
        self.assertEqual(parsed["broadcasts"][0]["name"], "beIN SPORTS 1")
        self.assertEqual(parsed["home_coach"], "Shota Arveladze")
        self.assertEqual(parsed["away_coach"], "Fatih Tekke")

    def test_parse_archive_match_stats_extracts_visible_stats(self) -> None:
        parsed = parse_archive_match_stats(ARCHIVE_MAC_HTML)
        self.assertEqual(parsed["genel"]["Topla Oynama"]["home"], "%44")
        self.assertEqual(parsed["hucum"]["Toplam Şut"]["away"], "11")

    def test_parse_archive_standings_extracts_rows(self) -> None:
        parsed = parse_archive_standings(ARCHIVE_MAC_HTML)
        self.assertEqual(parsed["competition_name"], "Trendyol Süper Lig")
        self.assertEqual(parsed["rows"][0]["team_id"], 656)
        self.assertEqual(parsed["rows"][0]["position"], 13)

    def test_parse_archive_other_matches_extracts_fixture_rows(self) -> None:
        parsed = parse_archive_other_matches(ARCHIVE_MAC_HTML)
        self.assertEqual(parsed[0]["match_id"], "4308331")
        self.assertEqual(parsed[0]["home_team"], "Beşiktaş")
        self.assertEqual(parsed[0]["score"], "2 - 1")

    def test_parse_archive_match_plus_extracts_opta_metadata(self) -> None:
        parsed = parse_archive_match_plus(ARCHIVE_MAC_PLUS_HTML)
        self.assertEqual(parsed["opta_identifiers"]["competition_id"], "115")
        self.assertEqual(parsed["opta_identifiers"]["season_year"], "2025")
        self.assertIn("heatmap", parsed["opta_identifiers"]["widget_names"])
        self.assertEqual(parsed["opta_identifiers"]["stats_categories"]["GENEL"], ["possession", "passes_accuracy"])
        self.assertTrue(parsed["squads_available"])
        self.assertIn("tmpl-widget-heatmap", parsed["mac_page_fragments"])

    def test_parse_jsonp_payload_extracts_dict(self) -> None:
        parsed = parse_jsonp_payload(F9_JSONP)
        self.assertIn("SoccerFeed", parsed)

    def test_missing_opta_feed_payload_is_detected(self) -> None:
        payload = "Error: feed_type, game_id combination not found in the feed repository"
        self.assertTrue(is_missing_opta_feed_payload(payload))

    def test_parse_f9_enrichment_extracts_players_and_meta(self) -> None:
        enrichment = parse_f9_enrichment(parse_jsonp_payload(F9_JSONP))
        self.assertEqual(enrichment["competition"]["matchday"], 2)
        self.assertEqual(enrichment["venue"]["attendance"], 3525)
        self.assertEqual(enrichment["home_team"]["coach_name"], "Shota Arveladze")
        self.assertEqual(enrichment["away_team"]["players"][0]["player_name"], "Paul Onuachu")
        self.assertEqual(enrichment["player_performance"]["pas"]["111"]["stats"]["accurate_pass"], 38)

    def test_build_document_uses_both_opta_sources(self) -> None:
        document = build_document(
            match_payload={
                "id": "4308329",
                "homeTeam": {"id": 656, "name": "Kasımpaşa", "slug": "kasimpasa", "recentForm": []},
                "awayTeam": {"id": 4, "name": "Trabzonspor", "slug": "trabzonspor", "recentForm": []},
                "score": {"home": 0, "away": 1},
                "statusBoxContent": "MS",
                "competitionName": "Trendyol Süper Lig",
            },
            match_ref=MatchRef(
                match_id="4308329",
                match_slug="kasimpasa-trabzonspor",
                match_date=date(2025, 8, 18),
                home_name="Kasımpaşa",
                away_name="Trabzonspor",
            ),
            header_payload={},
            key_events_payload={},
            match_page_html="",
            match_plus_page_html="",
            iddaa_page_html="",
            archive_match_page_html=ARCHIVE_MAC_HTML,
            archive_match_plus_page_html=ARCHIVE_MAC_PLUS_HTML,
            archive_opta_stats_raw=TOP_PERFORMERS_RAW,
            direct_opta_f24=F24_PAYLOAD,
            direct_opta_f9=parse_jsonp_payload(F9_JSONP),
            livedata_odds_markets=[],
            fetch_errors=[],
        )
        self.assertTrue(document["opta_feeds"]["mackolik_raw_available"])
        self.assertTrue(document["opta_feeds"]["f24_available"])
        self.assertTrue(document["opta_feeds"]["f9_available"])
        self.assertEqual(document["top_performers"]["ShotList"][0]["OYUNCU_ADI"], "Paul Onuachu")
        self.assertEqual(document["match_data"]["home_players"][0]["player_name"], "Andreas Gianniotis")
        self.assertEqual(document["match_data"]["away_players"][0]["player_name"], "Paul Onuachu")
        self.assertEqual(document["player_performance"]["hucum"]["222"]["stats"]["total_scoring_att"], 2)


if __name__ == "__main__":
    unittest.main()
