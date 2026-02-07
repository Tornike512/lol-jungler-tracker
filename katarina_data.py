"""
Download and filter Katarina games from the decoded LoL replay dataset.
Dataset: https://huggingface.co/datasets/maknee/league-of-legends-decoded-replay-packets
"""

import json
import gzip
from huggingface_hub import hf_hub_download, list_repo_tree

REPO_ID = "maknee/league-of-legends-decoded-replay-packets"
TARGET_CHAMPION = "katarina"


def list_available_batches(max_display=20):
    """List available batch files in the dataset."""
    print("Listing available files in dataset...")
    # First get top-level folders
    top_items = list(list_repo_tree(REPO_ID, repo_type="dataset"))
    folders = [item.path for item in top_items if not hasattr(item, "size")]

    batch_files = []
    for folder in folders:
        items = list(list_repo_tree(REPO_ID, path_in_repo=folder, repo_type="dataset"))
        for item in items:
            if hasattr(item, "size") and item.path.endswith(".jsonl.gz"):
                batch_files.append(item.path)

    print(f"Found {len(batch_files)} batch files across {len(folders)} patches")
    for f in batch_files[:max_display]:
        print(f"  {f}")
    if len(batch_files) > max_display:
        print(f"  ... and {len(batch_files) - max_display} more")
    return batch_files


def download_batch(filename):
    """Download a single batch file from HuggingFace."""
    print(f"\nDownloading {filename}...")
    local_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type="dataset",
    )
    print(f"  Saved to: {local_path}")
    return local_path


def find_katarina_games(local_path, max_games=None):
    """Scan a batch file for games containing Katarina."""
    katarina_games = []
    total_games = 0

    print(f"\nScanning for {TARGET_CHAMPION} games...")
    with gzip.open(local_path, "rt", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            total_games += 1
            match_data = json.loads(line)
            events = match_data.get("events", match_data.get("packets", []))

            # Find Katarina in CreateHero events
            kat_net_id = None
            champions = {}
            for event in events:
                if "CreateHero" in event:
                    hero = event["CreateHero"]
                    champ = hero.get("champion", hero.get("champion_name", ""))
                    net_id = hero.get("id", hero.get("net_id"))
                    name = hero.get("name", hero.get("player_name", ""))
                    champions[net_id] = {"champion": champ, "player": name}
                    if champ.lower() == TARGET_CHAMPION:
                        kat_net_id = net_id

            if kat_net_id is not None:
                katarina_games.append({
                    "game_index": line_num,
                    "katarina_net_id": kat_net_id,
                    "champions": champions,
                    "events": events,
                })
                print(f"  [{len(katarina_games)}] Game #{line_num}: Katarina found (net_id={kat_net_id})")

                if max_games and len(katarina_games) >= max_games:
                    break

    print(f"\nScanned {total_games} games, found {len(katarina_games)} with Katarina")
    return katarina_games


def analyze_katarina_game(game):
    """Extract and display Katarina-specific data from a game."""
    kat_id = game["katarina_net_id"]
    events = game["events"]

    spells = []
    movements = []
    deaths = []
    items_bought = []
    damage_dealt = []
    basic_attacks = []

    # Waypoint keys use sequential index: net_id - 1073741853
    kat_wp_key = str(kat_id - 1073741853)

    for event in events:
        if "CastSpellAns" in event:
            spell = event["CastSpellAns"]
            if spell.get("caster_net_id") == kat_id:
                spells.append(spell)

        if "WaypointGroup" in event or "WaypointGroupWithSpeed" in event:
            key = "WaypointGroup" if "WaypointGroup" in event else "WaypointGroupWithSpeed"
            wp = event[key]
            if kat_wp_key in wp.get("waypoints", {}):
                movements.append({
                    "time": wp.get("time", 0),
                    "waypoints": wp["waypoints"][kat_wp_key],
                })

        if "NPCDieMapViewBroadcast" in event:
            death = event["NPCDieMapViewBroadcast"]
            if death.get("killed_net_id") == kat_id:
                deaths.append(death)

        if "BuyItem" in event:
            buy = event["BuyItem"]
            if buy.get("net_id") == kat_id:
                items_bought.append(buy)

        if "UnitApplyDamage" in event:
            dmg = event["UnitApplyDamage"]
            if dmg.get("source_net_id") == kat_id:
                damage_dealt.append(dmg)

        if "BasicAttackPos" in event:
            atk = event["BasicAttackPos"]
            if atk.get("caster_net_id") == kat_id:
                basic_attacks.append(atk)

    print(f"\n{'='*60}")
    print(f"KATARINA GAME ANALYSIS (Game #{game['game_index']})")
    print(f"{'='*60}")

    print(f"\nChampions in game:")
    for net_id, info in game["champions"].items():
        marker = " <-- KATARINA" if net_id == kat_id else ""
        print(f"  {info['champion']:>15} - {info['player']}{marker}")

    print(f"\nKatarina stats:")
    print(f"  Spell casts:    {len(spells)}")
    print(f"  Basic attacks:  {len(basic_attacks)}")
    print(f"  Move commands:  {len(movements)}")
    print(f"  Deaths:         {len(deaths)}")
    print(f"  Items bought:   {len(items_bought)}")
    print(f"  Damage events:  {len(damage_dealt)}")

    if spells:
        print(f"\nFirst 10 spell casts:")
        for s in spells[:10]:
            time = s.get("time", 0)
            name = s.get("spell_name", "unknown")
            pos = s.get("source_position", {})
            x = pos.get("x", 0)
            z = pos.get("z", 0)
            cd = s.get("cooldown", "?")
            print(f"  t={time:>7.1f}s  {name:<25} pos=({x:.0f}, {z:.0f})  cd={cd}s")

    if movements:
        print(f"\nFirst 10 movement commands:")
        for m in movements[:10]:
            time = m.get("time", 0)
            waypoints = m.get("waypoints", [])
            print(f"  t={time:>7.1f}s  waypoints={len(waypoints)}")

    if items_bought:
        print(f"\nItems purchased:")
        for item in items_bought:
            time = item.get("time", 0)
            item_id = item.get("item_id", "?")
            item_name = item.get("item_name", "")
            print(f"  t={time:>7.1f}s  {item_name} (id={item_id})")

    return {
        "spells": spells,
        "basic_attacks": basic_attacks,
        "movements": movements,
        "deaths": deaths,
        "items": items_bought,
        "damage": damage_dealt,
    }


def save_katarina_data(games, output_path="katarina_games.jsonl"):
    """Save filtered Katarina games to a local file."""
    print(f"\nSaving {len(games)} Katarina games to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        for game in games:
            json.dump({
                "game_index": game["game_index"],
                "katarina_net_id": game["katarina_net_id"],
                "champions": {str(k): v for k, v in game["champions"].items()},
                "events": game["events"],
            }, f)
            f.write("\n")
    print(f"  Done! File size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    import os

    # Step 1: List available batch files
    batch_files = list_available_batches()

    # Step 2: Download first batch
    if batch_files:
        local_path = download_batch(batch_files[0])

        # Step 3: Find Katarina games (limit to 5 for quick test)
        kat_games = find_katarina_games(local_path, max_games=5)

        # Step 4: Analyze first Katarina game
        if kat_games:
            analyze_katarina_game(kat_games[0])

            # Step 5: Save filtered data
            save_katarina_data(kat_games, "katarina_games.jsonl")
