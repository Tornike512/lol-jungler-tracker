"""
Extract compact Katarina-only data from the full HuggingFace replay dataset.
Saves only Katarina-relevant events (spells, movements, damage, items, deaths)
plus nearby enemy positions for training a next-action predictor.

Output: katarina_training_data.jsonl.gz (~1-2 GB for ~6000 games)
"""

import json
import gzip
import os
import sys
import time
from huggingface_hub import hf_hub_download, list_repo_tree

REPO_ID = "maknee/league-of-legends-decoded-replay-packets"
TARGET_CHAMPION = "katarina"
OUTPUT_DIR = "D:\\katarina_dataset"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "katarina_training_data.jsonl.gz")
PROGRESS_FILE = os.path.join(OUTPUT_DIR, "extraction_progress.json")


def get_all_batch_files():
    """Get all batch file paths from the dataset."""
    print("Listing dataset files...")
    top_items = list(list_repo_tree(REPO_ID, repo_type="dataset"))
    folders = [item.path for item in top_items if not hasattr(item, "size")]

    batch_files = []
    for folder in folders:
        items = list(list_repo_tree(REPO_ID, path_in_repo=folder, repo_type="dataset"))
        for item in items:
            if hasattr(item, "size") and item.path.endswith(".jsonl.gz"):
                batch_files.append(item.path)

    print(f"Found {len(batch_files)} batch files across {len(folders)} patches")
    return batch_files


def load_progress():
    """Load extraction progress to resume interrupted runs."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return {"completed_batches": [], "total_katarina_games": 0}


def save_progress(progress):
    """Save extraction progress."""
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f)


def extract_katarina_from_game(events):
    """Extract compact Katarina-specific data from a single game.
    Returns None if Katarina is not in the game."""

    # Find Katarina and all champions
    kat_net_id = None
    champions = {}
    enemy_net_ids = []
    ally_net_ids = []

    for event in events:
        if "CreateHero" not in event:
            continue
        hero = event["CreateHero"]
        net_id = hero.get("net_id")
        champ = hero.get("champion", "")
        name = hero.get("name", "")
        champions[net_id] = {"champion": champ, "player": name}
        if champ.lower() == TARGET_CHAMPION:
            kat_net_id = net_id

    if kat_net_id is None:
        return None

    # Determine teams: heroes are created in order, first 5 = team 1, next 5 = team 2
    hero_ids = sorted(champions.keys())
    kat_team_idx = hero_ids.index(kat_net_id)
    if kat_team_idx < 5:
        ally_net_ids = hero_ids[:5]
        enemy_net_ids = hero_ids[5:]
    else:
        ally_net_ids = hero_ids[5:]
        enemy_net_ids = hero_ids[:5]

    kat_wp_key = str(kat_net_id - 1073741853)

    # Build compact action sequence
    actions = []

    for event in events:
        if "CastSpellAns" in event:
            spell = event["CastSpellAns"]
            if spell.get("caster_net_id") == kat_net_id:
                actions.append({
                    "t": round(spell.get("time", 0), 2),
                    "type": "spell",
                    "name": spell.get("spell_name", ""),
                    "slot": spell.get("slot"),
                    "x": round(spell.get("source_position", {}).get("x", 0), 1),
                    "z": round(spell.get("source_position", {}).get("z", 0), 1),
                    "tx": round(spell.get("target_position", {}).get("x", 0), 1),
                    "tz": round(spell.get("target_position", {}).get("z", 0), 1),
                    "cd": spell.get("cooldown"),
                    "mana": spell.get("mana_cost"),
                    "targets": spell.get("target_net_ids", []),
                })

        if "BasicAttackPos" in event:
            atk = event["BasicAttackPos"]
            if atk.get("caster_net_id") == kat_net_id:
                actions.append({
                    "t": round(atk.get("time", 0), 2),
                    "type": "attack",
                    "x": round(atk.get("source_position", {}).get("x", 0), 1),
                    "z": round(atk.get("source_position", {}).get("z", 0), 1),
                    "tx": round(atk.get("target_position", {}).get("x", 0), 1),
                    "tz": round(atk.get("target_position", {}).get("z", 0), 1),
                    "target": atk.get("target_net_id"),
                })

        if "WaypointGroup" in event or "WaypointGroupWithSpeed" in event:
            key = "WaypointGroup" if "WaypointGroup" in event else "WaypointGroupWithSpeed"
            wp = event[key]
            if kat_wp_key in wp.get("waypoints", {}):
                points = wp["waypoints"][kat_wp_key]
                actions.append({
                    "t": round(wp.get("time", 0), 2),
                    "type": "move",
                    "waypoints": [{"x": round(p["x"], 1), "z": round(p["z"], 1)} for p in points],
                })

        if "BuyItem" in event:
            buy = event["BuyItem"]
            if buy.get("net_id") == kat_net_id:
                actions.append({
                    "t": round(buy.get("time", 0), 2),
                    "type": "buy",
                    "item_id": buy.get("item_id"),
                    "item": buy.get("item_name", ""),
                })

        if "NPCDieMapViewBroadcast" in event:
            death = event["NPCDieMapViewBroadcast"]
            killed = death.get("killed_net_id")
            killer = death.get("killer_net_id")
            if killed == kat_net_id:
                actions.append({
                    "t": round(death.get("time", 0), 2),
                    "type": "death",
                    "killer": killer,
                })
            elif killed in enemy_net_ids and killer == kat_net_id:
                actions.append({
                    "t": round(death.get("time", 0), 2),
                    "type": "kill",
                    "killed": killed,
                })

        if "UnitApplyDamage" in event:
            dmg = event["UnitApplyDamage"]
            if dmg.get("source_net_id") == kat_net_id:
                actions.append({
                    "t": round(dmg.get("time", 0), 2),
                    "type": "dmg_out",
                    "target": dmg.get("target_net_id"),
                    "amount": round(dmg.get("damage", 0), 1),
                })
            elif dmg.get("target_net_id") == kat_net_id:
                actions.append({
                    "t": round(dmg.get("time", 0), 2),
                    "type": "dmg_in",
                    "source": dmg.get("source_net_id"),
                    "amount": round(dmg.get("damage", 0), 1),
                })

        if "DoSetCooldown" in event:
            cd = event["DoSetCooldown"]
            if cd.get("net_id") == kat_net_id and cd.get("cooldown", 0) > 0:
                actions.append({
                    "t": round(cd.get("time", 0), 2),
                    "type": "cooldown",
                    "slot": cd.get("slot"),
                    "cd": round(cd.get("cooldown", 0), 2),
                })

    # Sort by time
    actions.sort(key=lambda a: a["t"])

    # Build champion info
    matchup = {
        "allies": {str(nid): champions[nid]["champion"] for nid in ally_net_ids},
        "enemies": {str(nid): champions[nid]["champion"] for nid in enemy_net_ids},
    }

    return {
        "kat_net_id": kat_net_id,
        "matchup": matchup,
        "actions": actions,
    }


def process_batch(batch_file):
    """Download and process a single batch file. Returns list of compact Katarina games."""
    local_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=batch_file,
        repo_type="dataset",
    )

    katarina_games = []
    with gzip.open(local_path, "rt", encoding="utf-8") as f:
        for line in f:
            match_data = json.loads(line)
            events = match_data.get("events", match_data.get("packets", []))
            result = extract_katarina_from_game(events)
            if result:
                result["batch"] = batch_file
                katarina_games.append(result)

    # Delete cached batch file to free disk space on C:
    try:
        os.remove(local_path)
    except OSError:
        pass

    return katarina_games


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    batch_files = get_all_batch_files()
    progress = load_progress()
    completed = set(progress["completed_batches"])
    remaining = [f for f in batch_files if f not in completed]

    print(f"\nTotal batches: {len(batch_files)}")
    print(f"Already processed: {len(completed)}")
    print(f"Remaining: {len(remaining)}")
    print(f"Katarina games found so far: {progress['total_katarina_games']}")
    print(f"\nOutput: {OUTPUT_FILE}")
    print("=" * 60)

    # Open output file in append mode
    mode = "ab" if completed else "wb"
    with gzip.open(OUTPUT_FILE, mode) as out_f:
        for i, batch_file in enumerate(remaining):
            start = time.time()
            try:
                games = process_batch(batch_file)
                for game in games:
                    line = json.dumps(game, separators=(",", ":")) + "\n"
                    out_f.write(line.encode("utf-8"))

                elapsed = time.time() - start
                progress["completed_batches"].append(batch_file)
                progress["total_katarina_games"] += len(games)
                save_progress(progress)

                done = len(progress["completed_batches"])
                total_kat = progress["total_katarina_games"]
                print(
                    f"[{done}/{len(batch_files)}] {batch_file}: "
                    f"{len(games)} Kat games | "
                    f"Total: {total_kat} | "
                    f"{elapsed:.1f}s"
                )
            except Exception as e:
                print(f"[ERROR] {batch_file}: {e}")
                continue

    total = progress["total_katarina_games"]
    size_mb = os.path.getsize(OUTPUT_FILE) / 1024 / 1024
    print(f"\n{'=' * 60}")
    print(f"DONE! {total} Katarina games saved to {OUTPUT_FILE} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
