"""Debug script to understand ROFL v2 file structure."""
import sys
import json
import struct
import gzip
import zlib

filepath = sys.argv[1] if len(sys.argv) > 1 else "replay.rofl"

with open(filepath, 'rb') as f:
    content = f.read()

print(f"File size: {len(content):,} bytes")
print(f"Magic: {content[:6]}")

# ROFL Header structure (based on reverse engineering)
# Offset 0: Magic (6 bytes) - "RIOT\x02\x00" for v2
# Offset 6: Signature (256 bytes)
# Offset 262: Header length (2 bytes)
# Offset 264: File length (4 bytes)
# Offset 268: Metadata offset (4 bytes)
# Offset 272: Metadata length (4 bytes)
# etc.

offset = 6
signature = content[offset:offset+256]
offset += 256

header_length = struct.unpack('<H', content[offset:offset+2])[0]
offset += 2

file_length = struct.unpack('<I', content[offset:offset+4])[0]
offset += 4

metadata_offset = struct.unpack('<I', content[offset:offset+4])[0]
offset += 4

metadata_length = struct.unpack('<I', content[offset:offset+4])[0]
offset += 4

payload_header_offset = struct.unpack('<I', content[offset:offset+4])[0]
offset += 4

payload_header_length = struct.unpack('<I', content[offset:offset+4])[0]
offset += 4

payload_offset = struct.unpack('<I', content[offset:offset+4])[0]

print(f"\nHeader length: {header_length}")
print(f"File length: {file_length:,}")
print(f"Metadata offset: {metadata_offset}")
print(f"Metadata length: {metadata_length}")
print(f"Payload header offset: {payload_header_offset}")
print(f"Payload header length: {payload_header_length}")
print(f"Payload offset: {payload_offset}")

# Read metadata
print(f"\n--- Reading metadata at offset {metadata_offset} ---")
metadata_raw = content[metadata_offset:metadata_offset + metadata_length]
print(f"Raw metadata first 100 bytes: {metadata_raw[:100]}")

# Check if it starts with JSON
if metadata_raw.startswith(b'{'):
    print("Metadata appears to be JSON")
    try:
        metadata = json.loads(metadata_raw.decode('utf-8'))
        print(f"Game Version: {metadata.get('gameVersion')}")
        print(f"Game Length: {metadata.get('gameLength')}")

        if 'statsJson' in metadata:
            players = json.loads(metadata['statsJson'])
            print(f"\nPlayers ({len(players)}):")
            for p in players:
                champ = p.get('SKIN', 'Unknown')
                name = p.get('NAME', p.get('PUUID', ''))[:20]
                team = p.get('TEAM', '?')
                marker = " <-- KATARINA" if 'katarina' in champ.lower() else ""
                print(f"  [{team}] {champ:15} - {name}{marker}")

            # Save
            with open('replay_metadata.json', 'w') as out:
                json.dump({'metadata': metadata, 'players': players}, out, indent=2)
            print("\n[+] Saved to replay_metadata.json")

    except json.JSONDecodeError as e:
        print(f"JSON error: {e}")
else:
    # Maybe compressed?
    print("Metadata doesn't start with '{', trying decompression...")
    try:
        decompressed = gzip.decompress(metadata_raw)
        print(f"Gzip decompressed: {decompressed[:200]}")
    except:
        pass
    try:
        decompressed = zlib.decompress(metadata_raw)
        print(f"Zlib decompressed: {decompressed[:200]}")
    except:
        pass

    # Try searching for JSON in the metadata section
    import re
    json_match = re.search(rb'\{[^{}]*"gameVersion"[^{}]*\}', metadata_raw)
    if json_match:
        print(f"Found JSON-like in metadata: {json_match.group(0)[:200]}")
