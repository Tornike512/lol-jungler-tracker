"""
ROFL Parser - Extract metadata and player info from League of Legends replay files.
Based on community reverse engineering of the .rofl format.
"""
import base64
import dataclasses
import gzip
import io
import json
import struct
import sys
import os
from Crypto.Cipher import Blowfish


@dataclasses.dataclass
class Header:
    magic: bytes
    signature: bytes
    header_length: int
    file_length: int
    metadata_offset: int
    metadata_length: int
    payload_header_offset: int
    payload_header_length: int
    payload_offset: int

    @classmethod
    def read(cls, f):
        magic = f.read(6)
        signature = f.read(256)
        header_length = int.from_bytes(f.read(2), 'little')
        file_length = int.from_bytes(f.read(4), 'little')
        metadata_offset = int.from_bytes(f.read(4), 'little')
        metadata_length = int.from_bytes(f.read(4), 'little')
        payload_header_offset = int.from_bytes(f.read(4), 'little')
        payload_header_length = int.from_bytes(f.read(4), 'little')
        payload_offset = int.from_bytes(f.read(4), 'little')
        return cls(magic, signature, header_length, file_length,
                   metadata_offset, metadata_length, payload_header_offset,
                   payload_header_length, payload_offset)

    def validate(self):
        # Accept both old (RIOT\x00\x00) and new (RIOT\x02\x00) format versions
        if not self.magic.startswith(b'RIOT'):
            raise ValueError(f"Invalid magic: {self.magic}")
        return True


@dataclasses.dataclass
class PayloadHeader:
    game_id: int
    game_length: int
    keyframe_count: int
    chunk_count: int
    end_startup_chunk_id: int
    start_game_chunk_id: int
    keyframe_interval: int
    encryption_key_length: int
    encryption_key: bytes

    @classmethod
    def read(cls, f):
        game_id = int.from_bytes(f.read(8), 'little')
        game_length = int.from_bytes(f.read(4), 'little')
        keyframe_count = int.from_bytes(f.read(4), 'little')
        chunk_count = int.from_bytes(f.read(4), 'little')
        end_startup_chunk_id = int.from_bytes(f.read(4), 'little')
        start_game_chunk_id = int.from_bytes(f.read(4), 'little')
        keyframe_interval = int.from_bytes(f.read(4), 'little')
        encryption_key_length = int.from_bytes(f.read(2), 'little')
        encrypted_key = f.read(encryption_key_length)

        # Decrypt the encryption key using game_id as Blowfish key
        try:
            key_cipher = Blowfish.new(str(game_id).encode('ascii'), Blowfish.MODE_ECB)
            decrypted = key_cipher.decrypt(base64.b64decode(encrypted_key))
            # Remove PKCS5 padding
            encryption_key = decrypted[:-decrypted[-1]]
        except Exception as e:
            encryption_key = None

        return cls(game_id, game_length, keyframe_count, chunk_count,
                   end_startup_chunk_id, start_game_chunk_id, keyframe_interval,
                   encryption_key_length, encryption_key)


class RoflFile:
    def __init__(self, filepath):
        self.filepath = filepath
        self.header = None
        self.metadata = None
        self.payload_header = None

    def parse(self):
        with open(self.filepath, 'rb') as f:
            # Read header
            self.header = Header.read(f)
            self.header.validate()

            # Read metadata (JSON)
            f.seek(self.header.metadata_offset)
            metadata_raw = f.read(self.header.metadata_length)
            self.metadata = json.loads(metadata_raw.decode('utf-8'))

            # Parse statsJson if present
            if 'statsJson' in self.metadata:
                self.metadata['players'] = json.loads(self.metadata.pop('statsJson'))

            # Read payload header
            f.seek(self.header.payload_header_offset)
            payload_data = f.read(self.header.payload_header_length)
            with io.BytesIO(payload_data) as pf:
                self.payload_header = PayloadHeader.read(pf)

        return self

    def get_game_info(self):
        """Return basic game information."""
        return {
            'game_id': self.payload_header.game_id if self.payload_header else None,
            'game_length': self.metadata.get('gameLength', 0),
            'game_version': self.metadata.get('gameVersion', 'Unknown'),
        }

    def get_players(self):
        """Return list of players with their champions."""
        players = self.metadata.get('players', [])
        result = []
        for p in players:
            result.append({
                'name': p.get('NAME', p.get('PUUID', 'Unknown')),
                'champion': p.get('SKIN', p.get('CHAMPION', 'Unknown')),
                'team': 'Blue' if p.get('TEAM', '').upper() == 'BLUE' else 'Red',
                'position': p.get('INDIVIDUAL_POSITION', p.get('TEAM_POSITION', 'Unknown')),
                'kills': p.get('CHAMPIONS_KILLED', 0),
                'deaths': p.get('NUM_DEATHS', 0),
                'assists': p.get('ASSISTS', 0),
                'gold_earned': p.get('GOLD_EARNED', 0),
                'level': p.get('LEVEL', 0),
                'win': p.get('WIN', 'Unknown'),
            })
        return result

    def find_champion(self, champion_name):
        """Find a specific champion in the game."""
        players = self.get_players()
        for p in players:
            if champion_name.lower() in p['champion'].lower():
                return p
        return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python rofl_parser.py <replay.rofl>")
        return

    filepath = sys.argv[1]
    if not os.path.exists(filepath):
        print(f"[!] File not found: {filepath}")
        return

    print(f"--- ROFL PARSER ---")
    print(f"Parsing: {filepath}\n")

    try:
        rofl = RoflFile(filepath).parse()

        # Game info
        info = rofl.get_game_info()
        print(f"Game ID: {info['game_id']}")
        print(f"Game Version: {info['game_version']}")
        print(f"Game Length: {info['game_length']} seconds ({info['game_length']//60}m {info['game_length']%60}s)\n")

        # Players
        print("Players:")
        print("-" * 70)
        players = rofl.get_players()

        katarina_found = False
        for p in players:
            marker = ""
            if 'katarina' in p['champion'].lower():
                marker = " <-- KATARINA"
                katarina_found = True
            print(f"  [{p['team']:4}] {p['champion']:15} - {p['name'][:20]:20} "
                  f"KDA: {p['kills']}/{p['deaths']}/{p['assists']} "
                  f"Gold: {p['gold_earned']:,}{marker}")

        print("-" * 70)

        if katarina_found:
            kat = rofl.find_champion('Katarina')
            print(f"\n[+] Katarina found in this replay!")
            print(f"    Player: {kat['name']}")
            print(f"    Result: {kat['win']}")
            print(f"    KDA: {kat['kills']}/{kat['deaths']}/{kat['assists']}")
        else:
            print("\n[!] Katarina not found in this replay.")

        # Save metadata to JSON
        output_file = filepath.replace('.rofl', '_metadata.json')
        with open(output_file, 'w') as f:
            json.dump({
                'game_info': info,
                'players': players
            }, f, indent=2)
        print(f"\n[+] Metadata saved to: {output_file}")

    except Exception as e:
        print(f"[!] Error parsing replay: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
