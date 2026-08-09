import tempfile
import unittest
from pathlib import Path

from sieve.cli.generate_grounded import _load_config


class GroundedCliTests(unittest.TestCase):
    def test_config_reader_accepts_utf8_bom(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_bytes(b"\xef\xbb\xbf" + b'{"total": 140}')
            self.assertEqual(_load_config(path), {"total": 140})


if __name__ == "__main__":
    unittest.main()
