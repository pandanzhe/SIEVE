import hashlib
import tempfile
import unittest
from pathlib import Path

from sieve.data_factory.acquire import SourceArtifact, fetch_artifact


class AcquireTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.payload = b'{"records": [{"id": "source-1"}]}'
        self.origin = self.root / "origin.json"
        self.origin.write_bytes(self.payload)
        self.cache = self.root / "cache"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_fetch_artifact_writes_verified_local_cache(self) -> None:
        artifact = SourceArtifact(
            dataset="fixture",
            version="v1",
            license="MIT",
            url=self.origin.as_uri(),
            relative_path="fixture/source.json",
            sha256=hashlib.sha256(self.payload).hexdigest(),
        )
        result = fetch_artifact(artifact, self.cache)
        self.assertEqual(result.path.read_bytes(), self.payload)
        self.assertEqual(result.sha256, artifact.sha256)
        self.assertFalse(result.reused)

    def test_existing_verified_artifact_is_reused(self) -> None:
        digest = hashlib.sha256(self.payload).hexdigest()
        artifact = SourceArtifact(
            "fixture", "v1", "MIT", self.origin.as_uri(), "fixture/source.json", digest
        )
        fetch_artifact(artifact, self.cache)
        result = fetch_artifact(artifact, self.cache)
        self.assertTrue(result.reused)

    def test_checksum_mismatch_does_not_leave_cache_file(self) -> None:
        artifact = SourceArtifact(
            "fixture", "v1", "MIT", self.origin.as_uri(), "fixture/source.json", "0" * 64
        )
        with self.assertRaisesRegex(ValueError, "checksum"):
            fetch_artifact(artifact, self.cache)
        self.assertFalse((self.cache / "fixture" / "source.json").exists())


if __name__ == "__main__":
    unittest.main()
