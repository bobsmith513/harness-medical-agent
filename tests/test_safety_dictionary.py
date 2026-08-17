"""药名词典测试：数据可插拔（JSON 落位）与构造期 fail-closed 校验。"""

from __future__ import annotations

from pathlib import Path

import pytest

from harness_agent.safety import SEED_DRUG_DICTIONARY, DrugDictionary, DrugEntry

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "drug_dictionary.json"


class TestSeedDictionary:
    def test_seed_loads_minimal_set(self):
        dictionary = DrugDictionary(SEED_DRUG_DICTIONARY)
        assert len(dictionary) >= 5
        assert dictionary.get("penicillin") is not None

    def test_every_entry_has_atc_code(self):
        for entry in SEED_DRUG_DICTIONARY:
            assert entry.atc_code


class TestPluggableJson:
    def test_json_extension_slot_matches_seed(self):
        """data/drug_dictionary.json 即生产词典的可插拽数据落位。"""
        loaded = DrugDictionary.from_json(DATA_FILE)
        seed = DrugDictionary(SEED_DRUG_DICTIONARY)
        assert [e.normalized_name for e in loaded.entries] == [
            e.normalized_name for e in seed.entries
        ]
        assert loaded.alias_index == seed.alias_index

    def test_to_json_then_from_json_roundtrip(self, tmp_path: Path):
        dictionary = DrugDictionary(SEED_DRUG_DICTIONARY)
        path = tmp_path / "dict.json"
        dictionary.to_json(path)
        reloaded = DrugDictionary.from_json(path)
        assert reloaded.alias_index == dictionary.alias_index


class TestConstructionFailClosed:
    def test_duplicate_normalized_name_rejected(self):
        with pytest.raises(ValueError, match="重复"):
            DrugDictionary(
                [
                    DrugEntry(normalized_name="aspirin", atc_code="N02BA01"),
                    DrugEntry(normalized_name="aspirin", atc_code="N02BA01"),
                ]
            )

    def test_conflicting_alias_rejected(self):
        with pytest.raises(ValueError, match="同时映射"):
            DrugDictionary(
                [
                    DrugEntry(normalized_name="aspirin", atc_code="N02BA01", aliases=["止痛灵"]),
                    DrugEntry(normalized_name="ibuprofen", atc_code="M01AE01", aliases=["止痛灵"]),
                ]
            )

    def test_blank_alias_rejected(self):
        with pytest.raises(ValueError, match="空"):
            DrugDictionary(
                [DrugEntry(normalized_name="aspirin", atc_code="N02BA01", aliases=["  "])]
            )

    def test_match_keys_sorted_longest_first(self):
        dictionary = DrugDictionary(SEED_DRUG_DICTIONARY)
        lengths = [len(key) for key, _ in dictionary.match_keys]
        assert lengths == sorted(lengths, reverse=True)
