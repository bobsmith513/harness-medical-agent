"""药名归一化测试：折叠空间（全角/大小写/空格）与最长优先扫描。"""

from __future__ import annotations

import pytest

from harness_agent.safety import SEED_DRUG_DICTIONARY, DrugDictionary, DrugNormalizer, fold_text


@pytest.fixture()
def normalizer() -> DrugNormalizer:
    return DrugNormalizer(DrugDictionary(SEED_DRUG_DICTIONARY))


class TestFoldText:
    def test_fullwidth_lower_and_strip(self):
        assert fold_text("Ａ b　C d") == "abcd"

    def test_plain_text_passthrough(self):
        assert fold_text("阿莫西林") == "阿莫西林"


class TestNormalize:
    def test_generic_name(self, normalizer):
        assert normalizer.normalize("阿莫西林") == "amoxicillin"

    def test_historical_alias(self, normalizer):
        assert normalizer.normalize("盘尼西林") == "penicillin"

    def test_brand_name(self, normalizer):
        assert normalizer.normalize("拜阿司匹灵") == "aspirin"

    def test_english_uppercase(self, normalizer):
        assert normalizer.normalize("AMOXICILLIN") == "amoxicillin"

    def test_fullwidth_english(self, normalizer):
        assert normalizer.normalize("ＣＥＦＴＲＩＡＸＯＮＥ") == "ceftriaxone"

    def test_whitespace_inside(self, normalizer):
        assert normalizer.normalize("头孢 曲松") == "ceftriaxone"

    def test_normalized_name_itself(self, normalizer):
        assert normalizer.normalize("penicillin") == "penicillin"

    def test_unknown_returns_none(self, normalizer):
        assert normalizer.normalize("不存在的药") is None


class TestFindMentions:
    def test_multiple_drugs_in_one_text(self, normalizer):
        mentions = normalizer.find_mentions("既往用过阿莫西林和头孢曲松")
        assert {m.normalized_name for m in mentions} == {"amoxicillin", "ceftriaxone"}

    def test_longest_match_first(self, normalizer):
        """benzylpenicillin 含子串 penicillin，长键优先只产出一条提及。"""
        mentions = normalizer.find_mentions("静滴 benzylpenicillin 80万U")
        assert [m.normalized_name for m in mentions] == ["penicillin"]

    def test_no_drugs(self, normalizer):
        assert normalizer.find_mentions("患者血常规未见明显异常") == []

    def test_empty_text(self, normalizer):
        assert normalizer.find_mentions("") == []

    def test_english_with_dosage(self, normalizer):
        mentions = normalizer.find_mentions("AMOXICILLIN 500mg bid")
        assert [m.normalized_name for m in mentions] == ["amoxicillin"]
