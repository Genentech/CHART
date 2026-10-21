"""
The optional archetype annotation, with a fake model in place of the real one.

The fake answers each prompt in the format the prompt asks for, so the
steps run end to end on a small crops folder without a key.
"""

import os
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from chart.annotation import llm, run_annotation, steps, workflow

CHANNEL = 'TOM20'


def reply(prompt):
    if 'producing the final summary' in prompt:
        return '### Summary:\n' + 'refined ' * 30
    if 'Distinguishing Features' in prompt:
        return re.search(r'### Distinguishing Features.*:', prompt).group(0) + '\n- darker' * 10
    if '### Proposed Names:' in prompt:
        return '### Proposed Names:\n1. a\n2. b\n3. c\n\n### Functional Hypothesis:\n- resting cells'
    if '### Summary:' in prompt:
        number = re.search(r'grid for Archetype (\w+)', prompt).group(1)
        return '### Summary:\n' + ('long ' if number == '4' else 'round ') * 40
    if 'normal baseline archetype' in prompt:
        number = re.search(r'compare Archetype (\w+) against', prompt).group(1)
        return f'#### Archetype {number}:\n' + '- similar\n' * 10
    return '### Cell Observations\n' + 'faint ' * 10 + '\n### Shared Traits\nround'


class FakeModel:
    def __init__(self, prompts):
        self.prompts = prompts

    def invoke(self, messages):
        content = messages[-1]['content']
        prompt = content if isinstance(content, str) else content[0]['text']
        self.prompts.append(prompt)
        return SimpleNamespace(content=reply(prompt), response_metadata={'finish_reason': 'stop'})


class FakeEmbedder:
    def encode(self, text, convert_to_numpy=True):
        return np.array([text.count('round'), text.count('long'), 1.0])


def crops(root, archetypes=(1, 2, 3, 4), cells=15):
    """What the crops step leaves: a listing, and grids of five cells."""
    directory = root / 'archetypes' / CHANNEL / 'crops'
    directory.mkdir(parents=True)
    pd.DataFrame([{'Archetype': f'Archetype{n}', 'Rank': rank}
                  for n in archetypes for rank in range(1, cells + 1)]
                 ).to_parquet(directory / 'top_cells.parquet')
    for n in archetypes:
        for row in range(1, -(-cells // 5) + 1):
            Image.new('RGB', (8, 8), 'white').save(directory / f'archetype{n}_{CHANNEL}_grid_{row}.png')
    return directory


def config(root):
    return {'local_output_dir': str(root),
            'archetypes': {'channels': [CHANNEL],
                           'archetypes_to_remove': {'debris': [f'{CHANNEL}_Archetype2', 'FAK_Archetype1']}},
            'annotation': {'normal_archetypes': {CHANNEL: [1]}}}


@pytest.fixture
def model(monkeypatch):
    prompts = []
    monkeypatch.setattr(steps, 'make_llm', lambda model, **settings: FakeModel(prompts))
    monkeypatch.setattr(steps, '_embedder', FakeEmbedder)
    monkeypatch.setattr(workflow, 'check_llm_credentials', lambda **kwargs: 'a fake model')
    return prompts


@pytest.fixture
def no_keys(monkeypatch):
    for name in list(os.environ):
        if 'OPENAI' in name or 'LITELLM' in name or name in ('LLM_PROVIDER', 'API_KEY_SC', 'API_KEY_EUS2'):
            monkeypatch.delenv(name)


def test_every_step_writes_what_the_next_one_reads(tmp_path, model):
    crops(tmp_path)
    directory = Path(run_annotation(config(tmp_path))[CHANNEL])

    assert directory == tmp_path / 'annotation' / CHANNEL
    assert sorted(p.name for p in (directory / 'archetype_annot').iterdir()) == [
        f'archetype{n}_grid_{g}_annotation.md' for n in (1, 3, 4) for g in (1, 2)]
    for n in (1, 3, 4):
        final = (directory / 'archetype_reviews' / f'archetype{n}_final.md').read_text()
        assert all(h in final for h in (steps.SECTION_SUMMARY, steps.SECTION_NAMES, steps.SECTION_HYPOTHESIS))
    dashboard = (directory / 'archetype_dashboard.html').read_text()
    assert dashboard.count('<h2>') == 3 and dashboard.count('<img') == 6
    assert not any('Archetype 2' in prompt for prompt in model)


def test_summaries_that_read_alike_are_told_apart_both_ways(tmp_path, model):
    crops(tmp_path)
    reviews = Path(run_annotation(config(tmp_path))[CHANNEL]) / 'archetype_reviews'

    assert sorted(p.name for p in reviews.glob('*_contrast_*')) == [
        'archetype1_contrast_vs3.md', 'archetype3_contrast_vs1.md']
    assert 'refined against archetype(s): 3' in (reviews / 'archetype1_final.md').read_text()
    assert 'No archetype was similar' in (reviews / 'archetype4_final.md').read_text()


def test_the_dashboard_alone_needs_no_key(tmp_path, model, monkeypatch):
    crops(tmp_path)
    run_annotation(config(tmp_path))
    monkeypatch.setattr(workflow, 'check_llm_credentials', lambda **kwargs: pytest.fail('asked for a key'))
    model.clear()

    run_annotation(config(tmp_path), steps=['dashboard'])
    assert model == []


def test_a_grid_the_crops_step_did_not_write_is_reported(tmp_path, model):
    (crops(tmp_path) / f'archetype3_{CHANNEL}_grid_2.png').unlink()
    with pytest.raises(FileNotFoundError, match='rerun the crops step'):
        run_annotation(config(tmp_path))


def test_without_a_key_nothing_is_sent(tmp_path, monkeypatch, no_keys):
    crops(tmp_path)
    monkeypatch.setattr(llm.importlib.util, 'find_spec', lambda name: object())
    with pytest.raises(llm.ProviderConfigError, match='OPENAI_API_KEY'):
        run_annotation(config(tmp_path))
    assert not (tmp_path / 'annotation').exists()


def test_without_the_packages_the_stage_says_how_to_install_them(tmp_path, monkeypatch, no_keys):
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-test')
    monkeypatch.setattr(llm.importlib.util, 'find_spec', lambda name: None)
    with pytest.raises(llm.ProviderConfigError, match=r"\.\[annotation\]"):
        run_annotation(config(tmp_path))


def test_an_openai_key_alone_goes_to_openai_and_a_gateway_key_never_does(monkeypatch, no_keys):
    monkeypatch.setenv('LITELLM_API_KEY', 'gw-key')
    with pytest.raises(llm.ProviderConfigError):
        llm.resolve_provider()

    monkeypatch.setenv('OPENAI_API_KEY', 'sk-key')
    assert llm.GatewayConfig.get_credentials() == ('sk-key', None)
    assert llm.describe_provider() == 'openai'
