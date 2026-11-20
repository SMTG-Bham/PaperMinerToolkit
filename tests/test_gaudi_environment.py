"""Exercise Gaudi preflight checks without importing accelerator libraries."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.sol_gaudi import check_gaudi_environment as gaudi


@pytest.fixture
def package_locations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Prepare two regular packages and a Habana namespace in one environment."""
    prefix = tmp_path / 'vllm'
    packages = prefix / 'lib' / 'site-packages'
    specs = {
        'torch': SimpleNamespace(origin=str(packages / 'torch' / '__init__.py')),
        'torchvision': SimpleNamespace(origin=str(packages / 'torchvision' / '__init__.py')),
        'habana_frameworks': SimpleNamespace(
            origin=None,
            submodule_search_locations=[str(packages / 'habana_frameworks')],
        ),
    }
    monkeypatch.setattr(gaudi.sys, 'prefix', str(prefix))
    monkeypatch.setattr(gaudi.importlib.util, 'find_spec', specs.get)

    def version(name: str) -> str:
        """Model a namespace package without standalone distribution metadata."""
        if name == 'habana_frameworks':
            raise gaudi.metadata.PackageNotFoundError(name)
        return 'test-version'

    monkeypatch.setattr(gaudi.metadata, 'version', version)
    return specs


def test_gaudi_preflight_accepts_local_packages_and_namespace(
    package_locations: dict[str, object], capsys: pytest.CaptureFixture[str],
) -> None:
    """Accept local origins without importing Torch or requiring namespace metadata."""
    gaudi.main()
    output = capsys.readouterr().out
    assert 'torch test-version from ' in output
    assert 'torchvision test-version from ' in output
    assert 'habana_frameworks ? from ' in output


@pytest.mark.parametrize('spec', [None, SimpleNamespace(origin=None, submodule_search_locations=[])])
def test_gaudi_preflight_rejects_missing_package_locations(
    package_locations: dict[str, object], spec: object,
) -> None:
    """Reject both missing packages and empty namespace search paths."""
    package_locations['habana_frameworks'] = spec
    with pytest.raises(SystemExit, match='habana_frameworks is not importable'):
        gaudi.main()


@pytest.mark.parametrize('package', ['torch', 'habana_frameworks'])
def test_gaudi_preflight_rejects_shadowed_packages(
    package_locations: dict[str, object], tmp_path: Path, package: str,
) -> None:
    """Reject regular and namespace package locations outside the activated prefix."""
    outside = tmp_path / 'user-site' / package
    if package == 'torch':
        package_locations[package] = SimpleNamespace(origin=str(outside / '__init__.py'))
    else:
        package_locations[package].submodule_search_locations.append(str(outside))
    with pytest.raises(SystemExit, match=f'shadowed by another install: {package}'):
        gaudi.main()
