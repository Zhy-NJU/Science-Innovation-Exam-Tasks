#!/usr/bin/env python3
"""Independently inspect the offline thermal wheel environment without installing it.
No container build, Linux execution, or SDK integration is asserted by this test.
"""
import argparse
import email
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tempfile
import zipfile
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.tags import cpython_tags, compatible_tags, parse_tag
from packaging.utils import canonicalize_name

ROOT = next((path for path in Path(__file__).resolve().parents if (path / "official-repo").is_dir()), Path(__file__).resolve().parents[2])


def safe_path(root, relative):
    path = PurePosixPath(relative)
    assert relative and not path.is_absolute() and all(part not in ('', '.', '..') for part in relative.split('/')) and '\\' not in relative
    result = root.joinpath(*path.parts)
    assert not result.is_symlink()
    return result


def read_metadata(path):
    with zipfile.ZipFile(path) as archive:
        metadata_names = [name for name in archive.namelist() if name.endswith('.dist-info/METADATA') and name.count('/') == 1]
        wheel_names = [name for name in archive.namelist() if name.endswith('.dist-info/WHEEL') and name.count('/') == 1]
        assert len(metadata_names) == len(wheel_names) == 1
        metadata = email.message_from_bytes(archive.read(metadata_names[0]))
        wheel = email.message_from_bytes(archive.read(wheel_names[0]))
        return {'name': canonicalize_name(metadata['Name']), 'version': metadata['Version'], 'requires_python': metadata.get('Requires-Python'), 'requires_dist': metadata.get_all('Requires-Dist', []), 'wheel_tags': wheel.get_all('Tag', []), 'pure_python': wheel.get('Root-Is-Purelib') == 'true'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, default=ROOT / 'submission_bundle/tasks/thermal_conductivity')
    parser.add_argument('--report', type=Path, default=ROOT / 'submission_bundle/evidence/tests/offline_environment_audit.json')
    args = parser.parse_args()
    environment = args.package / 'environment'
    configuration = json.loads((environment / 'environment.json').read_text())
    assert configuration['type'] == 'containerfile' and configuration['schema_version'] == '1.0'
    containerfile = safe_path(environment, configuration['containerfile']).read_text()
    assert re.search(r'^ARG SIE_BASE_IMAGE\s*$', containerfile, re.M)
    assert re.search(r'^FROM \$\{SIE_BASE_IMAGE\}\s*$', containerfile, re.M)
    assert not re.search(r'^ENV\s+(?:PATH=|PATH\s)', containerfile, re.M), 'The private research venv must not replace SDK/default Python'
    installer = (environment / 'install_research_environment.py').read_text()
    assert '--no-index' in installer and '--require-hashes' in installer
    assert '--system-site-packages' not in containerfile + installer
    assert 'PIP_NO_INDEX' in installer and 'PIP_CONFIG_FILE' in installer
    assert '/opt/sie-research' in containerfile
    manifest = json.loads((environment / 'wheel_manifest.json').read_text())
    marker = dict(manifest['markers'])
    assert marker['python_full_version'] == '3.11.15' and marker['platform_machine'] == 'x86_64' and marker['sys_platform'] == 'linux'
    platforms = ['manylinux_2_%s_x86_64' % minor for minor in range(36, 4, -1)] + ['manylinux2014_x86_64', 'manylinux2010_x86_64', 'manylinux1_x86_64', 'linux_x86_64']
    supported = set(cpython_tags((3, 11), abis=['cp311'], platforms=platforms)) | set(compatible_tags((3, 11), interpreter='cp311', platforms=platforms))
    actual = {}
    audited = []
    actual_paths = set()
    with tempfile.TemporaryDirectory(prefix='sie-wheel-audit-') as directory:
        temporary = Path(directory)
        for record in manifest['wheels']:
            assert record['name'] not in actual
            assert record['filename'] == Path(record['filename']).name
            combined = hashlib.sha256()
            total = 0
            reconstruction = temporary / record['filename']
            with reconstruction.open('wb') as output:
                for artifact in record['artifacts']:
                    path = safe_path(environment, artifact['path'])
                    assert artifact['path'] not in actual_paths
                    actual_paths.add(artifact['path'])
                    assert path.stat().st_size == artifact['bytes'] <= 50 * 1024 * 1024
                    digest = hashlib.sha256()
                    with path.open('rb') as stream:
                        for block in iter(lambda: stream.read(1024 * 1024), b''):
                            digest.update(block)
                            combined.update(block)
                            output.write(block)
                            total += len(block)
                    assert digest.hexdigest() == artifact['sha256'], artifact['path']
            assert total == record['bytes'] and combined.hexdigest() == record['sha256']
            metadata = read_metadata(reconstruction)
            for key, value in metadata.items():
                assert value == record[key], (record['name'], key)
            tags = set()
            for tag in metadata['wheel_tags']:
                tags.update(parse_tag(tag))
            assert tags & supported, (record['name'], metadata['wheel_tags'])
            if metadata['requires_python']:
                assert marker['python_full_version'] in SpecifierSet(metadata['requires_python'])
            if record['source']['kind'] == 'upstream_wheel':
                assert record['source']['published_sha256'] == record['sha256']
            elif record['source']['kind'] == 'locally_built_pure_python_wheel':
                assert metadata['pure_python'] and metadata['wheel_tags'] == ['py3-none-any']
                with zipfile.ZipFile(reconstruction) as archive:
                    assert not any(name.endswith(('.so', '.pyd', '.dll', '.dylib')) for name in archive.namelist())
            else:
                raise AssertionError('Unexpected provenance type')
            actual[metadata['name']] = metadata
            audited.append({'name': metadata['name'], 'version': metadata['version'], 'filename': record['filename'], 'sha256': record['sha256'], 'bytes': total, 'compatible_tags': sorted(str(tag) for tag in tags & supported)})
            reconstruction.unlink()
    discovered = {p.relative_to(environment).as_posix() for p in (environment / 'wheelhouse').iterdir() if p.is_file()}
    assert discovered == actual_paths, (discovered - actual_paths, actual_paths - discovered)
    edges = []
    for metadata in actual.values():
        for raw in metadata['requires_dist']:
            dependency = Requirement(raw)
            if dependency.marker is not None and not dependency.marker.evaluate(marker):
                continue
            assert not dependency.extras, 'Extra dependency audit needed: ' + raw
            target = actual[canonicalize_name(dependency.name)]
            assert not dependency.specifier or target['version'] in dependency.specifier
            edges.append({'package': metadata['name'], 'requirement': raw, 'version': target['version']})
    locked = {}
    for filename in ('requirements-bootstrap.lock', 'requirements-research.lock'):
        for raw in (environment / filename).read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            match = re.fullmatch(r'([A-Za-z0-9_.-]+)==(\S+) --hash=sha256:([a-f0-9]{64})', line)
            assert match, (filename, line)
            name, version, sha = match.groups()
            key = canonicalize_name(name)
            assert key not in locked
            locked[key] = (version, sha)
    assert set(locked) == set(actual)
    for record in manifest['wheels']:
        assert locked[record['name']] == (record['version'], record['sha256'])
    instruction = json.loads((args.package / 'instruction.json').read_text())
    declared = {canonicalize_name(record['name']): record['version'] for record in instruction['environment']}
    missing = []
    for record in manifest['wheels']:
        if record['role'] == 'research_runtime' and declared.get(record['name']) != record['version']:
            missing.append({'name':record['name'],'manifest_version':record['version'],'instruction_version':declared.get(record['name'])})
    assert not missing, missing
    report = {'passed': True, 'scope': 'Static inspection of real wheel archives, split reconstruction, hashes, CPython 3.11 / Linux x86_64 tags, dependency markers, exact lock closure, package environment declaration, and Containerfile isolation/offline flags. No OCI build, Linux execution or private-framework certification.', 'wheel_count': len(audited), 'active_dependency_edges': len(edges), 'wheels': audited, 'dependencies': edges, 'containerfile_sha256': hashlib.sha256(containerfile.encode()).hexdigest(), 'wheel_manifest_sha256': hashlib.sha256((environment / 'wheel_manifest.json').read_bytes()).hexdigest()}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'passed': True, 'wheels': len(audited), 'dependencies': len(edges), 'report': str(args.report)}))


if __name__ == '__main__':
    main()
