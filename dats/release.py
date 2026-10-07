"""Audit and package only the public source tree; exclude private data and outputs."""
from pathlib import Path
import re
import tarfile
import zipfile

EXCLUDED_PARTS = {'.git', '.venv', 'venv', '__pycache__', 'outputs', 'external',
                  'build', 'dist', '.pytest_cache', '.ruff_cache', 'checkpoints'}
TEXT_SUFFIXES = {'.py', '.md', '.json', '.jsonl', '.toml', '.txt', '.yml', '.yaml', '.sh'}


def public_files(root):
    root = Path(root)
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        if not path.is_file() or path.is_symlink():
            continue
        if any(p in EXCLUDED_PARTS or p.endswith('.egg-info') for p in relative.parts):
            continue
        if relative.parts[:2] == ('data', 'private'):
            continue
        if path.name.startswith('.env') or path.name == '.DS_Store':
            continue
        if path.suffix in {'.pyc', '.log', '.bin', '.pt', '.pth', '.safetensors'}:
            continue
        yield path


def audit_public_tree(root):
    patterns = {
        'personal_absolute_path': re.compile(r'/(?:Users|home|Volumes|mnt|workspace)/[^\s\"\']+'),
        'email_address': re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'),
        'credential_literal': re.compile(r'\b(?:hf_|sk-)[A-Za-z0-9_-]{20,}\b'),
    }
    issues = []
    files = list(public_files(root))
    for path in files:
        if path.suffix not in TEXT_SUFFIXES and path.name not in {'LICENSE', '.gitignore'}:
            continue
        text = path.read_text(encoding='utf-8')
        relative = str(path.relative_to(root))
        if not text.isascii():
            issues.append({'file': relative, 'reason': 'non-ASCII text in the English-only release'})
        for label, pattern in patterns.items():
            if pattern.search(text):
                issues.append({'file': relative, 'reason': label})
    return {'public_files': len(files), 'issues': issues,
            'excluded': ['private datasets', 'local outputs', 'model checkpoints', 'virtual environments', 'external checkouts']}


def make_release(root, destination):
    root, destination = Path(root), Path(destination)
    report = audit_public_tree(root)
    if report['issues']:
        raise ValueError('Public tree audit must pass before packaging')
    destination.mkdir(parents=True, exist_ok=True)
    files = list(public_files(root))
    tar_path, zip_path = destination/'dats_github_ready.tar.gz', destination/'dats_github_ready.zip'
    with tarfile.open(tar_path, 'w:gz') as archive:
        for path in files:
            info = archive.gettarinfo(str(path), arcname=str(Path('dats')/path.relative_to(root)))
            info.uid = info.gid = 0
            info.uname = info.gname = ''
            with path.open('rb') as stream:
                archive.addfile(info, stream)
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, str(Path('dats')/path.relative_to(root)))
    with zipfile.ZipFile(zip_path) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('ZIP integrity check failed')
    return {'files': len(files), 'tar': str(tar_path), 'zip': str(zip_path)}
