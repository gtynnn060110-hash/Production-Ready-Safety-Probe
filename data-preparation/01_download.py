"""Download pinned HF snapshots or explicit public files; safe opt-in ZIP extraction."""
import argparse
import fnmatch
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile
from common import config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--sources', nargs='+', help='Only selected enabled source names')
    args = parser.parse_args()
    cfg = config(args.config)
    failures = []
    for source in cfg['sources']:
        if not source.get('enabled', True) or (args.sources and source['name'] not in args.sources):
            continue
        target = Path(cfg.get('raw_dir', 'data/raw')) / source['name']
        target.mkdir(parents=True, exist_ok=True)
        try:
            if source.get('repo_id'):
                from huggingface_hub import snapshot_download
                resolved = snapshot_download(repo_id=source['repo_id'], repo_type='dataset',
                    revision=source.get('revision', 'main'), local_dir=target,
                    allow_patterns=source.get('allow_patterns'))
            else:
                for item in source.get('files', []):
                    dest = target / item['path']
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    if dest.exists():
                        continue
                    temp = dest.with_name(dest.name + '.part')
                    with urllib.request.urlopen(item['url'], timeout=120) as response, temp.open('wb') as output:
                        shutil.copyfileobj(response, output)
                    temp.replace(dest)
            for archive in source.get('extract', []):
                with zipfile.ZipFile(target / archive) as z:
                    for item in z.infolist():
                        destination = (target / item.filename).resolve()
                        if not destination.is_relative_to(target.resolve()) or (item.external_attr >> 16) & 0o170000 == 0o120000:
                            raise ValueError('Unsafe archive member')
                    z.extractall(target)
            (target / 'download_manifest.json').write_text(json.dumps(
                {k: source[k] for k in ('name', 'repo_id', 'revision', 'files') if k in source}, indent=2))
            print(f'{source["name"]}: {target}')
        except Exception as exc:
            failures.append(source['name'])
            print(f'{source["name"]}: FAILED ({type(exc).__name__}); verify access/HF_TOKEN, files and revision')
    if failures:
        raise SystemExit('Failed sources: ' + ', '.join(failures))


if __name__ == '__main__':
    main()
