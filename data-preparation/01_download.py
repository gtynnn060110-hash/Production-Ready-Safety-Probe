"""Download configured native splits and missing images into the local raw directory."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import io
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import struct
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
import zlib
from common import config


class AssetDownloadError(ValueError):
    pass


def path_mappings(cfg, source, part):
    mappings = []
    for item in source.get('image_path_maps', []) + part.get('image_path_maps', []):
        prefix = item['prefix']
        if not isinstance(prefix, str) or not prefix or not prefix.endswith('/'):
            raise ValueError('Image path mapping prefixes must be nonempty directories ending in /')
        target = Path(item['root'])
        if not target.is_absolute():
            target = Path(cfg.get('raw_dir', 'data/raw')) / target
        mappings.append((prefix, target))
    return mappings


def image_candidates(reference, root, mappings):
    original = Path(reference)
    candidates = [original if original.is_absolute() else root / original]
    if not original.is_absolute():
        for prefix, mapped_root in mappings:
            if reference.startswith(prefix):
                candidates.append(mapped_root / reference[len(prefix):])
    return candidates


class RangeFile(io.RawIOBase):
    """A seekable ZIP metadata reader; image workers use range_bytes directly."""

    def __init__(self, url):
        self.url = url
        self.position = 0
        self.size = None
        self.range_bytes(0, 0)

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        return self.position

    def read(self, size=-1):
        size = self.size - self.position if size < 0 else min(size, self.size - self.position)
        if size <= 0:
            return b''
        result = self.range_bytes(self.position, self.position + size - 1)
        self.position += len(result)
        return result

    def range_bytes(self, start, end):
        if end - start + 1 > 128 * 1024 * 1024:
            raise AssetDownloadError('Archive entry or metadata exceeds the 128 MiB read limit')
        # Public Drive caches can otherwise reuse a response for a different Range.
        parts = urllib.parse.urlsplit(self.url)
        query = urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
        query.append(('_range', f'{start}-{end}'))
        url = urllib.parse.urlunsplit(parts._replace(query=urllib.parse.urlencode(query)))
        for attempt in range(4):
            try:
                request = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}', 'Cache-Control': 'no-cache'})
                with urllib.request.urlopen(request, timeout=60) as response:
                    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
                    if response.status != 206 or not match:
                        raise AssetDownloadError('Archive server must honor HTTP byte ranges')
                    actual_start, actual_end, total = map(int, match.groups())
                    if (actual_start, actual_end) != (start, end) or (self.size is not None and total != self.size):
                        raise AssetDownloadError('Archive byte range or total size changed')
                    result = response.read(end - start + 2)
                    if len(result) != end - start + 1:
                        raise AssetDownloadError('Truncated archive byte range')
                    self.size = total
                    return result
            except urllib.error.HTTPError as exc:
                if exc.code not in (408, 429, 500, 502, 503, 504) or attempt == 3:
                    raise AssetDownloadError(f'Archive HTTP {exc.code}') from None
            except (urllib.error.URLError, TimeoutError):
                if attempt == 3:
                    raise AssetDownloadError('Archive network request failed after retries') from None
            time.sleep(2 ** attempt)


def archive_entry(remote, info, block=None):
    if info.flag_bits & 1 or (info.external_attr >> 16) & 0o170000 == 0o120000:
        raise AssetDownloadError('Encrypted or symlink ZIP entries are unsupported')
    if info.file_size > 128 * 1024 * 1024:
        raise AssetDownloadError('Image exceeds the 128 MiB extraction limit')
    if block is None:
        end = min(remote.size - 1, entry_end(info))
        block = remote.range_bytes(info.header_offset, end)
    header = struct.unpack('<4s5H3L2H', block[:30])
    if header[0] != b'PK\x03\x04' or header[3] != info.compress_type:
        raise AssetDownloadError('ZIP local header does not match its directory')
    offset = 30 + header[9] + header[10]
    needed = offset + info.compress_size
    if needed > len(block):
        block = bytes(block) + remote.range_bytes(info.header_offset + len(block), info.header_offset + needed - 1)
    compressed = block[offset:needed]
    if info.compress_type == zipfile.ZIP_STORED:
        raw = compressed
    elif info.compress_type == zipfile.ZIP_DEFLATED:
        decoder = zlib.decompressobj(-15)
        try:
            raw = decoder.decompress(compressed, info.file_size + 1)
        except zlib.error:
            raise AssetDownloadError('Invalid compressed ZIP entry') from None
        if not decoder.eof:
            raise AssetDownloadError('Incomplete or oversized compressed ZIP entry')
    else:
        raise AssetDownloadError('Unsupported image ZIP compression')
    if len(raw) != info.file_size or zlib.crc32(raw) & 0xffffffff != info.CRC:
        raise AssetDownloadError('Image ZIP size or CRC verification failed')
    return raw


def entry_end(info):
    return info.header_offset + 30 + len(info.filename.encode('utf-8')) + len(info.extra) + info.compress_size + 511


def range_groups(planned):
    """Coalesce adjacent members to avoid one high-latency request per image."""
    groups = []
    for item in sorted(planned, key=lambda item: item[1].header_offset):
        if groups:
            last = groups[-1]
            start = last[0][1].header_offset
            previous_end = max(entry_end(last[-1][1]), start)
            if item[1].header_offset - previous_end <= 64 * 1024 and entry_end(item[1]) - start <= 16 * 1024**2:
                last.append(item)
                continue
        groups.append([item])
    return groups


def matches_entry(path, info):
    if path.stat().st_size != info.file_size:
        return False
    checksum = 0
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            checksum = zlib.crc32(chunk, checksum)
    return checksum & 0xffffffff == info.CRC


def download_missing_images(cfg, source, target, workers=12):
    settings = source['image_archive']
    missing, seen = [], set()
    checked = 0
    for part in source['imports']:
        fields = {**source['fields'], **part.get('fields', {})}
        image_field = fields.get('images')
        if not image_field:
            continue
        root = target / part.get('image_root', source.get('image_root', ''))
        mappings = path_mappings(cfg, source, part)
        files = sorted(target.glob(part['glob']))
        if not files:
            raise AssetDownloadError(f'No image reference file matched: {part["glob"]}')
        for file in files:
            if file.suffix not in ('.csv', '.tsv'):
                raise AssetDownloadError('Image archive references currently require CSV/TSV imports')
            with file.open(encoding='utf-8-sig', newline='') as stream:
                for row in csv.DictReader(stream, delimiter='\t' if file.suffix == '.tsv' else ','):
                    reference = row.get(image_field)
                    if not isinstance(reference, str) or not reference.strip():
                        raise AssetDownloadError(f'Missing image reference in {file}: {image_field}')
                    marker = (root, reference)
                    if marker in seen:
                        continue
                    seen.add(marker)
                    relative = PurePosixPath(reference)
                    if relative.is_absolute() or '..' in relative.parts or '\\' in reference:
                        raise AssetDownloadError('Unsafe image archive reference')
                    existing = [p for p in image_candidates(reference, root, mappings) if p.is_file()]
                    if existing:
                        from PIL import Image
                        for path in existing:
                            with Image.open(path) as image:
                                image.verify()
                        checked += 1
                    else:
                        missing.append((root / relative, reference))
    print(f'{source["name"]}: {checked} existing images; {len(missing)} missing', flush=True)
    if not missing:
        return
    remote = RangeFile(settings['url'])
    if settings.get('size') is not None and remote.size != settings['size']:
        raise AssetDownloadError('Official image archive size differs from the configured version')
    with zipfile.ZipFile(remote) as archive:
        entries = {info.filename: info for info in archive.infolist()}
    planned = []
    for destination, reference in missing:
        info = entries.get(settings.get('prefix', '') + reference)
        if info is None:
            raise AssetDownloadError(f'Official image archive lacks reference: {reference}')
        planned.append((destination, info))

    def save(destination, info, raw):
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent, prefix='.' + destination.name, suffix='.part', delete=False) as stream:
                temp = Path(stream.name)
                stream.write(raw)
            if destination.exists():
                if not matches_entry(destination, info):
                    raise AssetDownloadError('Image destination changed during download')
            else:
                os.replace(temp, destination)
        finally:
            if temp is not None:
                temp.unlink(missing_ok=True)
        return len(raw)

    def fetch(group):
        start = group[0][1].header_offset
        end = min(remote.size - 1, max(entry_end(info) for _, info in group))
        block = memoryview(remote.range_bytes(start, end))
        total = 0
        for destination, info in group:
            raw = archive_entry(remote, info, block[info.header_offset - start:])
            total += save(destination, info, raw)
        return len(group), total

    total, count, reported = 0, 0, 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, group) for group in range_groups(planned)]
        try:
            for future in as_completed(futures):
                completed, size = future.result()
                count += completed
                total += size
                if count - reported >= 100 or count == len(planned):
                    print(f'{source["name"]}: image download {count}/{len(planned)} ({total / 1024**2:.1f} MiB)', flush=True)
                    reported = count
        except BaseException:
            for future in futures:
                future.cancel()
            raise


def extract_missing(archive, target):
    """Keep existing matching files and reject traversal or symlink archive entries."""
    with zipfile.ZipFile(archive) as zipped:
        for item in zipped.infolist():
            destination = target / item.filename
            if (not destination.resolve().is_relative_to(target.resolve())
                    or (item.external_attr >> 16) & 0o170000 == 0o120000):
                raise ValueError(f'Unsafe archive member: {item.filename}')
            if item.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            if destination.exists():
                if not destination.is_file() or not matches_entry(destination, item):
                    raise ValueError(f'Existing file differs from archive: {destination}')
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            temp = destination.with_name(destination.name + '.part')
            try:
                with zipped.open(item) as source, temp.open('wb') as output:
                    shutil.copyfileobj(source, output)
                temp.replace(destination)
            finally:
                temp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(Path(__file__).with_name('config.json')))
    parser.add_argument('--sources', nargs='+', help='Only download selected enabled source names')
    args = parser.parse_args()
    cfg = config(args.config)
    unknown = set(args.sources or ()) - {source['name'] for source in cfg['sources']}
    if unknown:
        parser.error('Unknown sources: ' + ', '.join(sorted(unknown)))
    selected = [s for s in cfg['sources'] if s.get('enabled', True)
                and (not args.sources or s['name'] in args.sources)]
    failures = set()
    for source in selected:
        target = Path(cfg['raw_dir']) / source['name']
        target.mkdir(parents=True, exist_ok=True)
        try:
            if source.get('repo_id'):
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id=source['repo_id'], repo_type='dataset',
                                  revision=source.get('revision', 'main'), local_dir=target,
                                  allow_patterns=source.get('allow_patterns'),
                                  ignore_patterns=source.get('ignore_patterns'))
            else:
                for item in source.get('files', []):
                    destination = target / item['path']
                    if not destination.resolve().is_relative_to(target.resolve()):
                        raise ValueError('Download path must stay inside its source directory')
                    if destination.is_file():
                        continue
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    temp = destination.with_name(destination.name + '.part')
                    try:
                        with urllib.request.urlopen(item['url'], timeout=120) as response, temp.open('wb') as output:
                            shutil.copyfileobj(response, output)
                        temp.replace(destination)
                    finally:
                        temp.unlink(missing_ok=True)
            for archive in source.get('extract', []):
                extract_missing(target / archive, target)
            print(f'{source["name"]}: files ready at {target}', flush=True)
        except Exception as exc:
            failures.add(source['name'])
            print(f'{source["name"]}: FAILED ({type(exc).__name__}): {exc}', flush=True)
    # Some source image paths map into another source: resolve them after all snapshots.
    for source in selected:
        if source.get('image_archive') and source['name'] not in failures:
            try:
                download_missing_images(cfg, source, Path(cfg['raw_dir']) / source['name'])
            except Exception as exc:
                failures.add(source['name'])
                print(f'{source["name"]}: image download FAILED ({type(exc).__name__}): {exc}', flush=True)
    if failures:
        raise SystemExit('Failed sources: ' + ', '.join(sorted(failures)))


if __name__ == '__main__':
    main()
