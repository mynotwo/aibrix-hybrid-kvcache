"""File-local page-cache conditioning with residency evidence."""

import ctypes
import errno
import hashlib
import os
import sys
from pathlib import Path


def _resident_pages(path):
    if sys.platform != "linux":
        raise OSError(errno.ENOSYS, "mincore is Linux-only")
    path = Path(path)
    size = path.stat().st_size
    if size <= 0:
        raise OSError(errno.EINVAL, "cannot probe empty seed object")
    page_size = os.sysconf("SC_PAGE_SIZE")
    pages = (size + page_size - 1) // page_size
    length = pages * page_size
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_long,
    ]
    libc.mincore.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_ubyte),
    ]
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    fd = os.open(path, os.O_RDONLY)
    try:
        address = libc.mmap(None, length, 1, 1, fd, 0)
        if address == ctypes.c_void_p(-1).value:
            raise OSError(ctypes.get_errno(), "mmap failed")
        vector = (ctypes.c_ubyte * pages)()
        try:
            if libc.mincore(ctypes.c_void_p(address), length, vector) != 0:
                raise OSError(ctypes.get_errno(), "mincore failed")
            return [bool(value & 1) for value in vector]
        finally:
            libc.munmap(ctypes.c_void_p(address), length)
    finally:
        os.close(fd)


def prepare_page_cache(paths, mode, expected_hashes=None):
    """Use DONTNEED or a checked read, then verify pages with Linux mincore."""
    if mode not in ("cold", "warm"):
        raise ValueError("cache mode must be cold or warm")
    paths = [Path(path) for path in paths]
    if sys.platform != "linux" or not hasattr(os, "posix_fadvise"):
        return {
            "mode": mode,
            "status": "UNKNOWN",
            "verified": False,
            "reason": "file cache advice/mincore unavailable",
        }
    expected_hashes = expected_hashes or {}
    observations = []
    try:
        for path in paths:
            if path.is_symlink() or not path.is_file():
                raise ValueError("cache probe path is missing or a symlink")
            fd = os.open(path, os.O_RDONLY)
            try:
                if mode == "cold":
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                else:
                    digest = hashlib.sha256()
                    while chunk := os.read(fd, 1024 * 1024):
                        digest.update(chunk)
                    if (
                        path in expected_hashes
                        and digest.hexdigest() != expected_hashes[path]
                    ):
                        raise ValueError("warm-touch bytes failed hash check")
            finally:
                os.close(fd)
            pages = _resident_pages(path)
            expected = mode == "warm"
            observations.append(
                {
                    "path": str(path),
                    "resident_pages": sum(pages),
                    "page_count": len(pages),
                    "verified": all(page is expected for page in pages),
                }
            )
    except (OSError, ValueError) as exc:
        return {
            "mode": mode,
            "status": "UNKNOWN",
            "verified": False,
            "reason": f"file cache verification unavailable: {exc}",
            "files": observations,
        }
    verified = bool(observations) and all(
        item["verified"] for item in observations
    )
    return {
        "mode": mode,
        "status": "VERIFIED" if verified else "UNKNOWN",
        "verified": verified,
        "files": observations,
    }
