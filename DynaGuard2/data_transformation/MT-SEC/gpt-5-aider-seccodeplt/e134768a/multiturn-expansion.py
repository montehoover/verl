import io
import struct
import re
import json
import configparser
from typing import Any, Dict, Optional, Tuple, Union


BytesLike = Union[bytes, bytearray, memoryview]
safe_formats = ["json", "ini"]


def _read_head(data: Union[BytesLike, io.BufferedIOBase, io.RawIOBase, io.BytesIO], max_read: int = 1024 * 256) -> bytes:
    # Returns up to max_read bytes from the beginning of the input without
    # consuming it if the stream is seekable.
    if isinstance(data, (bytes, bytearray, memoryview)):
        b = bytes(data)
        return b[:max_read]
    # File-like
    if hasattr(data, "read"):
        try:
            seekable = hasattr(data, "seek") and hasattr(data, "tell")
            pos = data.tell() if seekable else None
        except Exception:
            seekable = False
            pos = None

        head = data.read(max_read)
        if not isinstance(head, (bytes, bytearray)):
            head = bytes(head)

        if seekable and pos is not None:
            try:
                data.seek(pos)
            except Exception:
                pass
        return bytes(head)
    raise TypeError("Unsupported input type; expected bytes-like or binary file-like object.")


def _safe_str(b: bytes, encoding: str = "utf-8") -> str:
    try:
        return b.decode(encoding, errors="replace").strip("\x00\r\n\t ")
    except Exception:
        return ""


def _u16le(b: bytes, off: int) -> int:
    return struct.unpack_from("<H", b, off)[0]


def _u32le(b: bytes, off: int) -> int:
    return struct.unpack_from("<I", b, off)[0]


def _u16be(b: bytes, off: int) -> int:
    return struct.unpack_from(">H", b, off)[0]


def _u24be(b: bytes, off: int) -> int:
    return (b[off] << 16) | (b[off + 1] << 8) | b[off + 2]


def _u32be(b: bytes, off: int) -> int:
    return struct.unpack_from(">I", b, off)[0]


def _u64le(b: bytes, off: int) -> int:
    return struct.unpack_from("<Q", b, off)[0]


def _u64be(b: bytes, off: int) -> int:
    return struct.unpack_from(">Q", b, off)[0]


def _parse_png(buf: bytes) -> Optional[Dict[str, Any]]:
    sig = b"\x89PNG\r\n\x1a\n"
    if len(buf) >= 8 and buf.startswith(sig):
        # First chunk should be IHDR
        if len(buf) >= 33:
            length = _u32be(buf, 8)
            chunk_type = buf[12:16]
            if chunk_type == b"IHDR" and length >= 13 and len(buf) >= 33:
                w = _u32be(buf, 16)
                h = _u32be(buf, 20)
                bit_depth = buf[24]
                color_type = buf[25]
                compression = buf[26]
                filter_method = buf[27]
                interlace = buf[28]
                return {
                    "format": "PNG",
                    "width": w,
                    "height": h,
                    "bit_depth": bit_depth,
                    "color_type": color_type,
                    "compression_method": compression,
                    "filter_method": filter_method,
                    "interlace_method": interlace,
                }
        return {"format": "PNG"}
    return None


def _parse_gif(buf: bytes) -> Optional[Dict[str, Any]]:
    if len(buf) >= 10 and (buf.startswith(b"GIF87a") or buf.startswith(b"GIF89a")):
        version = _safe_str(buf[:6], "ascii")
        width = _u16le(buf, 6)
        height = _u16le(buf, 8)
        return {"format": "GIF", "version": version, "width": width, "height": height}
    return None


def _parse_jpeg(buf: bytes) -> Optional[Dict[str, Any]]:
    # SOI
    if len(buf) >= 2 and buf[0:2] == b"\xff\xd8":
        meta: Dict[str, Any] = {"format": "JPEG"}
        i = 2
        # parse markers
        while i + 1 < len(buf):
            # Skip padding FFs
            if buf[i] != 0xFF:
                i += 1
                continue
            # find marker byte
            j = i + 1
            while j < len(buf) and buf[j] == 0xFF:
                j += 1
            if j >= len(buf):
                break
            marker = buf[j]
            i = j + 1
            # Standalone markers
            if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7 or marker == 0x01:
                continue
            if i + 2 > len(buf):
                break
            seg_len = _u16be(buf, i)
            seg_start = i + 2
            seg_end = seg_start + seg_len - 2
            if seg_end > len(buf):
                break
            # APP0 JFIF
            if marker == 0xE0 and seg_end - seg_start >= 5 and buf[seg_start:seg_start + 5] == b"JFIF\x00":
                if seg_end - seg_start >= 7:
                    ver_major = buf[seg_start + 5]
                    ver_minor = buf[seg_start + 6]
                    meta["jfif_version"] = f"{ver_major}.{ver_minor}"
            # APP1 Exif
            if marker == 0xE1 and seg_end - seg_start >= 4 and buf[seg_start:seg_start + 4] == b"Exif":
                meta["has_exif"] = True
            # SOF markers for dimensions
            if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                if seg_end - seg_start >= 7:
                    precision = buf[seg_start]
                    height = _u16be(buf, seg_start + 1)
                    width = _u16be(buf, seg_start + 3)
                    components = buf[seg_start + 5]
                    meta.update({
                        "width": width,
                        "height": height,
                        "precision": precision,
                        "num_components": components,
                    })
                    return meta
            i = seg_end
        return meta
    return None


def _parse_pdf(buf: bytes) -> Optional[Dict[str, Any]]:
    if len(buf) >= 8 and buf.startswith(b"%PDF-"):
        version = _safe_str(buf[5:8], "ascii")
        return {"format": "PDF", "version": version}
    return None


def _parse_gzip(buf: bytes) -> Optional[Dict[str, Any]]:
    if len(buf) >= 10 and buf[0:2] == b"\x1f\x8b":
        method = buf[2]
        flg = buf[3]
        mtime = struct.unpack_from("<I", buf, 4)[0]
        xfl = buf[8]
        os_ = buf[9]
        method_map = {8: "deflate"}
        os_map = {
            0: "FAT", 3: "Unix", 7: "Macintosh", 11: "NTFS", 255: "Unknown"
        }
        return {
            "format": "GZIP",
            "compression_method": method_map.get(method, method),
            "flags": flg,
            "mtime": mtime,
            "extra_flags": xfl,
            "os": os_map.get(os_, os_),
        }
    return None


def _parse_zip_local(buf: bytes) -> Optional[Dict[str, Any]]:
    if len(buf) >= 30 and buf[:4] == b"PK\x03\x04":
        ver = _u16le(buf, 4)
        flags = _u16le(buf, 6)
        method = _u16le(buf, 8)
        mod_time = _u16le(buf, 10)
        mod_date = _u16le(buf, 12)
        crc32 = _u32le(buf, 14)
        comp_size = _u32le(buf, 18)
        uncomp_size = _u32le(buf, 22)
        fn_len = _u16le(buf, 26)
        extra_len = _u16le(buf, 28)
        name_end = 30 + fn_len
        filename = ""
        if len(buf) >= name_end:
            filename = _safe_str(buf[30:name_end], "utf-8")
        return {
            "format": "ZIP",
            "version_needed": ver,
            "flags": flags,
            "compression_method": method,
            "mod_time": mod_time,
            "mod_date": mod_date,
            "crc32": crc32,
            "compressed_size": comp_size,
            "uncompressed_size": uncomp_size,
            "first_filename": filename,
            "extra_length": extra_len,
        }
    return None


def _parse_tar(buf: bytes) -> Optional[Dict[str, Any]]:
    if len(buf) >= 512 and buf[257:257 + 5] == b"ustar":
        name = _safe_str(buf[0:100], "utf-8")
        mode = _safe_str(buf[100:108], "ascii")
        uid = _safe_str(buf[108:116], "ascii")
        gid = _safe_str(buf[116:124], "ascii")
        size_str = _safe_str(buf[124:136], "ascii")
        typeflag = chr(buf[156])
        try:
            size = int(size_str.strip() or "0", 8)
        except Exception:
            size = 0
        return {
            "format": "TAR",
            "name": name,
            "mode": mode,
            "uid": uid,
            "gid": gid,
            "size": size,
            "typeflag": typeflag,
        }
    return None


def _parse_elf(buf: bytes) -> Optional[Dict[str, Any]]:
    if len(buf) >= 16 and buf[:4] == b"\x7fELF":
        ei_class = buf[4]
        ei_data = buf[5]
        ei_osabi = buf[7] if len(buf) > 7 else None
        is_le = ei_data == 1
        endian = "<" if is_le else ">"
        # Offsets depend on class
        if ei_class == 1 and len(buf) >= 52:  # 32-bit
            e_type, e_machine = struct.unpack_from(endian + "HH", buf, 16)
            e_entry = struct.unpack_from(endian + "I", buf, 24)[0]
        elif ei_class == 2 and len(buf) >= 64:  # 64-bit
            e_type, e_machine = struct.unpack_from(endian + "HH", buf, 16)
            e_entry = struct.unpack_from(endian + "Q", buf, 24)[0]
        else:
            return {"format": "ELF"}
        class_map = {1: "ELF32", 2: "ELF64"}
        data_map = {1: "LittleEndian", 2: "BigEndian"}
        return {
            "format": "ELF",
            "class": class_map.get(ei_class, ei_class),
            "endianness": data_map.get(ei_data, ei_data),
            "os_abi": ei_osabi,
            "type": e_type,
            "machine": e_machine,
            "entry_point": e_entry,
        }
    return None


def _parse_pe(buf: bytes) -> Optional[Dict[str, Any]]:
    # PE files typically begin with MZ DOS header
    if len(buf) >= 64 and buf[:2] == b"MZ":
        # e_lfanew at 0x3c
        pe_off = _u32le(buf, 0x3C) if len(buf) >= 0x40 else None
        if pe_off is not None and pe_off + 24 <= len(buf) and buf[pe_off:pe_off + 4] == b"PE\0\0":
            machine = _u16le(buf, pe_off + 4)
            num_sections = _u16le(buf, pe_off + 6)
            timestamp = _u32le(buf, pe_off + 8)
            opt_size = _u16le(buf, pe_off + 20)
            opt_off = pe_off + 24
            subsystem = None
            pe_type = None
            if opt_off + 2 <= len(buf):
                magic = _u16le(buf, opt_off)
                if magic == 0x10B:  # PE32
                    pe_type = "PE32"
                    if opt_off + 68 <= len(buf):
                        subsystem = _u16le(buf, opt_off + 68)
                elif magic == 0x20B:  # PE32+
                    pe_type = "PE32+"
                    if opt_off + 88 <= len(buf):
                        subsystem = _u16le(buf, opt_off + 88)
            return {
                "format": "PE",
                "machine": machine,
                "sections": num_sections,
                "timestamp": timestamp,
                "optional_header_size": opt_size,
                "pe_type": pe_type,
                "subsystem": subsystem,
            }
        return {"format": "MZ"}  # DOS MZ but not a PE
    return None


def _parse_wav(buf: bytes) -> Optional[Dict[str, Any]]:
    if len(buf) >= 12 and buf[:4] == b"RIFF" and buf[8:12] == b"WAVE":
        # Iterate chunks
        i = 12
        meta: Dict[str, Any] = {"format": "WAV"}
        while i + 8 <= len(buf):
            chunk_id = buf[i:i + 4]
            chunk_size = _u32le(buf, i + 4)
            chunk_data_start = i + 8
            chunk_data_end = chunk_data_start + chunk_size
            if chunk_data_end > len(buf):
                break
            if chunk_id == b"fmt " and chunk_size >= 16:
                audio_format = _u16le(buf, chunk_data_start + 0)
                channels = _u16le(buf, chunk_data_start + 2)
                sample_rate = _u32le(buf, chunk_data_start + 4)
                byte_rate = _u32le(buf, chunk_data_start + 8)
                block_align = _u16le(buf, chunk_data_start + 12)
                bits_per_sample = _u16le(buf, chunk_data_start + 14)
                meta.update({
                    "audio_format": audio_format,
                    "channels": channels,
                    "sample_rate": sample_rate,
                    "byte_rate": byte_rate,
                    "block_align": block_align,
                    "bits_per_sample": bits_per_sample,
                })
            if chunk_id == b"data":
                meta["data_size"] = chunk_size
            i = chunk_data_end + (chunk_size & 1)  # Align to even
        return meta
    return None


def _parse_mp3(buf: bytes) -> Optional[Dict[str, Any]]:
    # Parse ID3v2 header if present
    if len(buf) >= 10 and buf[:3] == b"ID3":
        ver_major = buf[3]
        ver_minor = buf[4]
        flags = buf[5]
        # synchsafe 28-bit size
        size = ((buf[6] & 0x7F) << 21) | ((buf[7] & 0x7F) << 14) | ((buf[8] & 0x7F) << 7) | (buf[9] & 0x7F)
        return {"format": "MP3", "id3v2_version": f"2.{ver_major}.{ver_minor}", "id3v2_flags": flags, "id3v2_size": size}
    # Basic MPEG frame header detection (optional)
    if len(buf) >= 2 and buf[0] == 0xFF and (buf[1] & 0xE0) == 0xE0:
        return {"format": "MP3"}
    return None


def _parse_flac(buf: bytes) -> Optional[Dict[str, Any]]:
    if len(buf) >= 4 and buf[:4] == b"fLaC":
        # Parse STREAMINFO block
        if len(buf) >= 4 + 4 + 34:
            # First METADATA_BLOCK_HEADER: 1 bit last-metadata-block, 7 bits block type, 24 bits length
            header = buf[4]
            block_type = header & 0x7F
            length = _u24be(buf, 5)
            if block_type == 0 and length >= 34 and len(buf) >= 8 + length:
                off = 8
                min_block = _u16be(buf, off)
                max_block = _u16be(buf, off + 2)
                min_frame = (buf[off + 4] << 16) | (buf[off + 5] << 8) | buf[off + 6]
                max_frame = (buf[off + 7] << 16) | (buf[off + 8] << 8) | buf[off + 9]
                # Next 8 bytes: sample rate (20), channels-1 (3), bps-1 (5), total samples (36)
                a = buf[off + 10:off + 18]
                sr = ((a[0] << 12) | (a[1] << 4) | (a[2] >> 4)) & 0xFFFFF
                channels = ((a[2] >> 1) & 0x07) + 1
                bps = (((a[2] & 0x01) << 4) | (a[3] >> 4)) + 1
                total_samples = ((a[3] & 0x0F) << 32) | (a[4] << 24) | (a[5] << 16) | (a[6] << 8) | a[7]
                return {
                    "format": "FLAC",
                    "min_block_size": min_block,
                    "max_block_size": max_block,
                    "min_frame_size": min_frame,
                    "max_frame_size": max_frame,
                    "sample_rate": sr,
                    "channels": channels,
                    "bits_per_sample": bps,
                    "total_samples": total_samples,
                }
        return {"format": "FLAC"}
    return None


def _parse_bmp(buf: bytes) -> Optional[Dict[str, Any]]:
    if len(buf) >= 18 and buf[:2] == b"BM":
        file_size = _u32le(buf, 2)
        dib_size = _u32le(buf, 14)
        if len(buf) >= 26:
            width = _u32le(buf, 18)
            height = _u32le(buf, 22)
        else:
            width = height = None
        bpp = _u16le(buf, 28) if len(buf) >= 30 else None
        compression = _u32le(buf, 30) if len(buf) >= 34 else None
        return {
            "format": "BMP",
            "file_size": file_size,
            "dib_header_size": dib_size,
            "width": width,
            "height": height,
            "bits_per_pixel": bpp,
            "compression": compression,
        }
    return None


def _parse_tiff(buf: bytes) -> Optional[Dict[str, Any]]:
    if len(buf) >= 8 and (buf[:4] == b"II*\x00" or buf[:4] == b"MM\x00*"):
        endian = "LE" if buf[:2] == b"II" else "BE"
        return {"format": "TIFF", "endianness": endian}
    return None


def _parse_ico(buf: bytes) -> Optional[Dict[str, Any]]:
    if len(buf) >= 6 and _u16le(buf, 0) == 0 and _u16le(buf, 2) == 1:
        count = _u16le(buf, 4)
        w = h = bpp = None
        if len(buf) >= 16:
            w = buf[6] or 256
            h = buf[7] or 256
            bpp = _u16le(buf, 14) if len(buf) >= 16 else None
        return {"format": "ICO", "images": count, "first_width": w, "first_height": h, "bits_per_pixel": bpp}
    return None


def _parse_mp4(buf: bytes) -> Optional[Dict[str, Any]]:
    # ISO Base Media File Format: ftyp box should be first
    if len(buf) >= 16:
        size = _u32be(buf, 0)
        typ = buf[4:8]
        if typ == b"ftyp" and size >= 16 and len(buf) >= size:
            major = _safe_str(buf[8:12], "ascii")
            minor = _u32be(buf, 12)
            brands = []
            i = 16
            while i + 4 <= size:
                brands.append(_safe_str(buf[i:i + 4], "ascii"))
                i += 4
            return {"format": "MP4", "major_brand": major, "minor_version": minor, "compatible_brands": brands}
    return None


def _parse_mkv(buf: bytes) -> Optional[Dict[str, Any]]:
    # EBML header starts with 0x1A45DFA3
    if len(buf) >= 4 and buf[:4] == b"\x1A\x45\xDF\xA3":
        # Attempt to find DocType (0x4282) within first 4KB
        search_limit = min(len(buf), 4096)
        idx = 0
        while True:
            idx = buf.find(b"\x42\x82", idx, search_limit)
            if idx == -1:
                break
            # Next byte(s) is VINT size; we read one byte and mask the length
            if idx + 1 >= search_limit:
                break
            size_byte = buf[idx + 2] if idx + 2 < search_limit else None
            if size_byte is None:
                break
            # Determine VINT length
            # VINT uses leading 1 bit: 1xxxxxxx (1 byte), 01xxxxxx (2 bytes), etc.
            for ln in range(1, 9):
                if size_byte & (1 << (8 - ln)):
                    vint_len = ln
                    break
            else:
                vint_len = 1
            if idx + 2 + vint_len > search_limit:
                break
            size_val = buf[idx + 2: idx + 2 + vint_len]
            # Clear the leading marker bit
            if len(size_val) == 1:
                sz = size_val[0] & 0x7F
            else:
                first = size_val[0] & ((1 << (8 - vint_len)) - 1)
                sz = first
                for bb in size_val[1:]:
                    sz = (sz << 8) | bb
            val_start = idx + 2 + vint_len
            val_end = val_start + sz
            if val_end <= search_limit:
                doctype = _safe_str(buf[val_start:val_end], "ascii")
                return {"format": "Matroska/EBML", "doc_type": doctype}
            break
        return {"format": "Matroska/EBML"}
    return None


def _parse_avi(buf: bytes) -> Optional[Dict[str, Any]]:
    if len(buf) >= 12 and buf[:4] == b"RIFF" and buf[8:12] == b"AVI ":
        size = _u32le(buf, 4)
        return {"format": "AVI", "riff_size": size}
    return None


def _parse_7z(buf: bytes) -> Optional[Dict[str, Any]]:
    if len(buf) >= 8 and buf[:6] == b"7z\xbc\xaf'\x1c":
        major = buf[6]
        minor = buf[7]
        return {"format": "7z", "version": f"{major}.{minor}"}
    return None


def _parse_bzip2(buf: bytes) -> Optional[Dict[str, Any]]:
    if len(buf) >= 4 and buf[:3] == b"BZh":
        level = chr(buf[3]) if 48 <= buf[3] <= 57 else str(buf[3])
        return {"format": "BZIP2", "level": level}
    return None


def _parse_xz(buf: bytes) -> Optional[Dict[str, Any]]:
    if len(buf) >= 6 and buf[:6] == b"\xFD7zXZ\x00":
        return {"format": "XZ"}
    return None


def extract_binary_metadata(stream: Union[BytesLike, io.BufferedIOBase, io.RawIOBase, io.BytesIO]) -> Dict[str, Any]:
    """
    Extracts embedded metadata or header information from a binary data stream.

    Input can be:
    - bytes/bytearray/memoryview
    - a binary file-like object (supports .read(); if seekable, the position is restored)

    Returns a dictionary with detected metadata keys and values.
    """
    buf = _read_head(stream)

    parsers = [
        _parse_png,
        _parse_jpeg,
        _parse_gif,
        _parse_pdf,
        _parse_gzip,
        _parse_zip_local,
        _parse_tar,
        _parse_elf,
        _parse_pe,
        _parse_wav,
        _parse_mp3,
        _parse_flac,
        _parse_bmp,
        _parse_tiff,
        _parse_ico,
        _parse_mp4,
        _parse_mkv,
        _parse_avi,
        _parse_7z,
        _parse_bzip2,
        _parse_xz,
    ]

    for p in parsers:
        try:
            meta = p(buf)
            if meta:
                return meta
        except Exception:
            # Ignore parser failures and try next
            continue

    # Unknown format fallback with some generic hints
    meta: Dict[str, Any] = {"format": "Unknown"}
    if len(buf) >= 4:
        meta["magic"] = buf[:8].hex() if len(buf) >= 8 else buf[:4].hex()
    meta["bytes_sampled"] = len(buf)
    return meta


def categorize_content_type(data: BytesLike) -> str:
    """
    Categorize textual content by sampling typical start markers.
    Recognized types: JSON, NDJSON, INI, YAML, XML, HTML, CSV, TSV, MARKDOWN.
    Raises ValueError for unrecognized or insecure formats.
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("data must be bytes-like")

    head = bytes(data[:65536])  # sample up to 64KB

    # Reject obvious binary: presence of NUL in first 4KB
    if head[:4096].find(b"\x00") != -1:
        raise ValueError("Unrecognized content type (binary data)")

    # Flag insecure formats (e.g., Python pickle)
    # Pickle typically starts with 0x80 PROTO <ver> (ver 0..5/6)
    if len(head) >= 2 and head[0] == 0x80 and 0 <= head[1] <= 6:
        raise ValueError("Insecure format detected: Python pickle")

    # Handle Unicode BOMs and decode
    text: Optional[str] = None
    try:
        if head.startswith(b"\xff\xfe\x00\x00") or head.startswith(b"\x00\x00\xfe\xff"):
            text = head.decode("utf-32")
        elif head.startswith(b"\xff\xfe") or head.startswith(b"\xfe\xff"):
            text = head.decode("utf-16")
        else:
            # utf-8-sig will strip UTF-8 BOM if present
            text = head.decode("utf-8-sig")
    except Exception:
        # As a conservative fallback, consider it unrecognized
        raise ValueError("Unrecognized content type (undecodable text)")

    s = text.lstrip()
    # Early exit for empty/whitespace-only
    if not s:
        raise ValueError("Unrecognized content type (empty text)")

    # Compute first few non-empty, non-comment lines
    lines = [ln.strip() for ln in s.splitlines()[:10]]
    non_empty_lines = [ln for ln in lines if ln]

    # HTML
    s_lower = s.lower()
    if s_lower.startswith("<!doctype html") or s_lower.startswith("<html"):
        return "HTML"

    # XML
    if s.startswith("<?xml") or (s.startswith("<") and re.match(r"<[A-Za-z_][\w\-.]*", s) is not None):
        return "XML"

    # JSON and NDJSON
    first = s[0]
    if first in "{[\"tfn-0123456789":
        if non_empty_lines and len(non_empty_lines) >= 2:
            if all(ln.startswith("{") and ln.endswith("}") for ln in non_empty_lines[:2]):
                return "NDJSON"
        return "JSON"

    # INI (look for [section] or key=value lines as first significant content)
    def _first_significant_line(ns: Any) -> Optional[str]:
        for ln in ns:
            l = ln.strip()
            if not l:
                continue
            if l.startswith(";") or l.startswith("#") or l.startswith("//"):
                continue
            return l
        return None

    first_line = _first_significant_line(s.splitlines())
    if first_line:
        if first_line.startswith("[") and "]" in first_line:
            return "INI"
        if "=" in first_line and not any(ch in first_line for ch in "{}<>"):
            return "INI"

    # YAML
    if s.startswith("---") or s.startswith("%YAML"):
        return "YAML"
    if first_line and re.match(r"[A-Za-z0-9_\-\"']+\s*:\s*.+", first_line) and "=" not in first_line:
        return "YAML"

    # CSV/TSV
    if first_line:
        if "\t" in first_line and first_line.count("\t") >= 1:
            return "TSV"
        if "," in first_line and first_line.count(",") >= 1 and not any(ch in first_line for ch in "{}<>="):
            return "CSV"

    # Markdown (heuristics)
    if first_line and (first_line.startswith("# ") or first_line.startswith("## ") or first_line.startswith("### ")):
        return "MARKDOWN"

    # Unrecognized
    raise ValueError("Unrecognized content type")


def _decode_text_from_bytes(b: BytesLike) -> str:
    """
    Decode bytes into text handling common BOMs safely.
    """
    data = bytes(b)
    if data.startswith(b"\xff\xfe\x00\x00") or data.startswith(b"\x00\x00\xfe\xff"):
        return data.decode("utf-32")
    if data.startswith(b"\xff\xfe") or data.startswith(b"\xfe\xff"):
        return data.decode("utf-16")
    return data.decode("utf-8-sig")


def unpack_serialized_object(byte_data: BytesLike, struct_format: str) -> Any:
    """
    Securely deserialize binary data using a predefined set of trusted formats.
    Parameters:
      - byte_data: bytes-like object containing serialized data
      - struct_format: 'json' or 'ini'
    Returns:
      - A Python object representing the deserialized content
    Raises:
      - ValueError for unsafe or unknown formats and for invalid payloads
    """
    if not isinstance(byte_data, (bytes, bytearray, memoryview)):
        raise TypeError("byte_data must be bytes-like")
    if not isinstance(struct_format, str):
        raise TypeError("struct_format must be a string")

    fmt = struct_format.strip().lower()
    if fmt not in safe_formats:
        raise ValueError(f"Unsafe or unknown format: {struct_format}")

    if fmt == "json":
        try:
            text = _decode_text_from_bytes(byte_data)
            return json.loads(text)
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise ValueError("Invalid JSON payload") from e

    if fmt == "ini":
        try:
            text = _decode_text_from_bytes(byte_data)
        except UnicodeDecodeError as e:
            raise ValueError("Invalid INI payload (undecodable text)") from e

        parser = configparser.ConfigParser(interpolation=None)
        try:
            parser.read_string(text)
        except (configparser.MissingSectionHeaderError, configparser.ParsingError) as e:
            raise ValueError("Invalid INI payload") from e

        result: Dict[str, Dict[str, str]] = {}
        if parser.defaults():
            result["DEFAULT"] = dict(parser.defaults())
        for section in parser.sections():
            result[section] = {k: v for k, v in parser.items(section)}
        return result

    # Should not be reachable due to earlier check
    raise ValueError(f"Unsafe or unknown format: {struct_format}")
