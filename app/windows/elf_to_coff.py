"""Convert GCC's simple freestanding AMD64 object to COFF, retaining PC-relative addends.

GNU objcopy can turn ELF R_X86_64_PC32 section relocations into absolute COFF
relocations and discard unnamed section symbols. This deliberately narrow
converter rejects relocations/sections beyond this launcher's requirements.
"""
from __future__ import annotations

from pathlib import Path
import struct


def convert(source: Path, destination: Path) -> None:
    elf = source.read_bytes()
    if elf[:6] != b"\x7fELF\x02\x01" or struct.unpack_from("<H", elf, 18)[0] != 62:
        raise ValueError("Expected little-endian ELF64 AMD64")
    table = struct.unpack_from("<Q", elf, 40)[0]
    entry_size, count, name_index = struct.unpack_from("<HHH", elf, 58)
    sections = [struct.unpack_from("<IIQQQQIIQQ", elf, table + i * entry_size)
                for i in range(count)]
    name_section = sections[name_index]
    names = elf[name_section[4]:name_section[4] + name_section[5]]

    def string(data: bytes, offset: int) -> str:
        return data[offset:data.index(0, offset)].decode("ascii")

    def section_name(index):
        return string(names, sections[index][0])

    allocated = [index for index, section in enumerate(sections) if section[2] & 2]
    if any(sections[index][1] not in (1, 8) for index in allocated):
        raise ValueError("Unsupported allocated ELF section")
    mapping = {index: i + 1 for i, index in enumerate(allocated)}
    symbols_section = next(section for section in sections if section[1] == 2)
    strings_section = sections[symbols_section[6]]
    symbol_strings = elf[strings_section[4]:strings_section[4] + strings_section[5]]
    symbols = []
    symbol_map = {}
    for index, offset in enumerate(range(symbols_section[4],
                                         symbols_section[4] + symbols_section[5], 24)):
        name, info, _, section, value, _ = struct.unpack_from("<IBBHQQ", elf, offset)
        kind, binding = info & 15, info >> 4
        if index == 0 or kind == 4:
            continue
        if section != 0 and section not in mapping:
            continue
        name = section_name(section) if kind == 3 else string(symbol_strings, name)
        if not name:
            raise ValueError("Unnamed symbol")
        symbol_map[index] = len(symbols)
        symbols.append((name, value, mapping.get(section, 0), 0x20 if kind == 2 else 0,
                        2 if binding else 3))

    payloads = {index: bytearray(elf[sections[index][4]:
                                        sections[index][4] + sections[index][5]])
                if sections[index][1] == 1 else bytearray() for index in allocated}
    relocations = {index: [] for index in allocated}
    for section in sections:
        if section[1] != 4 or section[7] not in mapping:
            continue
        target = section[7]
        for offset in range(section[4], section[4] + section[5], 24):
            address, info, addend = struct.unpack_from("<QQq", elf, offset)
            symbol, kind = info >> 32, info & 0xffffffff
            if symbol not in symbol_map:
                raise ValueError("Relocation references an unsupported symbol")
            if kind in (2, 4):  # ELF S+A-P; COFF REL32 S+A-P-4.
                code, form, value = 4, "<i", addend + 4
            elif kind == 1:
                code, form, value = 1, "<q", addend
            else:
                raise ValueError("Unsupported AMD64 relocation: " + str(kind))
            struct.pack_into(form, payloads[target], address, value)
            relocations[target].append(struct.pack("<IIH", address, symbol_map[symbol], code))

    pool = bytearray(4)
    string_offsets = {}

    def intern(name):
        if name not in string_offsets:
            string_offsets[name] = len(pool)
            pool.extend(name.encode("ascii") + b"\0")
        return string_offsets[name]

    def coff_name(name, is_section=False):
        encoded = name.encode("ascii")
        if len(encoded) <= 8:
            return encoded.ljust(8, b"\0")
        offset = intern(name)
        return (("/" + str(offset)).encode("ascii").ljust(8, b"\0") if is_section
                else struct.pack("<II", 0, offset))

    headers = []
    body = bytearray()
    cursor = 20 + len(allocated) * 40
    for index in allocated:
        section = sections[index]
        data = payloads[index]
        relocation_data = b"".join(relocations[index])
        data_offset = cursor if data else 0
        cursor += len(data)
        relocation_offset = cursor if relocation_data else 0
        cursor += len(relocation_data)
        body.extend(data)
        body.extend(relocation_data)
        if section[1] == 8:
            flags = 0xc0000080
        elif section[2] & 4:
            flags = 0x60000020
        else:
            flags = 0x40000040 | (0x80000000 if section[2] & 1 else 0)
        alignment = max(1, section[8])
        if alignment & (alignment - 1) or alignment > 8192:
            raise ValueError("Unsupported section alignment")
        flags |= alignment.bit_length() << 20
        headers.append(struct.pack("<8sIIIIIIHHI", coff_name(section_name(index), True),
                                   0, 0, section[5], data_offset, relocation_offset, 0,
                                   len(relocations[index]), 0, flags))
    symbol_bytes = b"".join(struct.pack("<8sIhHBB", coff_name(name), value, section, kind,
                                       storage, 0)
                             for name, value, section, kind, storage in symbols)
    struct.pack_into("<I", pool, 0, len(pool))
    header = struct.pack("<HHIIIHH", 0x8664, len(allocated), 0, cursor, len(symbols), 0, 0)
    destination.write_bytes(header + b"".join(headers) + body + symbol_bytes + pool)
