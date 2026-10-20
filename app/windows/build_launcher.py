"""Rebuild the x64 Windows GUI launcher using GNU GCC/binutils on Linux.

Only launcher.c is compiled. Temporary import libraries describe standard Windows
system functions; their stub DLLs are never copied into the application.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile

from elf_to_coff import convert


SOURCE = Path(__file__).resolve().with_name("launcher.c")
OUTPUT = SOURCE.parent.parent / "AlfaLab.exe"
IMPORTS = {
    "kernel32": ["GetModuleFileNameW", "GetFileAttributesW", "CreateProcessW",
                 "CloseHandle", "ExitProcess"],
    "user32": ["MessageBoxW"],
}


def run(*args):
    try:
        return subprocess.run([str(arg) for arg in args], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(error.stderr.strip() or str(error)) from error


def verify_pe(path: Path) -> dict:
    data = path.read_bytes()
    if data[:2] != b"MZ":
        raise ValueError("Missing DOS header")
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe:pe + 4] != b"PE\0\0":
        raise ValueError("Missing PE signature")
    machine, sections = struct.unpack_from("<HH", data, pe + 4)
    optional_size = struct.unpack_from("<H", data, pe + 20)[0]
    optional = pe + 24
    magic = struct.unpack_from("<H", data, optional)[0]
    entry = struct.unpack_from("<I", data, optional + 16)[0]
    subsystem, characteristics = struct.unpack_from("<HH", data, optional + 68)
    if (machine, magic, subsystem) != (0x8664, 0x20B, 2):
        raise ValueError("Expected Windows x64 PE32+ GUI executable")
    if not entry or characteristics & 0x140 != 0x140:
        raise ValueError("Expected entry point, ASLR and DEP flags")
    if struct.unpack_from("<II", data, optional + 112 + 14 * 8) != (0, 0):
        raise ValueError("The launcher must not require .NET")

    section_table = optional + optional_size
    regions = []
    for index in range(sections):
        location = section_table + index * 40
        virtual_size, rva, size, offset = struct.unpack_from("<IIII", data, location + 8)
        regions.append((rva, max(virtual_size, size), offset))

    def position(rva):
        for start, size, offset in regions:
            if start <= rva < start + size:
                return offset + rva - start
        raise ValueError("Invalid RVA")

    def cstring(rva):
        start = position(rva)
        return data[start:data.index(0, start)].decode("ascii")

    import_rva, _ = struct.unpack_from("<II", data, optional + 112 + 8)
    imports = {}
    descriptor = position(import_rva)
    while any(data[descriptor:descriptor + 20]):
        lookup, _, _, dll_name, first_thunk = struct.unpack_from("<IIIII", data, descriptor)
        name = cstring(dll_name).lower()
        cursor = position(lookup or first_thunk)
        functions = []
        while value := struct.unpack_from("<Q", data, cursor)[0]:
            if value >> 63:
                raise ValueError("Unexpected ordinal import")
            functions.append(cstring(value + 2))
            cursor += 8
        imports[name] = functions
        descriptor += 20
    expected = {name + ".dll": sorted(values) for name, values in IMPORTS.items()}
    if {name: sorted(values) for name, values in imports.items()} != expected:
        raise ValueError("Unexpected import table: " + str(imports))
    return {"machine": "AMD64", "format": "PE32+", "subsystem": "Windows GUI",
            "imports": imports, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def build(output: Path = OUTPUT) -> dict:
    for tool in ("gcc", "as", "objcopy", "ld"):
        if shutil.which(tool) is None:
            raise RuntimeError("Missing GNU build tool: " + tool)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="launcher-build-", dir=output.parent) as directory:
        temporary = Path(directory)
        libraries = []
        for name, functions in IMPORTS.items():
            definition = temporary / (name + ".def")
            definition.write_text("LIBRARY " + name + ".dll\nEXPORTS\n" +
                                  "\n".join(functions) + "\n", encoding="ascii")
            assembly = temporary / (name + ".s")
            assembly.write_text(".text\n" + "".join(
                ".globl " + symbol + "\n" + symbol + ":\nret\n" for symbol in functions),
                encoding="ascii")
            elf = temporary / (name + ".elf.o")
            coff = temporary / (name + ".o")
            library = temporary / ("lib" + name + ".a")
            run("as", "--64", "-o", elf, assembly)
            run("objcopy", "-O", "pe-x86-64", elf, coff)
            run("ld", "-mi386pep", "--dll", "--entry", "0", "--no-insert-timestamp",
                "--out-implib", library, "-o", temporary / (name + ".dll"), coff, definition)
            libraries.append(library)
        elf = temporary / "launcher.elf.o"
        coff = temporary / "launcher.o"
        target = temporary / "AlfaLab.exe"
        run("gcc", "-c", "-Os", "-Wall", "-Wextra", "-Werror", "-ffreestanding",
            "-fshort-wchar", "-fpie", "-mno-red-zone", "-maccumulate-outgoing-args",
            "-fcf-protection=none",
            "-fno-asynchronous-unwind-tables", "-fno-unwind-tables", "-fno-stack-protector",
            "-o", elf, SOURCE)
        convert(elf, coff)
        run("ld", "-mi386pep", "--entry", "start", "--subsystem", "windows",
            "--dynamicbase", "--nxcompat", "--no-insert-timestamp", "--strip-all",
            "-o", target, coff, *libraries)
        result = verify_pe(target)
        shutil.copyfile(target, output)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    result = verify_pe(args.verify) if args.verify else build(args.output)
    import json

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
