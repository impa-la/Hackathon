"""Safely validate and extract a ZIP or TAR.BZ2 archive into an empty directory."""

import argparse
import json
import pathlib
import stat
import tarfile
import zipfile


def ParseArguments():
    Parser = argparse.ArgumentParser()
    Parser.add_argument("ArchivePath")
    Parser.add_argument("DestinationPath")
    return Parser.parse_args()


def ValidateMemberName(MemberName):
    NormalizedName = MemberName.replace("\\", "/")
    MemberPath = pathlib.PurePosixPath(NormalizedName)
    if MemberPath.is_absolute():
        raise ValueError(f"Absolute archive path is not allowed: {MemberName}")
    if ".." in MemberPath.parts:
        raise ValueError(f"Parent traversal is not allowed: {MemberName}")
    if MemberPath.parts and ":" in MemberPath.parts[0]:
        raise ValueError(f"Drive-qualified archive path is not allowed: {MemberName}")


def EnsureEmptyDestination(DestinationPath):
    DestinationPath.mkdir(parents=True, exist_ok=True)
    if any(DestinationPath.iterdir()):
        raise FileExistsError(f"Destination is not empty: {DestinationPath}")


def ExtractZip(ArchivePath, DestinationPath):
    Names = set()
    MemberCount = 0
    FileCount = 0
    UncompressedBytes = 0
    with zipfile.ZipFile(ArchivePath) as Archive:
        BadMember = Archive.testzip()
        if BadMember is not None:
            raise ValueError(f"ZIP CRC failure: {BadMember}")
        for Member in Archive.infolist():
            ValidateMemberName(Member.filename)
            if Member.filename in Names:
                raise ValueError(f"Duplicate ZIP member: {Member.filename}")
            Names.add(Member.filename)
            UnixMode = (Member.external_attr >> 16) & 0xFFFF
            if UnixMode and stat.S_ISLNK(UnixMode):
                raise ValueError(f"ZIP symbolic link is not allowed: {Member.filename}")
        for Member in Archive.infolist():
            Archive.extract(Member, DestinationPath)
            MemberCount += 1
            if not Member.is_dir():
                FileCount += 1
                UncompressedBytes += Member.file_size
    return MemberCount, FileCount, UncompressedBytes


def ExtractTarBz2(ArchivePath, DestinationPath):
    Names = set()
    MemberCount = 0
    FileCount = 0
    UncompressedBytes = 0
    with tarfile.open(ArchivePath, mode="r|bz2") as Archive:
        for Member in Archive:
            ValidateMemberName(Member.name)
            if Member.name in Names:
                raise ValueError(f"Duplicate TAR member: {Member.name}")
            Names.add(Member.name)
            if Member.issym() or Member.islnk() or Member.isdev() or Member.isfifo():
                raise ValueError(f"Unsupported TAR member type: {Member.name}")
            Archive.extract(Member, DestinationPath, filter="data")
            MemberCount += 1
            if Member.isfile():
                FileCount += 1
                UncompressedBytes += Member.size
    return MemberCount, FileCount, UncompressedBytes


def Main():
    Arguments = ParseArguments()
    ArchivePath = pathlib.Path(Arguments.ArchivePath).resolve()
    DestinationPath = pathlib.Path(Arguments.DestinationPath).resolve()
    if not ArchivePath.is_file():
        raise FileNotFoundError(ArchivePath)
    EnsureEmptyDestination(DestinationPath)
    LowerName = ArchivePath.name.lower()
    if LowerName.endswith(".zip"):
        Result = ExtractZip(ArchivePath, DestinationPath)
    elif LowerName.endswith(".tar.bz2") or LowerName.endswith(".tbz2"):
        Result = ExtractTarBz2(ArchivePath, DestinationPath)
    else:
        raise ValueError(f"Unsupported archive type: {ArchivePath.name}")
    print(json.dumps({
        "ArchivePath": str(ArchivePath),
        "DestinationPath": str(DestinationPath),
        "MemberCount": Result[0],
        "FileCount": Result[1],
        "UncompressedBytes": Result[2],
    }, ensure_ascii=False))


if __name__ == "__main__":
    Main()
