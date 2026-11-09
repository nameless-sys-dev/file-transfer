import modal

app = modal.App("quick-transfer")

image = (
    modal.Image.from_registry("ubuntu:24.04", add_python="3.12")
    .pip_install(
        "fastapi>=0.115.6",
        "anyio>=4.4",
        "argon2-cffi>=23.1",
    )
)

volume = modal.Volume.from_name(
    "quick-transfer-data",
    create_if_missing=True,
)

MAX_FILE_BYTES = 2 * 1024**3  # 2 GiB per file; adjust as needed
IO_BLOCK = 1024**2           # 1 MiB application I/O buffers


@app.function(
    image=image,
    volumes={"/data": volume},
    cpu=2,
    memory=2048,
    timeout=3600,
    max_containers=1,        # Important: this design uses one storage writer
    scaledown_window=300,
)
@modal.concurrent(max_inputs=32)
@modal.asgi_app()
def web():
    import asyncio
    import json
    import os
    import secrets
    import shutil
    from pathlib import Path
    from typing import Annotated

    import anyio
    from argon2 import PasswordHasher
    from argon2.exceptions import VerifyMismatchError
    from fastapi import (
        FastAPI,
        Header,
        HTTPException,
        Path as APIPath,
        Query,
        Request,
        Response,
    )
    from fastapi.responses import FileResponse
    from pydantic import BaseModel

    api = FastAPI(
        title="Quick Transfer",
        version="1.0.0",
        description=(
            "Public file uploads and downloads. An optional password protects "
            "deletion only. Upload bodies are raw bytes, not multipart/form-data."
        ),
    )

    root = Path("/data/files")
    root.mkdir(parents=True, exist_ok=True)

    # Only short metadata operations and durable publication/deletion use this
    # lock. Upload/download byte transfers run concurrently outside it.
    gate = asyncio.Lock()
    readers: dict[str, int] = {}

    # PasswordHasher uses salted Argon2id by default.
    hasher = PasswordHasher(
        time_cost=3,
        memory_cost=65536,  # 64 MiB per password operation
        parallelism=1,
    )
    hash_slots = asyncio.Semaphore(2)

    FileID = Annotated[
        str,
        APIPath(pattern=r"^[0-9a-f]{32}$"),
    ]
    DeletePassword = Annotated[
        str | None,
        Header(
            alias="X-Delete-Password",
            min_length=1,
            max_length=1024,
            description="Optional on upload; required to delete a protected file.",
        ),
    ]

    class UploadResult(BaseModel):
        id: str
        filename: str
        size_bytes: int
        delete_protected: bool
        download_url: str

    def read_meta(file_id):
        try:
            return json.loads(
                (root / file_id / "meta.json").read_text(encoding="utf-8")
            )
        except FileNotFoundError:
            raise HTTPException(404, "File not found") from None

    class DownloadResponse(FileResponse):
        chunk_size = IO_BLOCK

        def __init__(self, file_id, **kwargs):
            super().__init__(**kwargs)
            self.file_id = file_id

        async def __call__(self, scope, receive, send):
            try:
                await super().__call__(scope, receive, send)
            finally:
                # Release even if the client disconnects or the transfer fails.
                with anyio.CancelScope(shield=True):
                    async with gate:
                        readers[self.file_id] -= 1
                        if readers[self.file_id] == 0:
                            del readers[self.file_id]

    @api.post(
        "/files",
        status_code=201,
        response_model=UploadResult,
        responses={
            400: {"description": "Invalid or mismatched Content-Length"},
            413: {"description": "File exceeds the configured size limit"},
            415: {"description": "Use Content-Type: application/octet-stream"},
        },
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    "application/octet-stream": {
                        "schema": {"type": "string", "format": "binary"}
                    }
                },
            }
        },
    )
    async def upload(
        request: Request,
        response: Response,
        filename: Annotated[
            str,
            Query(
                min_length=1,
                max_length=255,
                pattern=r"^[^/\\\x00-\x1f\x7f]+$",
                description="Download filename, without directory components.",
            ),
        ],
        delete_password: DeletePassword = None,
    ):
        media_type = (
            request.headers.get("content-type", "")
            .split(";", 1)[0]
            .strip()
            .lower()
        )
        if media_type != "application/octet-stream":
            raise HTTPException(415, "Send raw bytes as application/octet-stream")

        declared_length = None
        length_header = request.headers.get("content-length")
        if length_header is not None:
            try:
                declared_length = int(length_header)
            except ValueError:
                raise HTTPException(400, "Invalid Content-Length") from None
            if declared_length < 0:
                raise HTTPException(400, "Invalid Content-Length")
            if declared_length > MAX_FILE_BYTES:
                raise HTTPException(413, "File too large")

        delete_hash = None
        if delete_password is not None:
            async with hash_slots:
                delete_hash = await anyio.to_thread.run_sync(
                    hasher.hash, delete_password
                )

        file_id = secrets.token_hex(16)
        staging = root / f".incoming-{file_id}"
        destination = root / file_id
        size = 0

        try:
            await anyio.to_thread.run_sync(staging.mkdir)

            # Raw streaming avoids multipart's temporary-file spooling and copy.
            async with await anyio.open_file(staging / "data", "wb") as output:
                buffer = bytearray()

                async for chunk in request.stream():
                    size += len(chunk)
                    if size > MAX_FILE_BYTES:
                        raise HTTPException(413, "File too large")

                    buffer.extend(chunk)
                    if len(buffer) >= IO_BLOCK:
                        await output.write(buffer)
                        buffer.clear()

                if buffer:
                    await output.write(buffer)

            if declared_length is not None and declared_length != size:
                raise HTTPException(400, "Content-Length does not match body")

            meta = {
                "filename": filename,
                "size_bytes": size,
                "delete_hash": delete_hash,
            }
            await anyio.to_thread.run_sync(
                (staging / "meta.json").write_text,
                json.dumps(meta),
                "utf-8",
            )

            # Publish only completed files, then persist before acknowledging.
            with anyio.CancelScope(shield=True):
                async with gate:
                    await anyio.to_thread.run_sync(
                        os.replace, staging, destination
                    )
                    await volume.commit.aio()

        except BaseException:
            # Includes interrupted uploads. After publication, staging no longer
            # exists, so this never removes an already-published file.
            with anyio.CancelScope(shield=True):
                await anyio.to_thread.run_sync(
                    shutil.rmtree, staging, True
                )
            raise

        url = str(request.url_for("download_file", file_id=file_id))
        response.headers["Location"] = url
        response.headers["Cache-Control"] = "no-store"

        # Never return the stored password hash.
        return {
            "id": file_id,
            "filename": filename,
            "size_bytes": size,
            "delete_protected": delete_hash is not None,
            "download_url": url,
        }

    @api.head(
        "/files/{file_id}",
        name="head_file",
        response_class=Response,
        responses={404: {"description": "File not found"}},
    )
    @api.get(
        "/files/{file_id}",
        name="download_file",
        response_class=FileResponse,
        responses={
            200: {
                "content": {
                    "application/octet-stream": {
                        "schema": {"type": "string", "format": "binary"}
                    }
                }
            },
            206: {"description": "Partial content for a byte-range request"},
            400: {"description": "Malformed Range header"},
            404: {"description": "File not found"},
            416: {"description": "Requested range is not satisfiable"},
        },
    )
    async def download(
        file_id: FileID,
        range_header: Annotated[
            str | None,
            Header(alias="Range", description="Example: bytes=0-1048575"),
        ] = None,
    ):
        # FileResponse handles Range directly from the ASGI request headers.
        async with gate:
            meta = await anyio.to_thread.run_sync(read_meta, file_id)
            result = DownloadResponse(
                file_id,
                path=root / file_id / "data",
                filename=meta["filename"],
                media_type="application/octet-stream",
                headers={
                    "Cache-Control": "no-store",
                    "X-Content-Type-Options": "nosniff",
                },
            )
            readers[file_id] = readers.get(file_id, 0) + 1
            return result

    @api.delete(
        "/files/{file_id}",
        status_code=204,
        response_class=Response,
        responses={
            403: {"description": "Missing or incorrect deletion password"},
            404: {"description": "File not found"},
            409: {"description": "File is being downloaded; retry later"},
        },
    )
    async def delete(
        file_id: FileID,
        delete_password: DeletePassword = None,
    ):
        async with gate:
            meta = await anyio.to_thread.run_sync(read_meta, file_id)

        if meta["delete_hash"] is not None:
            if delete_password is None:
                raise HTTPException(403, "Deletion password required")

            try:
                async with hash_slots:
                    await anyio.to_thread.run_sync(
                        hasher.verify,
                        meta["delete_hash"],
                        delete_password,
                    )
            except VerifyMismatchError:
                raise HTTPException(403, "Incorrect deletion password") from None

        with anyio.CancelScope(shield=True):
            async with gate:
                # Another deletion may have completed during verification.
                await anyio.to_thread.run_sync(read_meta, file_id)

                # Prevent unlink/open races and interrupted active downloads.
                if readers.get(file_id, 0):
                    raise HTTPException(
                        409, "File is being downloaded; retry later"
                    )

                await anyio.to_thread.run_sync(
                    shutil.rmtree, root / file_id
                )
                await volume.commit.aio()

        return Response(status_code=204)

    return api