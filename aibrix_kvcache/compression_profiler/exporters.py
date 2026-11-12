# Copyright 2024 The Aibrix Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# 	http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2026 The Aibrix Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from collections.abc import Sequence
from pathlib import Path

from .schema import TraceEvent


class JsonlExporter:
    """One file descriptor and one write per batch; owned by the worker."""

    def __init__(self, output: str | Path):
        self._output = output
        self._fd: int | None = None

    def write_batch(self, events: Sequence[TraceEvent]) -> None:
        data = b"".join(
            (json.dumps(event.to_record(), allow_nan=False) + "\n").encode()
            for event in events
        )
        if self._fd is None:
            self._fd = os.open(
                self._output, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600
            )
            os.fchmod(self._fd, 0o600)
        # A partial write cannot be retried as a full batch without duplicating
        # bytes. Keep its ring entries and invalidate the run instead.
        if os.write(self._fd, data) != len(data):
            raise OSError("incomplete trace batch write")

    def close(self) -> None:
        if self._fd is not None:
            fd, self._fd = self._fd, None
            os.close(fd)
