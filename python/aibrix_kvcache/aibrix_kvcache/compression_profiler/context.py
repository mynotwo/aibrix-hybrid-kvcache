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

from collections.abc import Iterable
from dataclasses import dataclass, replace

from .schema import ProfileMode


@dataclass(frozen=True, slots=True)
class TraceContext:
    """Explicit request metadata carried into storage and codec workers."""

    experiment_id: str
    run_id: str
    request_id: str
    mode: ProfileMode
    operation_id: str = ""
    parent_operation_id: str = ""
    dependency_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", ProfileMode(self.mode))
        dependencies = tuple(self.dependency_ids)
        if any(type(value) is not str for value in dependencies):
            raise TypeError("dependency_ids must contain only strings")
        object.__setattr__(self, "dependency_ids", dependencies)

    def child(
        self, operation_id: str, *, dependencies: Iterable[str] = ()
    ) -> "TraceContext":
        return replace(
            self,
            operation_id=operation_id,
            parent_operation_id=self.operation_id,
            dependency_ids=tuple(dependencies),
        )
