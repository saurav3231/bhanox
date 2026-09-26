"""Front-end: turning any input into a vector.

The public entry point is :func:`encode`, the byte 4-gram encoder. It has no
vocabulary, so it has no UNK token: every possible input, in any script, gets
an id.
"""

from __future__ import annotations

from bhanox.frontend.hashbind import BYTE_GRAM_ORDER, HashBind, encode_bytes, mix64

#: Public alias. The architecture calls this operation "encode"; the module
#: calls it ``encode_bytes`` to say what it consumes. The public API uses the
#: short name, so the short name is what is exported.
encode = encode_bytes

__all__ = ["BYTE_GRAM_ORDER", "HashBind", "encode", "encode_bytes", "mix64"]
