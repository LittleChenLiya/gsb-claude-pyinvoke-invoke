"""
Explicit per-channel (stdout/stderr) stream handling primitives.

This module implements the building blocks used by `.Runner` to move output
fragments from a subprocess' stdout/stderr pipes to their consumers (local
output streams, capture buffers and `.StreamWatcher` instances) *without*
merging the two channels and then guessing where data came from.

The key ideas:

- `StreamFragment` tags every piece of subprocess output with the real
  channel it was read from (``"stdout"`` or ``"stderr"``).
- `ChannelBuffer` is a bounded, per-channel handoff buffer. Each channel
  owns exactly one, so backpressure caused by a slow consumer only ever
  blocks the channel that consumer is reading from - never its sibling.
- End-of-file on a channel (`ChannelBuffer.close`) is finalized separately
  from, and independently of, the subprocess' own exit.

.. versionadded:: 3.0
"""

import threading
from collections import deque
from typing import Deque, NamedTuple, Optional


class StreamFragment(NamedTuple):
    """
    A single chunk of subprocess output, attributed to its real channel.

    .. versionadded:: 3.0
    """

    #: Name of the channel the data was read from: ``"stdout"`` or
    #: ``"stderr"``.
    channel: str
    #: The decoded text read from that channel.
    data: str


class ChannelClosed(Exception):
    """
    Raised when writing to a `ChannelBuffer` whose channel already hit EOF.

    .. versionadded:: 3.0
    """

    pass


class ChannelBuffer:
    """
    A bounded buffer of `StreamFragment` objects for a single channel.

    Each subprocess output channel (stdout/stderr) gets exactly one of
    these; they are deliberately independent so that a slow consumer only
    ever applies backpressure to its own channel:

    - `put` blocks (via a `threading.Condition` - never via sleeping)
      whenever the buffer is at its ``limit``, until the consumer makes
      room with `get`/`get_nowait`. A full stdout buffer therefore cannot
      stall the stderr channel, and vice versa.
    - `close` marks end-of-file for the channel. This is finalized
      separately from subprocess exit: readers may keep draining buffered
      fragments after EOF, and `get` only signals completion once the
      buffer is both closed *and* drained.

    .. versionadded:: 3.0
    """

    #: Default maximum number of unconsumed fragments held per channel.
    DEFAULT_LIMIT = 1024

    def __init__(self, channel: str, limit: int = DEFAULT_LIMIT) -> None:
        """
        Create a bounded buffer for ``channel``.

        :param channel: Channel name, e.g. ``"stdout"`` or ``"stderr"``.
        :param limit:
            Maximum number of unconsumed fragments held at once. Additional
            `put` calls block (or, for `put_nowait`, fail) while at this
            limit. Must be at least 1.
        """
        if limit < 1:
            raise ValueError("ChannelBuffer limit must be >= 1")
        #: The channel name this buffer is bound to.
        self.channel = channel
        #: Maximum number of unconsumed fragments held at once.
        self.limit = limit
        self._fragments: Deque[StreamFragment] = deque()
        self._eof = False
        self._cond = threading.Condition()

    def __len__(self) -> int:
        with self._cond:
            return len(self._fragments)

    @property
    def full(self) -> bool:
        """
        Whether the buffer is at its ``limit`` (puts will now block).
        """
        with self._cond:
            return len(self._fragments) >= self.limit

    @property
    def eof(self) -> bool:
        """
        Whether the channel has been marked as end-of-file via `close`.
        """
        with self._cond:
            return self._eof

    def put(self, fragment: StreamFragment) -> None:
        """
        Add ``fragment``, blocking while the buffer is at its limit.

        This is where per-channel backpressure lives: a slow consumer only
        ever causes *this* channel's producer to block here.

        :raises ChannelClosed: if the channel already reached EOF.
        """
        with self._cond:
            while len(self._fragments) >= self.limit and not self._eof:
                self._cond.wait()
            if self._eof:
                raise ChannelClosed(
                    "channel {!r} is at EOF".format(self.channel)
                )
            self._fragments.append(fragment)
            self._cond.notify_all()

    def put_nowait(self, fragment: StreamFragment) -> bool:
        """
        Add ``fragment`` only if there is room, returning success.

        Never blocks; returns ``False`` when the buffer is at its limit.

        :raises ChannelClosed: if the channel already reached EOF.
        """
        with self._cond:
            if self._eof:
                raise ChannelClosed(
                    "channel {!r} is at EOF".format(self.channel)
                )
            if len(self._fragments) >= self.limit:
                return False
            self._fragments.append(fragment)
            self._cond.notify_all()
            return True

    def get(self) -> Optional[StreamFragment]:
        """
        Remove and return the oldest fragment, blocking if necessary.

        Returns ``None`` only once the channel is at EOF *and* all
        previously buffered fragments have been drained - i.e. tail data is
        never dropped just because EOF arrived.
        """
        with self._cond:
            while not self._fragments and not self._eof:
                self._cond.wait()
            if self._fragments:
                fragment = self._fragments.popleft()
                self._cond.notify_all()
                return fragment
            return None

    def get_nowait(self) -> Optional[StreamFragment]:
        """
        Remove and return the oldest fragment, or ``None`` if none pending.

        Never blocks; may return ``None`` even before EOF (simply means the
        buffer is momentarily empty).
        """
        with self._cond:
            if self._fragments:
                fragment = self._fragments.popleft()
                self._cond.notify_all()
                return fragment
            return None

    def close(self) -> None:
        """
        Mark this channel as end-of-file and wake all waiters.

        Buffered fragments remain available via `get`/`get_nowait` after
        this call; EOF is finalized here, separately from subprocess exit.
        """
        with self._cond:
            self._eof = True
            self._cond.notify_all()
